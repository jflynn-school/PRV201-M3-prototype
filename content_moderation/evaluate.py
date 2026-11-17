import argparse
import random
import time

import pandas as pd

from moderation import Message, build_default_pipeline
from moderation.dataset import TEST
from moderation.preprocess import preprocess


def summarize(name, preds, golds, hards, reviewed, latencies, cat_pred, cat_gold, review_seconds):
    n = len(golds)
    neg = [i for i in range(n) if not golds[i]]
    pos = [i for i in range(n) if golds[i]]
    auto = [i for i in range(n) if not reviewed[i]]
    hard = [i for i in range(n) if hards[i]]
    acc = lambda idx: (sum(preds[i] == golds[i] for i in idx) / len(idx)) if idx else float("nan")
    return {
        "approach": name,
        "accuracy": acc(range(n)),
        "false_positive_rate": sum(preds[i] for i in neg) / len(neg),
        "false_negative_rate": sum(not preds[i] for i in pos) / len(pos),
        "hard_case_accuracy": acc(hard),
        "human_review_rate": sum(reviewed) / n,
        "accuracy_of_auto_decisions": acc(auto),
        "category_accuracy": (sum(a == b for a, b in zip(cat_pred, cat_gold)) / n) if cat_pred else float("nan"),
        "mean_machine_latency_ms": sum(latencies) / n,
        "human_minutes": sum(reviewed) * review_seconds / 60,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--human-acc", type=float, default=0.95)
    ap.add_argument("--review-seconds", type=float, default=30)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    pipe = build_default_pipeline()
    golds = [cat != "none" for _, cat, _ in TEST]
    cats = [cat for _, cat, _ in TEST]
    hards = [h for _, _, h in TEST]
    n = len(TEST)

    def human(i, rng):
        return golds[i] if rng.random() < args.human_acc else (not golds[i])

    rng = random.Random(args.seed)
    manual = [human(i, rng) for i in range(n)]
    r_manual = summarize("manual_only", manual, golds, hards, [True] * n, [0] * n, [], [], args.review_seconds)

    auto_preds, auto_cats, lat = [], [], []
    for text, _, _ in TEST:
        t0 = time.perf_counter()
        c = pipe.classifier.predict(preprocess(text).text)
        lat.append((time.perf_counter() - t0) * 1000)
        auto_preds.append(c.category != "none")
        auto_cats.append(c.category)
    r_auto = summarize("fully_automated", auto_preds, golds, hards, [False] * n, lat, auto_cats, cats,
                       args.review_seconds)

    rng = random.Random(args.seed)
    hyb_preds, hyb_rev, hyb_lat, hyb_cats, rows = [], [], [], [], []
    for i, (text, cat, hard) in enumerate(TEST):
        res = pipe.moderate(Message(text, user_id=f"user{i}", target_id=f"target{i}"))
        reviewed = res.decision.action == "human_review"
        if reviewed:
            final = human(i, rng)
            pipe.resolve(res.message_id, "remove" if final else "allow", "simulated_moderator")
        else:
            final = res.decision.action == "auto_remove"
        hyb_preds.append(final)
        hyb_rev.append(reviewed)
        hyb_lat.append(res.latency_ms)
        hyb_cats.append(res.classification.category)
        rows.append({"text": text, "gold": cat, "predicted": res.classification.category,
                     "action": res.decision.action, "reason": res.decision.reason,
                     "final_remove": final, "should_remove": golds[i], "hard": hard})
    r_hyb = summarize("hybrid", hyb_preds, golds, hards, hyb_rev, hyb_lat, hyb_cats, cats, args.review_seconds)

    df = pd.DataFrame([r_manual, r_auto, r_hyb]).set_index("approach")
    pd.set_option("display.width", 200)
    print(f"\nTest set: {n} fictional messages ({sum(golds)} violations, {n - sum(golds)} benign, "
          f"{sum(hards)} hard cases). Simulated moderator accuracy: {args.human_acc:.0%}, "
          f"{args.review_seconds:.0f}s per review.\n")
    print(df.round(3).T.to_string())

    print("\nM2.6 goal check (machine-involved approaches):")
    for name in ("fully_automated", "hybrid"):
        r = df.loc[name]
        print(f"  {name:16s} accuracy>=85%: {'PASS' if r.accuracy >= .85 else 'FAIL'} ({r.accuracy:.1%}) | "
              f"FPR<5%: {'PASS' if r.false_positive_rate < .05 else 'FAIL'} ({r.false_positive_rate:.1%}) | "
              f"latency<5s: {'PASS' if r.mean_machine_latency_ms < 5000 else 'FAIL'} "
              f"({r.mean_machine_latency_ms:.1f} ms)")

    errs = pd.DataFrame(rows)
    auto_errs = errs[(errs.action != "human_review") & ((errs.action == "auto_remove") != errs.should_remove)]
    print(f"\nAuto-decisions that were wrong (hybrid): {len(auto_errs)}")
    for _, r in auto_errs.iterrows():
        print(f"  [{r.action}] gold={r.gold} pred={r.predicted} :: {r.text}")
    print("\nRouting reasons for reviewed messages:")
    print(errs[errs.action == "human_review"].reason.str.replace(r"[\d.]+%", "N%", regex=True)
          .value_counts().to_string())

    import os
    os.makedirs(args.out, exist_ok=True)
    df.round(4).to_csv(f"{args.out}/eval_summary.csv")
    errs.to_csv(f"{args.out}/eval_per_message.csv", index=False)
    try:
        with pd.ExcelWriter(f"{args.out}/eval_results.xlsx") as xw:
            df.round(4).to_excel(xw, sheet_name="summary")
            errs.to_excel(xw, sheet_name="per_message", index=False)
    except ImportError:
        print("(install openpyxl to also get an Excel export)")
    print(f"\nSaved results to {args.out}/")


if __name__ == "__main__":
    main()

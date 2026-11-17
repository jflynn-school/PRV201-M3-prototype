import itertools
import random
import string

from .preprocess import preprocess


def _expand(template, **fills):
    fields = [f for _, f, _, _ in string.Formatter().parse(template) if f]
    return [template.format(**dict(zip(fields, combo)))
            for combo in itertools.product(*[fills[f] for f in fields])]


_NAMES = ["Sam", "Priya", "Jordan", "Alex", "Maya", "Chris", "Dana", "Marcus"]


def _templates():
    subj = ["chemistry", "calculus", "history", "biology", "econ", "physics", "statistics", "psych"]
    data = {}
    data["none"] = (
        _expand("Anyone want to study for the {s} exam tonight?", s=subj)
        + _expand("Can someone share the {s} notes from today's lecture?", s=subj)
        + _expand("Congrats {n} on the {e}, well deserved!", n=_NAMES,
                  e=["internship offer", "scholarship", "award", "new job"])
        + _expand("The dining hall has {f} today, anyone want to go?", f=["tacos", "pasta", "curry", "pizza"])
        + [
            "I'm going to kill this presentation tomorrow, wish me luck!",
            "This homework is killing me, send help.",
            "I could murder a slice of pizza right now.",
            "You're so dead if you don't save me a seat, haha.",
            "lol you're the worst, I can't believe you got the last donut",
            "Ugh, that professor is the worst, I hated that lecture.",
            "The Muslim Student Association is hosting iftar on Friday, everyone is welcome.",
            "The women's soccer team won the conference title yesterday!",
            "The Disability Services office is open all week for accommodations.",
            "Our immigrant heritage night is Saturday, bring a dish from home.",
            "Pride Alliance meeting tonight at 7 in the student center.",
            "My email is [REDACTED_EMAIL] if you want the slides.",
            "Here is the syllabus: [REDACTED_URL]",
            "Call me at [REDACTED_PHONE] if you are running late to the meetup.",
            "Meet at [REDACTED_ADDRESS] for the study group.",
            "Thanks for the ride, you're a lifesaver!",
            "That movie was so bad it was good lol",
            "I hate Mondays so much",
            "This traffic is absolute garbage today",
            "The referee was terrible, worst call of the season.",
            "You crushed that interview, I'm so proud of you!",
            "Oh wow, another group project, my favorite.",
        ])
    ins = ["loser", "clown", "joke", "freak", "waste of space", "pathetic mess", "idiot", "moron", "creep", "fraud"]
    ins2 = ["idiot", "loser", "complete joke", "pathetic freak", "total embarrassment", "worthless clown"]
    intro = ["Honestly,", "Everyone thinks so:", "Lol,", "Seriously,", ""]
    data["harassment"] = (
        _expand("{i} you are such a {x}, nobody likes you.", i=intro, x=ins)
        + _expand("{i} stop posting, you {x}.", i=intro, x=ins2)
        + _expand("Everyone is laughing at you, you {x}.", x=ins2)
        + _expand("Why do you even bother showing up, {x}?", x=ins))
    groups = ["immigrants", "Muslims", "gay people", "women", "disabled people", "Jewish people",
              "Black students", "foreigners", "trans people", "old people"]
    data["hate_speech"] = (
        _expand("{g} are ruining this school and should be kicked out.", g=groups)
        + _expand("I can't stand {g}, they make me sick.", g=groups)
        + _expand("{g} don't belong here, go back where you came from.", g=groups)
        + _expand("All {g} are dangerous and stupid.", g=groups)
        + _expand("We need to get rid of all the {g} on campus.", g=groups)
        + _expand("{g} are subhuman and everyone knows it.", g=groups))
    data["threat"] = (
        _expand("I'm going to {v} you {w}, watch your back.", v=["hurt", "beat", "destroy", "wreck"],
                w=["after class", "tonight", "tomorrow", "when I find you"])
        + _expand("I know where you {l} and I'm coming for you.", l=["live", "sleep", "work", "park"])
        + _expand("{i} you're dead if I see you alone.", i=["Next time", "Tomorrow", "Tonight", "One more word and"])
        + _expand("I'm bringing a {wp} to the {pl} on Friday, you'd better not show up.",
                  wp=["knife", "gun", "bat", "weapon"], pl=["party", "dorm", "game", "meeting"])
        + _expand("You'll regret this, I swear I'll make you {r}.", r=["suffer", "pay", "cry", "bleed"])
        + _expand("Watch your back, {n}.", n=_NAMES)
        + ["I will find you and make you pay for this.", "Sleep with one eye open, I'm not done with you."])
    data["doxxing"] = (
        _expand("{n} lives at [REDACTED_ADDRESS], go pay a visit.", n=_NAMES)
        + _expand("{n}'s number is [REDACTED_PHONE], everyone call all night.", n=_NAMES)
        + _expand("Here is {n}'s email [REDACTED_EMAIL], flood the inbox.", n=_NAMES)
        + _expand("Posting the home address of the guy who reported us: [REDACTED_ADDRESS].", )
        + _expand("{n} works at the campus cafe, here is the cell [REDACTED_PHONE], make life miserable.", n=_NAMES)
        + _expand("Full name {n}, dorm room, phone [REDACTED_PHONE], enjoy.", n=_NAMES)
        + _expand("Found {n}'s address [REDACTED_ADDRESS] and parents' number [REDACTED_PHONE].", n=_NAMES)
        + _expand("Where does {n} live? We should all show up.", n=_NAMES))
    data["spam"] = (
        _expand("WIN a FREE {p}!!! Click [REDACTED_URL] now!!!", p=["iPhone", "gift card", "laptop", "vacation", "PlayStation"])
        + _expand("Make ${m} a week from home, no experience needed, visit [REDACTED_URL]", m=["2000", "5000", "10000"])
        + _expand("Buy cheap {t} now, DM me for deals [REDACTED_URL]", t=["followers", "likes", "essays", "pills", "watches"])
        + _expand("Limited time offer, act now and get {o}!!! [REDACTED_URL]",
                  o=["50% off", "free shipping", "a bonus", "double crypto"])
        + _expand("Earn crypto fast, guaranteed returns, join my {g}", g=["telegram", "group", "channel", "server"])
        + _expand("Congratulations, you have been selected! Claim your {p} at [REDACTED_URL]", p=["prize", "reward", "gift card"])
        + ["Hot singles in your area want to chat [REDACTED_URL]"])
    return data


def build_training_set(seed: int = 42, cap: int = 60):
    rng = random.Random(seed)
    texts, labels = [], []
    for label, items in _templates().items():
        items = sorted(set(items))
        rng.shuffle(items)
        for t in items[:cap]:
            texts.append(preprocess(t).text)
            labels.append(label)
    return texts, labels

TEST = [
    # --- benign (14) ---
    ("Anyone have the notes from Tuesday's chemistry lecture?", "none", False),
    ("Good luck on your finals everyone, you've got this!", "none", False),
    ("The library is open until 2am this week.", "none", False),
    ("I'm going to destroy this midterm tomorrow haha", "none", True),
    ("My roommate is going to murder me if I'm late again lol", "none", True),
    ("Oh great, another 8am class. Just what I always wanted.", "none", True),
    ("The Black Student Union is hosting a dinner Friday, all welcome.", "none", True),
    ("You're killing it in that role, honestly the best performance.", "none", True),
    ("my email is jsmith@school.edu if you want the slides", "none", False),
    ("Check out the syllabus here https://registrar.example.edu/syllabus", "none", False),
    ("Lol you absolute legend, thanks for covering my shift", "none", False),
    ("That referee was a disgrace, worst call I've ever seen", "none", False),
    ("We should hang out this weekend, are you free?", "none", False),
    ("Dying laughing at that meme omg", "none", True),
    # --- harassment (9) ---
    ("You're pathetic and everyone in this class knows it.", "harassment", False),
    ("Nobody likes you, just delete your account already.", "harassment", False),
    ("What a worthless clown, stop posting.", "harassment", False),
    ("You're such a freak, no wonder you eat alone.", "harassment", False),
    ("lol imagine being this ugly and still posting selfies", "harassment", True),
    ("Everyone is laughing at you behind your back, loser.", "harassment", False),
    ("You should be embarrassed to even show your face here.", "harassment", True),
    ("Stop talking, you idiot, no one cares what you think.", "harassment", False),
    ("Seriously, you're a joke and so is your whole project.", "harassment", False),
    # --- hate speech (9) ---
    ("Immigrants are ruining this school and should be kicked out.", "hate_speech", False),
    ("I can't stand gay people, they make me sick.", "hate_speech", False),
    ("Women are too stupid for engineering, they don't belong in this program.", "hate_speech", False),
    ("All Muslims are dangerous and shouldn't be allowed on campus.", "hate_speech", False),
    ("People like you should go back to where you came from.", "hate_speech", False),
    ("Disabled people are a burden on everyone here.", "hate_speech", False),
    ("Those people are subhuman, honestly.", "hate_speech", True),
    ("Some students only got in because of quotas, they don't deserve to be here.", "hate_speech", True),
    ("I hate every single one of those foreigners, they're vermin.", "hate_speech", False),
    # --- threats (10) ---
    ("I'm going to hurt you after class, watch your back.", "threat", False),
    ("I know where you live and I'm coming for you.", "threat", False),
    ("You'll regret this. I'll make sure you never walk out of that dorm.", "threat", False),
    ("Next time I see you alone you're dead.", "threat", False),
    ("I'm bringing a knife to the party Friday, you better not show up.", "threat", False),
    ("Keep talking and I'll end you, I mean it.", "threat", True),
    ("I will find you and make you pay for what you did.", "threat", False),
    ("Someone should beat you up and I'm happy to do it.", "threat", True),
    ("Sleep with one eye open tonight.", "threat", True),
    ("Everyone in that building should be scared tomorrow.", "threat", True),
    # --- doxxing (8) ---
    ("Her name is Dana Whitfield, she lives at 42 Maple Street Apt 3, go pay her a visit.", "doxxing", False),
    ("His number is 973-555-0142, everyone call him all night.", "doxxing", False),
    ("Here's Marcus's email marcus.t@school.edu and his class schedule, flood his inbox.", "doxxing", False),
    ("Posting the home address of the guy who reported us: 18 Birch Lane, Montclair NJ.", "doxxing", False),
    ("She works at the campus cafe on weekends, here's her cell 201-555-0188, make her life miserable.", "doxxing", False),
    ("Found his Instagram, his dorm, and his parents' info. Enjoy.", "doxxing", True),
    ("Anyone know where Priya lives? Asking so we can all show up.", "doxxing", True),
    ("Full name: Jake Roman, dorm: Hall B room 204, phone 862-555-0110.", "doxxing", False),
    # --- spam (8) ---
    ("WIN a FREE iPhone!!! Click http://bit.example/win now!!!", "spam", False),
    ("Make $5000 a week from home, no experience needed, visit http://cash.example", "spam", False),
    ("Buy 10k followers cheap, DM me now", "spam", False),
    ("Cheap essays written for you, 100% undetectable, http://essays.example", "spam", False),
    ("Hot singles in your area want to chat http://dates.example", "spam", False),
    ("Limited time crypto giveaway, send 1 coin get 2 back", "spam", False),
    ("Congratulations you've been selected for a gift card, claim at http://gift.example", "spam", False),
    ("Follow me for a follow back! f4f l4l", "spam", True),
]

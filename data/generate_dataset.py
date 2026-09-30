"""
Phase 2, step 2: "Build a labeled dataset: Write 200+ example prompts across
all three tiers. Label each one by hand."

Hand-writing 200+ genuinely distinct prompts one at a time doesn't scale, so
this generates them from ~10 hand-written templates per tier (the same
templates the guide's own tier definitions describe - extraction/reformatting
for Tier 1, summarization/classification for Tier 2, multi-step
reasoning/creative generation for Tier 3) crossed with varied entities, topics
and numbers. Every template is authored once per tier, so the *label* is
still a deliberate per-template decision, not an algorithm guessing at
difficulty - only the filler values are randomized. Review data/labeled_dataset.csv
and edit/relabel individual rows before training if any read wrong to you;
that hand review is what step 2 is really asking for.

Run with:  python -m data.generate_dataset
"""

import csv
import random
from pathlib import Path

random.seed(42)

DATA_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = DATA_DIR / "labeled_dataset.csv"

PER_TIER_TARGET = 150  # generated per tier before dedup; final count may be a bit lower

# --- shared entity pools --------------------------------------------------

COUNTRIES = [
    "France", "Japan", "Brazil", "Egypt", "Canada", "Kenya", "Norway",
    "Thailand", "Peru", "Portugal", "Vietnam", "Morocco", "Chile", "Poland",
    "Greece", "Indonesia", "Nigeria", "Finland", "Argentina", "South Korea",
    "Turkey", "Sweden", "Ireland", "New Zealand",
]
WORD_PAIRS = [  # (word, antonym) - used for plural and opposite prompts
    ("cactus", "ordinary"), ("octopus", "typical"), ("analysis", "synthesis"),
    ("criterion", "consequence"), ("index", "content"), ("matrix", "outline"),
    ("hypothesis", "certainty"), ("formula", "improvisation"),
    ("syllabus", "footnote"), ("phenomenon", "commonplace"),
    ("expand", "shrink"), ("ancient", "modern"), ("generous", "stingy"),
    ("transparent", "opaque"), ("abundant", "scarce"), ("cautious", "reckless"),
    ("fragile", "sturdy"), ("genuine", "fake"), ("rigid", "flexible"),
    ("optimistic", "pessimistic"),
]
FIELDS = [
    ("email address", "Contact Raj at raj418060@gmail.com for details."),
    ("phone number", "Call our support line at 011-4567-8901 for help."),
    ("order ID", "Your order #ORD-88214 has shipped."),
    ("price", "The item costs Rs. 2,499 including tax."),
    ("date", "The meeting is scheduled for 14 March 2027."),
    ("invoice number", "Please reference invoice INV-2024-0091 in your reply."),
    ("tracking number", "Your package tracking ID is TRK-556231JP."),
    ("employee ID", "New hires are assigned an ID like EMP-4471."),
    ("zip code", "Our warehouse is located at 201301 in Noida."),
    ("flight number", "Boarding begins for flight AI-302 at gate 14."),
    ("website URL", "More details are available at www.example-store.com."),
    ("username", "Log in to the portal as user raj_r to continue."),
]
UNITS = [
    ("km", "miles", 42), ("kg", "lb", 18), ("celsius", "fahrenheit", 30),
    ("liters", "gallons", 10), ("meters", "feet", 5), ("bytes", "kilobytes", 4096),
    ("hours", "minutes", 3), ("USD", "INR", 50), ("cm", "inches", 25),
    ("watts", "kilowatts", 1500),
]
EVENTS = [
    "the first moon landing", "the fall of the Berlin Wall",
    "the invention of the World Wide Web", "India's independence",
    "the founding of the United Nations", "the release of the first iPhone",
    "the signing of the Treaty of Versailles", "the first modern Olympics",
    "the launch of Sputnik 1", "the abolition of slavery in the US",
]
NAME_PAIRS = [
    ("Rajput", "Raj"), ("Sharma", "Anita"), ("Iyer", "Karthik"),
    ("Khan", "Sana"), ("Das", "Priya"), ("Mehta", "Vivek"),
    ("Nair", "Divya"), ("Gupta", "Rohan"), ("Reddy", "Meera"),
    ("Chowdhury", "Arjun"), ("Bose", "Ishaan"), ("Verma", "Neha"),
]
CAP_TEXTS = [
    "welcome to the quarterly all-hands meeting",
    "please submit your timesheet by friday",
    "the server will be down for maintenance tonight",
    "thank you for your recent purchase",
    "this offer expires at the end of the month",
    "our office will be closed for the holiday",
    "the new update improves battery life significantly",
    "your subscription renews automatically next week",
]

# --- Tier 2 pools ----------------------------------------------------------

PARAGRAPHS = [
    "The company reported record quarterly revenue driven by strong cloud "
    "services growth, though operating margins narrowed due to increased "
    "infrastructure spending.",
    "Researchers found that the new battery chemistry retains 92% of its "
    "capacity after 1,000 charge cycles, a marked improvement over the "
    "previous generation.",
    "City officials announced a phased rollout of the new metro line, "
    "with the first segment expected to open next spring after repeated "
    "construction delays.",
    "The study surveyed 4,000 remote workers and found that flexible "
    "hours correlated more strongly with reported job satisfaction than "
    "salary did.",
    "The hospital's new triage system cut average emergency room wait "
    "times by nearly a third, though staff say it added extra paperwork "
    "per patient.",
    "A recent audit found that the recycling program diverted 60% more "
    "waste from landfills than projected, but collection costs also rose.",
    "The startup's pivot from hardware to software subscriptions boosted "
    "gross margins but slowed customer acquisition in its first two quarters.",
    "Local farmers reported a stronger-than-expected harvest this year, "
    "attributed to a milder monsoon season and new irrigation subsidies.",
]
REVIEWS = [
    "The product works fine but shipping took way too long.",
    "Absolutely loved it - arrived early and exceeded expectations.",
    "Package arrived damaged and support was slow to respond.",
    "It's okay, does what it says, nothing special.",
    "Fantastic build quality, but the price feels a bit steep.",
    "Customer service was rude and unhelpful when I asked for a refund.",
    "Exactly as described, works perfectly, would buy again.",
    "Instructions were confusing and the app kept crashing on setup.",
]
TOPICS = [
    "microservices architecture", "remote work", "electric vehicles",
    "open-source software", "standardized testing", "nuclear energy",
    "four-day work weeks", "cryptocurrency payments", "AI code assistants",
    "gig economy platforms", "vertical farming", "universal basic income",
]
CONCEPT_PAIRS = [
    ("REST", "GraphQL"), ("SQL", "NoSQL"), ("supervised learning", "unsupervised learning"),
    ("TCP", "UDP"), ("monolith", "microservices"), ("Python", "Go"),
    ("Docker", "Kubernetes"), ("Redis", "Memcached"), ("Kafka", "RabbitMQ"),
    ("agile", "waterfall"),
]
ITEMS_LISTS = [
    "apple, carrot, salmon, broccoli, tuna, banana",
    "hammer, screwdriver, wrench, pliers, drill",
    "Python, HTML, PostgreSQL, React, Docker",
    "lion, sparrow, salmon, cobra, dolphin",
    "sedan, bicycle, ferry, helicopter, scooter",
    "violin, drum, flute, guitar, trumpet",
    "Visa, MasterCard, PayPal, bank transfer, cash",
]
JSON_SCHEMAS = [
    '{"name": "string", "age": "integer", "is_active": "boolean"}',
    '{"product_id": "string", "price": "number", "in_stock": "boolean"}',
    '{"title": "string", "pages": "integer", "genres": "array"}',
    '{"user": "string", "score": "number", "passed": "boolean"}',
    '{"city": "string", "population": "integer", "country": "string"}',
    '{"order_id": "string", "items": "array", "total": "number"}',
]
FORMAL_PARAGRAPHS = [
    "hey just wanted to check if you got my last email, let me know when you can",
    "so basically the deadline moved up and we need this done way sooner now",
    "thanks a ton for the help earlier, really saved us on the demo",
    "can we push the call to tomorrow, something came up on my end",
    "just a heads up the numbers look off, might want to double check them",
    "no worries if you can't make it, we'll fill you in after",
]
ASPECTS = ["cost", "scalability", "developer experience", "performance", "security"]

# --- Tier 3 pools ------------------------------------------------------------

REASONING_THEMES = [
    "a programmer debugging code late at night", "a monsoon evening in Noida",
    "an astronaut's first steps on Mars", "a chess match between old rivals",
    "a train station at dawn", "a lighthouse keeper's last night on the job",
    "a street musician playing in the rain", "an old library after closing time",
]
SYSTEMS = [
    "a ride-sharing app's real-time matching service",
    "a high-traffic e-commerce checkout flow",
    "a multi-tenant SaaS billing system",
    "a video streaming recommendation pipeline",
    "a food delivery app's live order tracking",
    "a bank's fraud detection pipeline",
    "a social media feed ranking service",
    "an IoT fleet of smart thermostats",
]
DECISIONS = [
    "whether to build their own auth system or use a third-party provider",
    "whether to migrate from a monolith to microservices before their Series A",
    "whether to prioritize a mobile app or a web app for their MVP",
    "whether to hire senior engineers now or juniors and scale later",
    "whether to self-host their infrastructure or stay on the cloud",
    "whether to charge a flat subscription or usage-based pricing",
]
SCENARIOS = [
    "using AI to screen job applicants automatically",
    "deploying facial recognition in public transit stations",
    "using predictive policing algorithms in city budgeting",
    "using AI-generated content without disclosing it to readers",
    "collecting biometric data from employees for attendance tracking",
    "using students' academic data to train a commercial AI product",
]
LOGIC_PUZZLES = [
    "if all Bloops are Razzies and all Razzies are Lazzies, are all Bloops "
    "definitely Lazzies?",
    "three friends split a bill evenly, but one paid the whole thing "
    "upfront and was reimbursed by only one other friend - who still owes "
    "money, and how much, if the bill was 1200?",
    "a function is supposed to return true only when a list is sorted, "
    "but it returns true for an empty list and a single-element list too "
    "- is that a bug?",
    "a clock shows 3:15 - what is the angle between the hour and minute "
    "hands?",
]

# --- Tier 1: simple - extraction, reformatting, basic Q&A -----------------

def gen_tier1(n: int) -> list[str]:
    prompts = []
    for _ in range(n):
        choice = random.randint(1, 9)
        if choice == 1:
            field, text = random.choice(FIELDS)
            prompts.append(f"Extract the {field} from this text: '{text}'")
        elif choice == 2:
            prompts.append(
                f"What is the capital of {random.choice(COUNTRIES)}? Answer in one word."
            )
        elif choice == 3:
            last, first = random.choice(NAME_PAIRS)
            prompts.append(f"Reformat this name from '{last}, {first}' to '{first} {last}'.")
        elif choice == 4:
            a, b, val = random.choice(UNITS)
            prompts.append(f"Convert {val} {a} to {b}.")
        elif choice == 5:
            word, _ = random.choice(WORD_PAIRS)
            prompts.append(f"What is the plural of '{word}'?")
        elif choice == 6:
            prompts.append(f"What year did {random.choice(EVENTS)} happen?")
        elif choice == 7:
            prompts.append(f"Capitalize the following text: '{random.choice(CAP_TEXTS)}'")
        elif choice == 8:
            a, b = random.randint(2, 500), random.randint(2, 500)
            prompts.append(f"What is {a} + {b}?")
        else:
            _, antonym_source = random.choice(WORD_PAIRS)
            word2, _ = random.choice(WORD_PAIRS)
            prompts.append(f"What is the opposite of '{word2}'?")
    return prompts


# --- Tier 2: moderate - summarization, classification, structured analysis

def gen_tier2(n: int) -> list[str]:
    prompts = []
    for _ in range(n):
        choice = random.randint(1, 8)
        if choice == 1:
            prompts.append(
                f"Summarize the following in two sentences: '{random.choice(PARAGRAPHS)}'"
            )
        elif choice == 2:
            prompts.append(
                "Classify the sentiment of this review as positive, negative, "
                f"or neutral: '{random.choice(REVIEWS)}'"
            )
        elif choice == 3:
            prompts.append(f"List three pros and three cons of {random.choice(TOPICS)}.")
        elif choice == 4:
            prompts.append(
                f"Given this JSON schema, generate one valid example object: {random.choice(JSON_SCHEMAS)}"
            )
        elif choice == 5:
            prompts.append(f"Categorize these items into groups: {random.choice(ITEMS_LISTS)}")
        elif choice == 6:
            a, b = random.choice(CONCEPT_PAIRS)
            prompts.append(f"Explain the difference between {a} and {b} in 3-4 sentences.")
        elif choice == 7:
            prompts.append(
                f"Rewrite this paragraph in a more formal tone: '{random.choice(FORMAL_PARAGRAPHS)}'"
            )
        else:
            a, b = random.choice(CONCEPT_PAIRS)
            prompts.append(
                f"Compare {a} and {b} in terms of {random.choice(ASPECTS)}, in about 100 words."
            )
    return prompts


# --- Tier 3: complex - multi-step reasoning, creative generation, judgment

def gen_tier3(n: int) -> list[str]:
    prompts = []
    for _ in range(n):
        choice = random.randint(1, 8)
        if choice == 1:
            speed1, speed2 = random.randint(40, 80), random.randint(60, 100)
            distance = random.randint(200, 500)
            prompts.append(
                f"A train leaves City A at {speed1} km/h heading to City B, "
                f"{distance}km away. Another train leaves City B at the same "
                f"time heading to City A at {speed2} km/h. How far from City "
                "A do they meet, and how long does it take? Show your "
                "reasoning step by step."
            )
        elif choice == 2:
            prompts.append(
                f"Write a short, original four-line poem about {random.choice(REASONING_THEMES)}. "
                "Do not reuse lines from any existing published poem."
            )
        elif choice == 3:
            a, b = random.choice(CONCEPT_PAIRS)
            prompts.append(
                f"Compare and contrast {a} and {b} for {random.choice(SYSTEMS)}, "
                "and recommend one with justification."
            )
        elif choice == 4:
            prompts.append(
                f"Design a system architecture for {random.choice(SYSTEMS)}, "
                "considering scalability, cost, and maintainability. Explain "
                "your key trade-offs."
            )
        elif choice == 5:
            prompts.append(
                f"You are advising a startup on {random.choice(DECISIONS)}. "
                "Weigh the trade-offs and give a recommendation with reasoning."
            )
        elif choice == 6:
            prompts.append(
                f"Analyze the ethical implications of {random.choice(SCENARIOS)} "
                "and provide a nuanced recommendation."
            )
        elif choice == 7:
            prompts.append(
                f"Write a short story (150-200 words) about {random.choice(REASONING_THEMES)}, "
                "with a twist ending."
            )
        else:
            prompts.append(
                f"Debug this logic: {random.choice(LOGIC_PUZZLES)} Explain your "
                "reasoning at each step."
            )
    return prompts


def main() -> None:
    rows: list[tuple[str, int]] = []
    for tier, gen_fn in ((1, gen_tier1), (2, gen_tier2), (3, gen_tier3)):
        for prompt in gen_fn(PER_TIER_TARGET):
            rows.append((prompt, tier))

    # de-dupe (random sampling can repeat) while keeping order roughly stable
    seen = set()
    deduped = []
    for prompt, tier in rows:
        if prompt not in seen:
            seen.add(prompt)
            deduped.append((prompt, tier))

    random.shuffle(deduped)

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["prompt", "tier"])
        writer.writerows(deduped)

    counts = {1: 0, 2: 0, 3: 0}
    for _, tier in deduped:
        counts[tier] += 1
    print(f"wrote {len(deduped)} labeled prompts to {OUTPUT_PATH}")
    print(f"tier 1 (simple): {counts[1]}  |  tier 2 (moderate): {counts[2]}  |  tier 3 (complex): {counts[3]}")


if __name__ == "__main__":
    main()

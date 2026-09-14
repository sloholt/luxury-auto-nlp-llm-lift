import os
import sys
import json
import random
import itertools
from collections import Counter

import pandas as pd

TASK_A_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Task_A")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

sys.path.insert(0, TASK_A_DIR)

from extract_candidates import load_and_clean_data, DATA_PATH  

TOP10_BRANDS_PATH = os.path.join(TASK_A_DIR, "output", "final_brands_top10.csv")

MODEL = "claude-sonnet-5"
PRICE_PER_MTOK_INPUT = 2.00
PRICE_PER_MTOK_OUTPUT = 10.00
BATCH_SIZE = 20


def load_top10_brands(path=TOP10_BRANDS_PATH):
    """Read Task A's own ranked top-10 output, same convention as Task B, so this
    script always reflects whatever Task A currently reports as final. Falls back
    to the hardcoded list if Task A's output CSV isn't present in this checkout."""
    if os.path.exists(path):
        return pd.read_csv(path)["candidate"].tolist()
    return ["BMW", "Acura", "Cadillac", "Audi", "Lexus", "GM",
            "Infiniti", "Mercedes-Benz", "Volvo", "Nissan"]


SYSTEM_PROMPT = """You are annotating forum posts from a luxury-car discussion board for a competitive \
analysis. For each post, decide which of these 10 brands the post is GENUINELY, SUBSTANTIVELY \
discussing, based on meaning and context -- not simple keyword matching.

Brands: {brands}

Rules:
- Count a brand if it is referenced by name, by a model/trim/chassis code, or by common slang \
(e.g. "bimmer"=BMW, "benz"=Mercedes-Benz, "e46"=BMW, "tl"=Acura, "g35"=Infiniti), AND the post is \
actually saying something about that brand or a product of it.
- Comparisons, negations, and sarcasm still count as genuine discussion of a brand \
("I'd never buy an X over a Y" -> both X and Y count).
- A brand quoted from another user within the post still counts if named.
- Do NOT count a brand if the reference is a text-corruption artifact (a brand name spliced \
mid-word into an unrelated word, e.g. "soph[toyota]icated") rather than a real mention.
- Do NOT map non-listed brands (e.g. Kia, Hyundai, Toyota if not in the list) onto a listed brand.
- Multiple models of the SAME brand in one post count as ONE mention of that brand, not multiple.
- If no listed brand is genuinely discussed, return an empty list.

Respond with ONLY a JSON object: {{"post_id": <id>, "brands": [<brand names from the list, exact \
spelling>]}}. No other text."""


def build_batch_prompt(posts_batch):
    items = [{"post_id": p["post_id"], "text": p["text"][:2000]} for p in posts_batch]
    return ("Classify each of the following posts. Respond with ONLY a JSON array, one object per "
            "post, each of the form {\"post_id\": <id>, \"brands\": [...]}, in the same order given.\n\n"
            + json.dumps(items, ensure_ascii=False))


def classify_batch_live(client, posts_batch, brands, usage_log):
    resp = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        system=SYSTEM_PROMPT.format(brands=", ".join(brands)),
        messages=[{"role": "user", "content": build_batch_prompt(posts_batch)}],
    )
    usage_log.append({"input_tokens": resp.usage.input_tokens,
                       "output_tokens": resp.usage.output_tokens})
    text = resp.content[0].text.strip()
    if text.startswith("```"):
        text = text.strip("`").split("\n", 1)[-1]
    return json.loads(text)


def estimate_cost(usage_log):
    in_tok = sum(u["input_tokens"] for u in usage_log)
    out_tok = sum(u["output_tokens"] for u in usage_log)
    cost = in_tok / 1e6 * PRICE_PER_MTOK_INPUT + out_tok / 1e6 * PRICE_PER_MTOK_OUTPUT
    return {"model": MODEL, "calls": len(usage_log), "input_tokens": in_tok,
            "output_tokens": out_tok, "total_tokens": in_tok + out_tok,
            "est_cost_usd": round(cost, 4)}



TOY_SET = [
    {"id": "t1", "text": "I love my BMW 3 series, best car I've owned.",
     "expected": {"BMW"}, "tests": "plain genuine mention"},
    {"id": "t2", "text": "stat toyota ics show sedans are declining, unrelated to brands really.",
     "expected": set(), "tests": "corruption artifact must NOT be read as a real mention"},
    {"id": "t3", "text": "I'd never buy a Cadillac over an Audi, the A4 is just better built.",
     "expected": {"Cadillac", "Audi"}, "tests": "comparison/negation -- both genuinely discussed"},
    {"id": "t4", "text": "someone quoted \"the bmw is overrated\" but I actually drive a Lexus GS and love it.",
     "expected": {"BMW", "Lexus"}, "tests": "nested quote still names BMW; own brand also named"},
    {"id": "t5", "text": "yeah right, like a Volvo is going to out-handle anything. lol.",
     "expected": {"Volvo"}, "tests": "sarcasm -- brand still substantively discussed"},
    {"id": "t6", "text": "my e46 has 120k miles and still drives like new, way better than my buddy's TL.",
     "expected": {"BMW", "Acura"}, "tests": "slang/model-code resolution (e46->BMW, TL->Acura)"},
    {"id": "t7", "text": "does anyone know a good mechanic in Dallas? not car-brand related.",
     "expected": set(), "tests": "no brand present"},
    {"id": "t8", "text": "the g35 and the is300 are both great but I'd pick the Infiniti for the interior.",
     "expected": {"Infiniti", "Lexus"}, "tests": "two model codes, one brand also spelled out"},
    {"id": "t9", "text": "gm's been struggling but the cts-v from Cadillac is a genuine performance car.",
     "expected": {"GM", "Cadillac"}, "tests": "parent + subsidiary named separately (both count)"},
    {"id": "t10", "text": "my cousin works at a benz dealership and says the C-class outsells everything.",
     "expected": {"Mercedes-Benz"}, "tests": "slang + model code, single brand"},
    {"id": "t11", "text": "nissan altima or a maxima, which one rides smoother on the highway?",
     "expected": {"Nissan"}, "tests": "two models same brand -> single brand, not double"},
    {"id": "t12", "text": "kia and hyundai aren't luxury brands so they're irrelevant to this thread anyway.",
     "expected": set(), "tests": "non-listed brands should not be force-mapped onto a listed one"},
]


def validate_toy(client, brands):
    usage_log = []
    rows = []
    for t in TOY_SET:
        result = classify_batch_live(client, [{"post_id": t["id"], "text": t["text"]}], brands, usage_log)
        rec = result[0] if isinstance(result, list) else result
        got = set(rec["brands"])
        rows.append({"id": t["id"], "text": t["text"], "expected": sorted(t["expected"]),
                     "got": sorted(got), "match": got == t["expected"], "tests": t["tests"]})
    df = pd.DataFrame(rows)
    print(df[["id", "match", "tests"]].to_string(index=False))
    print(f"\n{df['match'].sum()}/{len(df)} exact matches")
    print(estimate_cost(usage_log))
    return df


def compute_lift_matrix(classifications, brands):
    n = len(classifications)
    brand_post_count = Counter()
    pair_post_count = Counter()
    for c in classifications:
        present = set(b for b in c["brands"] if b in brands)
        for b in present:
            brand_post_count[b] += 1
        for a, b in itertools.combinations(sorted(present), 2):
            pair_post_count[(a, b)] += 1

    matrix = pd.DataFrame(index=brands, columns=brands, dtype=float)
    for a, b in itertools.combinations(brands, 2):
        n_a, n_b = brand_post_count[a], brand_post_count[b]
        n_both = pair_post_count.get((a, b), 0)
        lift = 0.0 if n_a == 0 or n_b == 0 or n_both == 0 else (n * n_both) / (n_a * n_b)
        matrix.loc[a, b] = lift
        matrix.loc[b, a] = lift
    return matrix, brand_post_count, pair_post_count


def compute_pair_table(brand_post_count, pair_post_count, n, brands):
    rows = []
    for a, b in itertools.combinations(brands, 2):
        n_a, n_b = brand_post_count[a], brand_post_count[b]
        n_both = pair_post_count.get((a, b), 0)
        lift = 0.0 if n_a == 0 or n_b == 0 or n_both == 0 else (n * n_both) / (n_a * n_b)
        rows.append({"brand_a": a, "brand_b": b, "n_a": n_a, "n_b": n_b, "n_both": n_both,
                     "support_both": round(n_both / n, 5),
                     "expected_n_both_if_independent": round((n_a * n_b) / n, 2),
                     "lift": round(lift, 3)})
    return pd.DataFrame(rows).sort_values("lift", ascending=False).reset_index(drop=True)


def main():
    import anthropic
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    client = anthropic.Anthropic()  

    top10 = load_top10_brands()
    print(f"Top 10 brands (from Task A): {top10}")

    print("\n=== Toy validation -- run this first, every time you touch the prompt ===")
    toy_df = validate_toy(client, top10)
    toy_df.to_csv(os.path.join(OUTPUT_DIR, "toy_validation_results.csv"), index=False)
    if not toy_df["match"].all():
        print("\nWARNING: not all toy cases passed. Review before running the full corpus.")

    df = load_and_clean_data(os.path.join(TASK_A_DIR, DATA_PATH))
    df = df.reset_index(drop=True)
    N = len(df)
    print(f"\nCorpus size after cleaning: {N} messages")

    posts = [{"post_id": i, "text": row.message} for i, row in df.iterrows()]

    usage_log = []
    classifications = []
    for i in range(0, len(posts), BATCH_SIZE):
        batch = posts[i:i + BATCH_SIZE]
        classifications.extend(classify_batch_live(client, batch, top10, usage_log))
        if (i // BATCH_SIZE) % 25 == 0:
            print(f"  ...{i + len(batch)}/{len(posts)} messages classified")

    class_df = pd.DataFrame([{"post_id": c["post_id"], "brands": "|".join(c["brands"])} for c in classifications])
    class_df.to_csv(os.path.join(OUTPUT_DIR, "semantic_classifications.csv"), index=False)

    cost_report = estimate_cost(usage_log)
    pd.DataFrame([cost_report]).to_csv(os.path.join(OUTPUT_DIR, "usage_report.csv"), index=False)
    print(f"\n{cost_report}")

    lift_matrix, brand_post_count, pair_post_count = compute_lift_matrix(classifications, top10)
    lift_matrix.to_csv(os.path.join(OUTPUT_DIR, "semantic_lift_matrix.csv"))

    pair_table = compute_pair_table(brand_post_count, pair_post_count, N, top10)
    pair_table.to_csv(os.path.join(OUTPUT_DIR, "semantic_pair_lift.csv"), index=False)

    print("\nPer-brand message presence (semantic):")
    print(pd.Series(brand_post_count).sort_values(ascending=False))
    print("\nTop 10 brand pairs by semantic lift:")
    print(pair_table.head(10).to_string(index=False))

    
    random.seed(42)
    sample_ids = random.sample(range(N), min(30, N))
    class_lookup = {c["post_id"]: c["brands"] for c in classifications}
    sample_rows = [{"post_id": pid, "message": df.loc[pid, "message"][:400],
                    "semantic_brands": "|".join(class_lookup.get(pid, []))} for pid in sample_ids]
    pd.DataFrame(sample_rows).to_csv(os.path.join(OUTPUT_DIR, "manual_inspection_sample.csv"), index=False)
    print(f"\nWrote manual_inspection_sample.csv -- read this and characterize the errors you find.")

    print(f"\nAll outputs written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

import re
import pandas as pd

BRAND_ANCHORS = [
    "bmw",
    "acura",
    "audi",
    "honda",
    "lexus",
    "infiniti",
    "volvo",
    "mercedes-benz",
    "lincoln",
    "cadillac",
    "nissan",
    "gm",
    "toyota",
    "pontiac",
    "mazda",
    "volkswagen",
]


def merge_substring_variants(table: pd.DataFrame, anchors=BRAND_ANCHORS):
    table = table.copy()
    table["canonical_brand"] = None

    for anchor in anchors:
        pattern = re.compile(rf"\b{re.escape(anchor)}\b", re.IGNORECASE)
        mask = table["candidate"].apply(lambda c: bool(pattern.search(c)))
        table.loc[mask, "canonical_brand"] = anchor

    merged_rows = []
    for anchor in anchors:
        sub = table[table["canonical_brand"] == anchor]
        if sub.empty:
            continue
        merged_rows.append(
            {
                "candidate": anchor,
                "raw_count": sub["raw_count"].sum(),
                "message_reach": sub["message_reach"].sum(),
                "variant_count": len(sub),
            }
        )

    unresolved = table[table["canonical_brand"].isna()].drop(columns="canonical_brand")
    merged = pd.DataFrame(merged_rows).sort_values("message_reach", ascending=False)
    return merged, unresolved


def compute_brand_presence(
    df: pd.DataFrame, anchors=BRAND_ANCHORS, text_col="message"
) -> pd.DataFrame:
    presence = pd.DataFrame(index=df.index)
    for anchor in anchors:
        pattern = re.compile(rf"\b{re.escape(anchor)}\b", re.IGNORECASE)
        presence[anchor] = df[text_col].str.contains(pattern, na=False)
    return presence


def lift_assign_candidate(
    df: pd.DataFrame,
    candidate: str,
    brand_presence: pd.DataFrame,
    text_col="message",
    min_lift_gap=1.5,
) -> dict:
    N = len(df)
    cand_pattern = re.compile(rf"\b{re.escape(candidate)}\b", re.IGNORECASE)
    cand_mask = df[text_col].str.contains(cand_pattern, na=False)
    n_cand = cand_mask.sum()
    if n_cand == 0:
        return {
            "candidate": candidate,
            "best_brand": None,
            "lift": 0,
            "confident": False,
        }

    lifts = {}
    for brand in brand_presence.columns:
        n_brand = brand_presence[brand].sum()
        n_both = (cand_mask & brand_presence[brand]).sum()
        lifts[brand] = (
            0.0 if n_brand == 0 or n_both == 0 else (N * n_both) / (n_cand * n_brand)
        )

    ranked = sorted(lifts.items(), key=lambda kv: kv[1], reverse=True)
    best_brand, best_lift = ranked[0]
    runner_up_lift = ranked[1][1] if len(ranked) > 1 else 0

    confident = best_lift > 0 and (
        runner_up_lift == 0 or best_lift / runner_up_lift >= min_lift_gap
    )
    return {
        "candidate": candidate,
        "best_brand": best_brand,
        "lift": round(best_lift, 2),
        "runner_up_lift": round(runner_up_lift, 2),
        "confident": confident,
    }


# Candidates below this message_reach are long-tail noise (near-singleton
# mentions) and are not worth an LLM call either for brand-variant detection
# or attribute discovery. 30 was chosen by inspecting the reach distribution:
# it captures known brand slang/abbreviations that a smaller cutoff would miss
# (e.g. "bimmer" reach=55, "caddy" reach=83, "mb" reach=86 -- all below the
# original top-50-by-reach cutoff this replaces) while keeping the LLM-review
# set to a manageable ~250 candidates. See report notes.
MIN_REACH_FOR_REVIEW = 30


def main():
    from extract_candidates import load_and_clean_data, DATA_PATH

    df = load_and_clean_data(DATA_PATH)
    table = pd.read_csv("output/candidate_frequencies_clean.csv")

    merged, unresolved = merge_substring_variants(table)
    merged.to_csv("output/brands_merged.csv", index=False)
    print(
        f"Merged {len(table) - len(unresolved) - len(merged)} variant rows into {len(merged)} brand rows"
    )

    presence = compute_brand_presence(df)
    review_candidates = (
        unresolved[unresolved["message_reach"] >= MIN_REACH_FOR_REVIEW]
        .sort_values("message_reach", ascending=False)["candidate"]
        .tolist()
    )

    results = [
        lift_assign_candidate(df, cand, presence) for cand in review_candidates
    ]
    results_df = pd.DataFrame(results)
    results_df.to_csv("output/lift_assignments.csv", index=False)

    # Co-occurrence lift alone is not trustworthy evidence of a brand
    # reference -- e.g. "forum" showed lift=3.89 toward lincoln and "fwd"
    # showed lift=3.87 toward nissan in an earlier run, both spurious
    # coincidences in a corpus this size. So "confident" lift is kept only
    # as context passed to the LLM, not as an auto-merge gate: every
    # candidate above the reach threshold goes to LLM adjudication.
    print(f"Candidates sent to LLM adjudication: {len(results_df)}")
    results_df.to_csv("output/needs_llm_review.csv", index=False)


if __name__ == "__main__":
    main()

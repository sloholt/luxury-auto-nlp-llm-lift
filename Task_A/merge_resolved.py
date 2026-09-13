"""
merge_resolved.py

Takes the LLM adjudication results (output/llm_adjudication_results.csv) and
folds each resolved candidate's counts into its parent brand's row in the
canonicalized brand table (output/brands_merged.csv from canonicalize.py).

Design choices, worth stating in the report:
  - Only candidates with type == "brand" AND confidence in ACCEPTED_CONFIDENCE
    are auto-merged. "low" confidence candidates are NOT auto-merged even if a
    brand was named, since a low-confidence guess shouldn't silently inflate a
    brand's count.
  - MANUAL_OVERRIDES is a safety net for candidates the LLM has previously
    mis-scoped as type=="brand" on manual spot-check, without editing the raw
    LLM output file:
      - "ellps" is an acronym for "Entry Level Luxury Performance Sedan", a
        market segment, not a model code.
      - "awd" is a generic drivetrain attribute (all-wheel drive), not a
        brand-specific model name.
    Both should now land in the "attribute" bucket under the current
    brand/attribute/other prompt (see llm_adjudicate.py), so these overrides
    are a belt-and-suspenders check, not load-bearing.
"""

import pandas as pd

ACCEPTED_CONFIDENCE = {"high", "medium"}

MANUAL_OVERRIDES = {
    "ellps": None,  # exclude -- market segment term, not a model code (see report notes)
    "awd": None,  # exclude -- generic attribute, not brand-specific
    "germans": None,  # exclude -- LLM's own reasoning says this is a stand-in for
    # German makes generally (BMW/Audi/Mercedes/VW), not BMW-specific, despite
    # being assigned brand="bmw" at medium confidence.
    "type": None,  # exclude -- LLM assigned brand="lexus" but its own reasoning
    # names Acura (Type-S, which IS a valid anchor) and Jaguar (not in the anchor
    # list) as the more likely referents -- an internally inconsistent answer,
    # too ambiguous a token to safely attribute to any one brand.
}


def merge_llm_resolved_candidates(
    brands_merged: pd.DataFrame,
    candidate_table: pd.DataFrame,
    llm_results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Returns an updated copy of brands_merged with each accepted LLM-resolved
    candidate's raw_count/message_reach added into its assigned brand's row.
    """
    brands_merged = brands_merged.copy().set_index("candidate")
    candidate_table = candidate_table.set_index("candidate")

    merged_log = []

    for _, row in llm_results.iterrows():
        cand = row["candidate"]
        brand = row["brand"]
        confidence = row["confidence"]

        if cand in MANUAL_OVERRIDES:
            override_brand = MANUAL_OVERRIDES[cand]
            if override_brand is None:
                merged_log.append(
                    {"candidate": cand, "action": "excluded (manual override)"}
                )
                continue
            brand = override_brand

        if row.get("type") != "brand" or pd.isna(brand):
            continue
        if confidence not in ACCEPTED_CONFIDENCE:
            merged_log.append(
                {"candidate": cand, "action": f"skipped (confidence={confidence})"}
            )
            continue
        if brand not in brands_merged.index:
            merged_log.append(
                {
                    "candidate": cand,
                    "action": f"skipped (brand '{brand}' not a known anchor)",
                }
            )
            continue
        if cand not in candidate_table.index:
            merged_log.append(
                {"candidate": cand, "action": "skipped (no count data found)"}
            )
            continue

        brands_merged.loc[brand, "raw_count"] += candidate_table.loc[cand, "raw_count"]
        brands_merged.loc[brand, "message_reach"] += candidate_table.loc[
            cand, "message_reach"
        ]
        brands_merged.loc[brand, "variant_count"] += 1
        merged_log.append({"candidate": cand, "action": f"merged into '{brand}'"})

    print(pd.DataFrame(merged_log).to_string())
    return brands_merged.reset_index().sort_values("message_reach", ascending=False)


def main():
    brands_merged = pd.read_csv("output/brands_merged.csv")
    candidate_table = pd.read_csv("output/candidate_frequencies_clean.csv")
    llm_results = pd.read_csv("output/llm_adjudication_results.csv")

    final_brands = merge_llm_resolved_candidates(
        brands_merged, candidate_table, llm_results
    )
    final_brands.to_csv("output/final_brands.csv", index=False)

    print("\nFinal brand table:")
    print(final_brands.to_string())


if __name__ == "__main__":
    main()

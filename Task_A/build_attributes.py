"""
build_attributes.py

Builds the final product-attribute frequency table. Attribute discovery reuses
the same candidate pool and LLM adjudication pass as brand discovery
(llm_adjudicate.py classifies each ambiguous candidate as brand / attribute /
other in one call) -- this file just filters that output down to
type == "attribute" and rolls surface-form variants (e.g. "gas", "mpg" as
separate raw candidates) up into their shared canonical attribute_category.

Design choices, worth stating in the report:
  - Only "high"/"medium" confidence attribute calls are counted, matching the
    brand pipeline's ACCEPTED_CONFIDENCE bar -- a low-confidence guess
    shouldn't inflate an attribute's count either.
  - Canonicalization happens inside the LLM prompt itself (a closed
    attribute_category list, see ATTRIBUTE_CATEGORIES in llm_adjudicate.py)
    rather than as a separate post-hoc clustering step, so there's no risk of
    near-duplicate categories ("fuel economy" vs "gas mileage") surviving
    into the final table.
"""

import pandas as pd

ACCEPTED_CONFIDENCE = {"high", "medium"}


def build_attribute_table(
    llm_results: pd.DataFrame, candidate_table: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate_table = candidate_table.set_index("candidate")

    attribute_rows = llm_results[
        (llm_results["type"] == "attribute")
        & (llm_results["confidence"].isin(ACCEPTED_CONFIDENCE))
        & (llm_results["attribute_category"].notna())
    ].copy()

    detail_rows = []
    for _, row in attribute_rows.iterrows():
        cand = row["candidate"]
        if cand not in candidate_table.index:
            continue
        detail_rows.append(
            {
                "attribute_category": row["attribute_category"],
                "candidate": cand,
                "raw_count": candidate_table.loc[cand, "raw_count"],
                "message_reach": candidate_table.loc[cand, "message_reach"],
                "confidence": row["confidence"],
            }
        )
    detail = pd.DataFrame(detail_rows)

    summary = (
        detail.groupby("attribute_category")
        .agg(
            raw_count=("raw_count", "sum"),
            message_reach=("message_reach", "sum"),
            variant_count=("candidate", "nunique"),
            example_terms=(
                "candidate",
                lambda s: ", ".join(sorted(s, key=len)[:6]),
            ),
        )
        .reset_index()
        .sort_values("message_reach", ascending=False)
        .reset_index(drop=True)
    )
    return summary, detail


def main():
    llm_results = pd.read_csv("output/llm_adjudication_results.csv")
    candidate_table = pd.read_csv("output/candidate_frequencies_clean.csv")

    summary, detail = build_attribute_table(llm_results, candidate_table)

    detail.to_csv("output/attribute_term_detail.csv", index=False)
    summary.to_csv("output/final_attributes.csv", index=False)

    print(f"Attribute categories found: {len(summary)}")
    print(summary.to_string())


if __name__ == "__main__":
    main()

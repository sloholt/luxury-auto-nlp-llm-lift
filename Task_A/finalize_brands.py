"""
finalize_brands.py

Applies the corruption correction from resolve_ambiguous_brands.py to the
merged brand table (output/final_brands.csv) and selects the final top 10.

Why this step exists (see report for the full writeup): raw substring
matching on "toyota", "pontiac", and "honda" picks up a large amount of
mid-word text-injection corruption specific to this corpus (e.g. "statistics"
-> "stat toyota ics", "according" -> "ac honda ing", "G35" -> "pontiac 5").
resolve_ambiguous_brands.py LLM-sampled a random set of raw matches per term
and estimated what fraction are genuine brand mentions vs. corruption
artifacts; this script substitutes that corrected estimate back in before
ranking, since ranking on the uncorrected counts would materially change
which brands land in the top 10 (uncorrected: toyota #2, pontiac #7 --
corrected: both fall well outside the top 10).

Assumption stated explicitly: the sampled genuine-rate is applied uniformly
to both message_reach and raw_count for a given brand, since the injection
mechanism inserts one spurious token per corrupted message (so the corruption
rate in reach and in raw mention count should track closely).
"""

import pandas as pd

TOP_N = 10


def apply_corrections(
    final_brands: pd.DataFrame, corrections: pd.DataFrame
) -> pd.DataFrame:
    final_brands = final_brands.copy().set_index("candidate")
    corrections = corrections.set_index("term")

    final_brands["corrected"] = False
    final_brands["genuine_rate"] = None

    for term, row in corrections.iterrows():
        if term not in final_brands.index:
            continue
        rate = row["genuine_rate"]
        final_brands.loc[term, "raw_count"] = round(
            final_brands.loc[term, "raw_count"] * rate
        )
        final_brands.loc[term, "message_reach"] = row["corrected_reach_estimate"]
        final_brands.loc[term, "corrected"] = True
        final_brands.loc[term, "genuine_rate"] = rate

    return final_brands.reset_index().sort_values(
        "message_reach", ascending=False
    ).reset_index(drop=True)


def main():
    final_brands = pd.read_csv("output/final_brands.csv")
    corrections = pd.read_csv("output/ambiguous_brand_correction.csv")

    corrected = apply_corrections(final_brands, corrections)
    corrected.to_csv("output/final_brands_corrected.csv", index=False)

    top10 = corrected.head(TOP_N)
    top10.to_csv("output/final_brands_top10.csv", index=False)

    print("Full corrected/ranked brand table:")
    print(corrected.to_string())
    print(f"\nTop {TOP_N}:")
    print(top10.to_string())


if __name__ == "__main__":
    main()

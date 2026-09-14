"""
calculate_lift.py

Computes pairwise co-occurrence lift among Task A's final top-10 luxury-auto
brands (see ../Task_A/output/final_brands_top10.csv and ../Task_A/REPORT.md):

    bmw, acura, cadillac, audi, lexus, gm, infiniti, mercedes-benz, volvo, nissan

Method (deliberately lexical, no LLM call -- fully reproducible and auditable):
  1. Reuse Task A's own cleaned corpus loader (`extract_candidates.load_and_clean_data`),
     which strips the corpus's known quote-artifact and burst-repeat noise
     (`cleaning.strip_quote_artifact`, `cleaning.collapse_burst_repeats`) before any
     matching happens.
  2. Reuse Task A's own presence test (`canonicalize.compute_brand_presence`): a
     brand is "present" in a message if the brand's canonical anchor token appears
     as a whole word (`\\b<anchor>\\b`, case-insensitive) anywhere in that message's
     text. This is the exact same regex machinery Task A already validated and used
     for its own (candidate -> brand) lift signal in `canonicalize.lift_assign_candidate`.
  3. For every unordered pair of brands (A, B), compute lift the standard
     market-basket-analysis way:

         lift(A, B) = P(A and B) / (P(A) * P(B)) = (N * n_both) / (n_A * n_B)

     where N = total messages, n_A/n_B = messages mentioning that brand, and
     n_both = messages mentioning both. lift > 1 means the two brands are
     discussed together more often than chance would predict (comparison/
     cross-shopping pairs); lift < 1 means they co-occur less than chance
     (brands rarely discussed in the same breath); lift ~= 1 means no
     association either way.

Why anchor-only, and not also the LLM-resolved slang/model-code variants
(e.g. "e46", "bimmer" -> bmw; "tl" -> acura; "mb" -> mercedes-benz) that Task A
folded into its final brand *counts*: those resolutions were validated by an LLM
one candidate at a time, with forum excerpts as context, for the specific purpose
of ranking brand importance. Re-applying an ambiguous short token like "ls" (Lexus
LS) or "gm"-adjacent slang as a blind full-corpus regex match here would reintroduce
exactly the kind of spurious-lift risk Task A's own report calls out (e.g. "forum"
showing lift=3.89 toward Lincoln, "fwd" showing lift=3.87 toward Nissan -- both
coincidence, not signal). Sticking to the single unambiguous anchor token per brand
keeps this script's presence test transparent and conservative -- every "co-mention"
counted here is a message that says both brand names outright, nothing inferred.

Note: because this presence test scans raw message text directly, rather than
Task A's spaCy noun-chunk candidate extraction + substring/LLM merge pipeline, the
resulting single-brand message counts will not exactly match `final_brands_top10.csv`'s
message_reach column (that pipeline's reach includes slang/model-code variants folded
in per-brand, and is itself an overcount per Task A's own "overlapping-span double
counting" limitation note). The relative brand ranking is consistent either way; see
`output/brand_presence_counts.csv` for a side-by-side.

Outputs (written to ./output/):
  - brand_presence_counts.csv : per-brand message count (N and n_brand) used below
  - brand_lift_matrix.csv     : symmetric 10x10 brand-by-brand lift matrix
  - brand_pair_lift.csv       : long-format one row per unordered brand pair,
                                sorted by lift descending, with n_both and support
"""

import os
import sys

import pandas as pd

TASK_A_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Task_A")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

sys.path.insert(0, TASK_A_DIR)

from extract_candidates import load_and_clean_data, DATA_PATH  # noqa: E402
from canonicalize import compute_brand_presence  # noqa: E402

TOP10_BRANDS_PATH = os.path.join(TASK_A_DIR, "output", "final_brands_top10.csv")


def load_top10_brands(path=TOP10_BRANDS_PATH):
    """Read Task A's own ranked top-10 output rather than re-typing the list,
    so this script always reflects whatever Task A currently reports as final."""
    top10 = pd.read_csv(path)
    return top10["candidate"].tolist()


def compute_lift_matrix(presence: pd.DataFrame) -> pd.DataFrame:
    """Symmetric brand x brand lift matrix; diagonal is left as NaN since
    self-lift is trivially undefined (a message always co-occurs with itself)."""
    brands = list(presence.columns)
    N = len(presence)
    matrix = pd.DataFrame(index=brands, columns=brands, dtype=float)

    for i, brand_a in enumerate(brands):
        n_a = presence[brand_a].sum()
        for brand_b in brands[i:]:
            if brand_a == brand_b:
                continue
            n_b = presence[brand_b].sum()
            n_both = (presence[brand_a] & presence[brand_b]).sum()
            lift = 0.0 if n_a == 0 or n_b == 0 or n_both == 0 else (N * n_both) / (n_a * n_b)
            matrix.loc[brand_a, brand_b] = lift
            matrix.loc[brand_b, brand_a] = lift

    return matrix


def compute_pair_table(presence: pd.DataFrame) -> pd.DataFrame:
    """Long-format one-row-per-pair view: easier to read/sort than the matrix."""
    brands = list(presence.columns)
    N = len(presence)
    rows = []

    for i, brand_a in enumerate(brands):
        n_a = int(presence[brand_a].sum())
        for brand_b in brands[i + 1 :]:
            n_b = int(presence[brand_b].sum())
            n_both = int((presence[brand_a] & presence[brand_b]).sum())
            lift = 0.0 if n_a == 0 or n_b == 0 or n_both == 0 else (N * n_both) / (n_a * n_b)
            rows.append(
                {
                    "brand_a": brand_a,
                    "brand_b": brand_b,
                    "n_a": n_a,
                    "n_b": n_b,
                    "n_both": n_both,
                    "support_both": round(n_both / N, 5),
                    "expected_n_both_if_independent": round((n_a * n_b) / N, 2),
                    "lift": round(lift, 3),
                }
            )

    return pd.DataFrame(rows).sort_values("lift", ascending=False).reset_index(drop=True)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    top10 = load_top10_brands()
    print(f"Top 10 brands (from Task A): {top10}")

    df = load_and_clean_data(os.path.join(TASK_A_DIR, DATA_PATH))
    N = len(df)
    print(f"Corpus size after cleaning: {N} messages")

    presence = compute_brand_presence(df, anchors=top10)

    counts = presence.sum().rename("n_messages").reset_index().rename(columns={"index": "brand"})
    counts["share_of_corpus"] = (counts["n_messages"] / N).round(4)
    counts = counts.sort_values("n_messages", ascending=False).reset_index(drop=True)
    counts.to_csv(os.path.join(OUTPUT_DIR, "brand_presence_counts.csv"), index=False)
    print("\nPer-brand message presence:")
    print(counts.to_string(index=False))

    lift_matrix = compute_lift_matrix(presence)
    lift_matrix.to_csv(os.path.join(OUTPUT_DIR, "brand_lift_matrix.csv"))

    pair_table = compute_pair_table(presence)
    pair_table.to_csv(os.path.join(OUTPUT_DIR, "brand_pair_lift.csv"), index=False)

    print("\nTop 10 brand pairs by lift (most co-mentioned relative to chance):")
    print(pair_table.head(10).to_string(index=False))
    print("\nBottom 10 brand pairs by lift (least co-mentioned relative to chance):")
    print(pair_table.tail(10).to_string(index=False))

    print(f"\nWrote brand_presence_counts.csv, brand_lift_matrix.csv, brand_pair_lift.csv to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

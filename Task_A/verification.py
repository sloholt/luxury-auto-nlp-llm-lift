import pandas as pd
from extract_candidates import load_and_clean_data, DATA_PATH


def flag_suspicious_candidates(
    table: pd.DataFrame, ratio_threshold: float = 2.5
) -> pd.DataFrame:
    """
    Flag candidates where raw_count is disproportionately high relative to message_reach —
    i.e. repeating many times within few messages, consistent with burst/injection noise
    rather than organic, spread-out discussion.
    """
    table = table.copy()
    table["repeat_ratio"] = table["raw_count"] / table["message_reach"]
    return table[table["repeat_ratio"] >= ratio_threshold].sort_values(
        "repeat_ratio", ascending=False
    )


def show_source_messages(
    df: pd.DataFrame, candidate: str, n: int = 5, text_col: str = "message"
):
    """Print up to n messages containing the given candidate string, for manual verification."""
    matches = df[df[text_col].str.contains(candidate, case=False, na=False)]
    print(f"'{candidate}' appears in {len(matches)} messages (showing up to {n}):")
    for _, row in matches.head(n).iterrows():
        print("-" * 60)
        print(row[text_col][:300])


def print_inspection(sample: pd.DataFrame, n: int = 5) -> None:
    """Print the first n annotated rows for manual review."""
    for _, row in sample.head(n).iterrows():
        print("=" * 80)
        print(f"user: {row['user']}")
        print(f"noun chunks: {row['noun_chunks'][:15]}")
        print(f"regex model-code hits: {row['regex_model_code']}")


def main():
    df = load_and_clean_data(DATA_PATH)
    table = pd.read_csv("output/candidate_frequencies.csv")

    suspicious = flag_suspicious_candidates(table)
    print(f"\nSuspicious candidates (high repeat ratio): {len(suspicious)}")
    print(suspicious.head(20))
    suspicious.to_csv("output/suspicious_candidates.csv", index=False)

    candidates_to_trace = list(table.head(10)["candidate"]) + list(
        suspicious["candidate"]
    )
    for cand in dict.fromkeys(candidates_to_trace):  # dedupe, preserve order
        show_source_messages(df, cand, n=3)


if __name__ == "__main__":
    main()

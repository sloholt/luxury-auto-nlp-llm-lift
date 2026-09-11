import pandas as pd
import spacy
import re
from collections import Counter, defaultdict

DATA_PATH = "sample_data.csv"

MODEL_CODE_PATTERN = re.compile(
    r"\b(\d{1,3}\s?series|\d{2,3}[a-z]{1,3}(?:-[a-z])?|[a-z]\d{1,3}[a-z]{0,2})\b",
    re.IGNORECASE,
)
LEADING_DETERMINERS = re.compile(
    r"^(the|a|an|my|your|his|her|their|our|some|this|that)\s+", re.IGNORECASE
)
# tokens that are noun-chunk-shaped but never brand/model/attribute candidates
STOP_CANDIDATES = {
    "i",
    "you",
    "he",
    "she",
    "we",
    "they",
    "it",
    "this",
    "that",
    "these",
    "those",
    "me",
    "him",
    "her",
    "us",
    "them",
    "who",
    "which",
    "something",
    "someone",
}


def load_and_clean_data(data_path):
    df = pd.read_csv(
        DATA_PATH,
        header=None,
        names=["user", "date", "message"],
        dtype={"user": "string", "date": "string", "message": "string"},
        keep_default_na=False,
    )

    # Drop empty messages
    before = len(df)
    df = df[df["message"].str.strip() != ""].copy()
    print(f"Dropped {before - len(df)} empty messages")

    # Normalize dates
    split_cols = df["date"].str.split("-", n=1, expand=True)
    df["date"] = split_cols[0].str.zfill(2) + "-" + split_cols[1]
    df["date_parsed"] = pd.to_datetime(df["date"], format="%y-%b", errors="coerce")

    # print(f"Unparseable dates: {df['date_parsed'].isna().sum()}")
    # print(df[["date", "date_parsed"]].drop_duplicates().sort_values("date_parsed"))
    return df


# Helper function for creating a sample of the dataset for inspection
def get_sample(df, n=80, seed: int = 42):
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


# Noun chunks & candidates
def extract_noun_chunks(text, nlp, max_chars=2000):
    doc = nlp(text[:max_chars])
    nouns = [chunk.text.strip() for chunk in doc.noun_chunks]
    # print(f"Noun chunks for {text} are {nouns}")
    return nouns


def extract_model_codes(text):
    codes = MODEL_CODE_PATTERN.findall(text)
    # print(f"Text Block: {text}")
    # print(f"Model Codes: {codes}")
    return codes


def annotate_sample(sample, nlp):
    sample = sample.copy()
    sample["noun_chunks"] = sample["message"].apply(
        lambda t: extract_noun_chunks(t, nlp)
    )
    sample["regex_model_code"] = sample["message"].apply(extract_model_codes)
    return sample


def export_sample(sample, outpath):
    export_df = sample.copy()
    export_df["noun_chunks"] = export_df["noun_chunks"].apply(lambda lst: "|".join(lst))
    export_df["regex_model_code"] = export_df["regex_model_code"].apply(
        lambda lst: " | ".join(lst)
    )
    export_df.to_csv(outpath, index=False)
    print(f"Wrote {len(export_df)} rows to {outpath}")


def clean_chunk(text):
    """Strip leading determiners/possessives and surrounding whitespace/punctuation."""
    text = text.strip().strip(".,!?\"'")
    text = LEADING_DETERMINERS.sub("", text)
    return text.strip()


def is_plausible_candidate(text):
    """ "Remove empty, pronoun-only or single character chunks"""
    if not text:
        return False
    if text.lower() in STOP_CANDIDATES:
        return False
    if len(text) < 2:
        return False
    return True


def extract_candidate_counts(df, nlp, text_col="message", batch_size=50):
    chunk_counts = Counter()
    chunk_reach = defaultdict(int)
    texts = df[text_col].tolist()

    for doc in nlp.pipe(texts, batch_size=batch_size):
        seen_in_this_message = set()

        # Extracted noun chunks from parsed doc
        for span in doc.noun_chunks:
            cleaned = clean_chunk(span.text)
            if is_plausible_candidate(cleaned):
                chunk_counts[cleaned] += 1
                seen_in_this_message.add(cleaned)

        # Regex model codes
        for hit in extract_model_codes(doc.text):
            hit = hit.strip()
            chunk_counts[hit] += 1
            seen_in_this_message.add(hit)

        for cand in seen_in_this_message:
            chunk_reach[cand] += 1
    return chunk_counts, chunk_reach


def build_candidate_table(chunk_counts: Counter, chunk_reach, min_reach=2):
    rows = [
        {
            "candidate": cand,
            "raw_count": chunk_counts[cand],
            "message_reach": chunk_reach[cand],
        }
        for cand in chunk_counts
        if chunk_reach[cand] >= min_reach
    ]
    table = (
        pd.DataFrame(rows)
        .sort_values("message_reach", ascending=False)
        .reset_index(drop=True)
    )
    return table


def main():
    nlp = spacy.load("en_core_web_sm")
    df = load_and_clean_data(DATA_PATH)

    # Stage 1: Sample Inspection
    # sample = get_sample(df, n=80)
    # sample = annotate_sample(sample, nlp)
    # export_sample(sample, "sample_inspection.csv")

    # Stage 2: full candidate frequnecy table
    counts, reach = extract_candidate_counts(df, nlp)
    table = build_candidate_table(counts, reach, min_reach=2)

    print(f"\nCandidates with reach >=2: {len(table)}")
    print(table.head(30))

    table.to_csv("candidate_frequencies.csv", index=False)
    # Stage 3: verification
    suspicious = flag_suspicious_candidates(table)
    print(f"\nSuspicious candidates (high repeat ratio): {len(suspicious)}")
    print(suspicious.head(20))
    suspicious.to_csv("suspicious_candidates.csv", index=False)

    # Trace a handful of top candidates and all suspicious ones back to source messages
    candidates_to_trace = list(table.head(10)["candidate"]) + list(
        suspicious["candidate"]
    )
    for cand in dict.fromkeys(candidates_to_trace):  # dedupe, preserve order
        show_source_messages(df, cand, n=3)


if __name__ == "__main__":
    main()

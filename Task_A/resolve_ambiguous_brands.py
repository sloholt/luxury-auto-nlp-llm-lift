"""
resolve_ambiguous_brands.py

Estimates the genuine-mention rate for brands known to have corruption in their
raw counts (toyota, honda: mid-word injection, e.g. "statistics" -> "stat toyota
ics", "according" -> "ac honda ing"; pontiac: substitution for "G" in model
codes like G35 -- see report notes). Rather than trying to individually catch
every corrupted instance, this samples a manageable number of real occurrences,
has the LLM classify each as genuine vs. corrupted, and uses that sample's rate
to produce a corrected estimate for the full count.

cleaning.py's AMBIGUOUS_CANDIDATES also flags volkswagen and mazda as
injection-prone, but neither shows up in the top-10 brand ranking even at its
raw (uncorrected) count, so they are left unsampled here -- correcting them
could not change the final top-10 answer.
"""

import json
import re
import time

import pandas as pd
from anthropic import Anthropic
from llm_utils import extract_text_from_response

client = Anthropic()
MODEL = "claude-sonnet-5"
MAX_RETRIES = 3
SLEEP_BETWEEN_CALLS = 0.5
SAMPLE_SIZE = 100
RANDOM_SEED = 42


def sample_messages_containing(
    df: pd.DataFrame,
    term: str,
    n: int = SAMPLE_SIZE,
    text_col: str = "message",
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Draw a reproducible random sample of messages containing the given term."""
    pattern = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
    matches = df[df[text_col].str.contains(pattern, na=False)]
    n = min(n, len(matches))
    return matches.sample(n=n, random_state=seed).reset_index(drop=True)


def _parse_llm_json(raw_text: str) -> dict:
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Could not parse LLM response as JSON: {raw_text!r}") from e


def classify_mention(term: str, message: str) -> dict:
    """Ask the LLM whether a single occurrence of `term` in `message` is genuine or corrupted."""
    prompt = f"""You are cleaning noisy automotive forum data. Some messages in this corpus
contain corrupted text where a brand-like word was injected mid-word into an
unrelated word (e.g. "sophisticated" corrupted to "soph {term} icated"), or
substituted for part of a model code (e.g. "G35" corrupted to "{term} 5").

Here is a message containing the term "{term}":
"{message[:500]}"

Is this a GENUINE mention of the "{term}" brand (a real car brand being discussed),
or is it CORRUPTED text (injected mid-word noise, or a substitution artifact)?

Respond ONLY with JSON, no other text:
{{"classification": "genuine" or "corrupted", "confidence": "high|medium|low", "reasoning": "one sentence"}}"""

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=1024,
                thinking={"type": "disabled"},
                messages=[{"role": "user", "content": prompt}],
            )
            raw_text = extract_text_from_response(response)
            if raw_text is None:
                raise ValueError("Response contained only a thinking block, no text")
            result = _parse_llm_json(raw_text)
            result["term"] = term
            return result
        except Exception as e:
            last_error = e
            print(f"  Attempt {attempt}/{MAX_RETRIES} failed: {e}")
            time.sleep(1.5 * attempt)

    return {
        "term": term,
        "classification": "error",
        "confidence": "error",
        "reasoning": f"LLM call failed after {MAX_RETRIES} attempts: {last_error}",
    }


def estimate_genuine_rate(
    df: pd.DataFrame, term: str, full_reach: int, n: int = SAMPLE_SIZE
) -> tuple[dict, pd.DataFrame]:
    """
    Samples messages containing `term`, classifies each, and scales the sample's
    genuine-rate up to the full reach count as a corrected estimate.
    """
    sample = sample_messages_containing(df, term, n=n)
    print(
        f"Sampling {len(sample)} messages containing '{term}' (full reach: {full_reach})"
    )

    results = []
    for i, row in sample.iterrows():
        print(f"  [{i + 1}/{len(sample)}] classifying...")
        result = classify_mention(term, row["message"])
        results.append(result)
        time.sleep(SLEEP_BETWEEN_CALLS)

    results_df = pd.DataFrame(results)
    n_valid = (results_df["classification"] != "error").sum()
    n_genuine = (results_df["classification"] == "genuine").sum()
    genuine_rate = n_genuine / n_valid if n_valid > 0 else None
    corrected_reach = (
        round(full_reach * genuine_rate) if genuine_rate is not None else None
    )

    summary = {
        "term": term,
        "sample_size": len(sample),
        "n_genuine_in_sample": n_genuine,
        "n_valid": n_valid,
        "genuine_rate": round(genuine_rate, 3) if genuine_rate is not None else None,
        "raw_reach": full_reach,
        "corrected_reach_estimate": corrected_reach,
    }
    return summary, results_df


def main():
    from extract_candidates import load_and_clean_data, DATA_PATH

    df = load_and_clean_data(DATA_PATH)
    final_brands = pd.read_csv("output/final_brands.csv").set_index("candidate")

    terms_to_check = {
        "toyota": final_brands.loc["toyota", "message_reach"],
        "pontiac": final_brands.loc["pontiac", "message_reach"],
        "honda": final_brands.loc["honda", "message_reach"],
    }

    all_summaries = []
    for term, full_reach in terms_to_check.items():
        summary, details = estimate_genuine_rate(df, term, int(full_reach))
        all_summaries.append(summary)
        details.to_csv(f"output/ambiguous_sample_{term}.csv", index=False)

    summary_df = pd.DataFrame(all_summaries)
    print("\n" + summary_df.to_string())
    summary_df.to_csv("output/ambiguous_brand_correction.csv", index=False)


if __name__ == "__main__":
    main()

import re
import json
import time
import pandas as pd
from anthropic import Anthropic
from llm_utils import extract_text_from_response

client = Anthropic()
MODEL = "claude-sonnet-5"
MAX_RETRIES = 3
SLEEP_BETWEEN_CALLS = 0.5

# Closed set of product-attribute categories, derived by inspecting the
# highest-reach non-brand candidates (output/candidate_frequencies_clean.csv)
# and grouping surface forms that clearly describe the same underlying
# attribute (e.g. "mpg"/"gas" -> fuel economy; "ride"/"brakes" -> handling).
# Kept closed (rather than free-text) so the LLM's output is already
# canonicalized -- no separate synonym-clustering pass is needed afterward.
ATTRIBUTE_CATEGORIES = [
    "price/value",
    "performance/powertrain",
    "drivetrain configuration",
    "handling/ride",
    "interior/comfort",
    "reliability/quality",
    "styling/body type",
    "fuel economy",
    "segment/positioning",
    "technology/features",
    "safety",
]
# Kept as short, exact tokens (no parenthetical examples in the label itself)
# so the model echoes them back verbatim instead of paraphrasing -- an
# earlier version with "segment/positioning (luxury, class, entry-level)"
# sometimes came back as the shorter "segment/positioning", silently
# splitting one category into two rows downstream.


def get_context_snippets(df, candidate, n=5, text_col="message"):
    pattern = re.compile(rf"\b{re.escape(candidate)}\b", re.IGNORECASE)
    matches = df[df[text_col].str.contains(pattern, na=False)]
    return matches[text_col].str.slice(0, 300).tolist()[:n]


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


def adjudicate_candidate(
    candidate: str,
    context_snippets: list[str],
    brand_options: list[str],
    attribute_categories: list[str] = ATTRIBUTE_CATEGORIES,
) -> dict:
    prompt = f"""You are labeling automotive forum data. The candidate term is: "{candidate}"

Here are real forum message excerpts containing this term:
{chr(10).join(f'- {s}' for s in context_snippets)}

Classify "{candidate}" as exactly ONE of:
  (a) BRAND -- it functions here as a reference to a specific car brand, via a
      model name, trim, chassis code, abbreviation, or slang (e.g. "tl" ->
      acura, "e46" -> bmw, "bimmer" -> bmw). Brand options: {', '.join(brand_options)}.
  (b) ATTRIBUTE -- it names a product attribute/feature being discussed, not
      tied to one brand. Attribute categories: {', '.join(attribute_categories)}.
  (c) OTHER -- generic forum discourse, a pronoun, a person/place name, an
      unrelated word, or too ambiguous to classify confidently.

Respond ONLY with JSON, no other text:
{{"candidate": "...", "type": "brand" | "attribute" | "other", "brand": "<name from the brand options, or null>", "attribute_category": "<name from the attribute categories, or null>", "confidence": "high|medium|low", "reasoning": "one sentence"}}"""

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
            result["candidate"] = candidate
            return result
        except Exception as e:
            last_error = e
            print(f"  Attempt {attempt}/{MAX_RETRIES} failed for '{candidate}': {e}")
            time.sleep(1.5 * attempt)
    return {
        "candidate": candidate,
        "type": "error",
        "brand": None,
        "attribute_category": None,
        "confidence": "error",
        "reasoning": f"LLM call failed after {MAX_RETRIES} attempts: {last_error}",
    }


def run_adjudication_batch(
    df: pd.DataFrame,
    candidates: list[str],
    brand_options: list[str],
    attribute_categories: list[str] = ATTRIBUTE_CATEGORIES,
) -> pd.DataFrame:
    results = []
    total = len(candidates)
    for i, cand in enumerate(candidates, start=1):
        print(f"[{i}/{total}] Adjudicating '{cand}'...")
        snippets = get_context_snippets(df, cand)
        if not snippets:
            print(f"  No context found for '{cand}' -- skipping")
            continue
        result = adjudicate_candidate(cand, snippets, brand_options, attribute_categories)
        results.append(result)
        time.sleep(SLEEP_BETWEEN_CALLS)
    return pd.DataFrame(results)


def main():
    from extract_candidates import load_and_clean_data, DATA_PATH
    from canonicalize import BRAND_ANCHORS

    df = load_and_clean_data(DATA_PATH)
    needs_review = pd.read_csv("output/needs_llm_review.csv")
    candidates = needs_review["candidate"].tolist()

    print(
        f"Adjudicating {len(candidates)} candidates against {len(BRAND_ANCHORS)} brand "
        f"options and {len(ATTRIBUTE_CATEGORIES)} attribute categories..."
    )
    results = run_adjudication_batch(df, candidates, BRAND_ANCHORS)

    print("\n" + results.to_string())
    results.to_csv("output/llm_adjudication_results.csv", index=False)

    n_brand = (results["type"] == "brand").sum()
    n_attribute = (results["type"] == "attribute").sum()
    n_other = (results["type"] == "other").sum()
    n_errors = (results["type"] == "error").sum()
    print(
        f"\nBrand: {n_brand} | Attribute: {n_attribute} | Other: {n_other} | Errors: {n_errors}"
    )


if __name__ == "__main__":
    main()

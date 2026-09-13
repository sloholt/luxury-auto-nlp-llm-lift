import re
import pandas as pd
from wordfreq import zipf_frequency
from spellchecker import SpellChecker

_spell = SpellChecker()

QUOTE_ARTIFACT = re.compile(r'"\s*kia hyundai\s*', re.IGNORECASE)
AMBIGUOUS_CANDIDATES = {"toyota", "honda", "pontiac"}
BURST_PATTERN = re.compile(
    r"(\b\w[\w\-]*\b(?:\s+\w[\w\-]*\b){0,3})(\s+\1){1,}", re.IGNORECASE
)
DOMAIN_GENERIC_TERMS = {
    "car",
    "cars",
    "vehicle",
    "vehicles",
    "auto",
    "automobile",
    "automobiles",
}


def strip_quote_artifact(text):
    return QUOTE_ARTIFACT.sub("", text)


def is_generic_word(candidate, threshold=5.5):
    if " " in candidate:
        return False
    if candidate.lower() in DOMAIN_GENERIC_TERMS:
        return True
    return zipf_frequency(candidate, "en") >= threshold


def filter_generic_candidates(table, threshold=5.5):
    mask = table["candidate"].apply(
        lambda c: not is_generic_word(c, threshold=threshold)
    )
    return table[mask].reset_index(drop=True)


def is_real_word(word):
    return word.lower() in _spell


def flag_midword_injection(prev_word, candidate, next_word):
    if not prev_word or not next_word:
        return False
    combine = (prev_word + next_word).lower()
    return (
        is_real_word(combine)
        and not is_real_word(prev_word)
        and not is_real_word(next_word)
    )


def scan_message_for_injections(message, candidate):
    tokens = message.split()
    flagged = []
    for i, tok in enumerate(tokens):
        if tok.strip(".,!?\"'").lower() != candidate.lower():
            continue
        prev_word = tokens[i - 1].strip(".,!?\"'") if i > 0 else ""
        next_word = tokens[i + 1].strip(".,!?\"'") if i < len(tokens) - 1 else ""
        if flag_midword_injection(prev_word, candidate, next_word):
            context = " ".join(tokens[max(0, i - 4) : i + 5])
            flagged.append({"candidate": candidate, "context": context})
    return flagged


def collapse_burst_repeats(text):
    prev = None
    while prev != text:
        prev = text
        text = BURST_PATTERN.sub(lambda m: m.group(1), text)
    return text


def build_injection_report(df, candidates=AMBIGUOUS_CANDIDATES, text_col="message"):
    rows = []
    for msg in df[text_col]:
        for cand in candidates:
            if cand not in msg.lower():
                continue
            rows.extend(scan_message_for_injections(msg, cand))
    return pd.DataFrame(rows)


def main():
    from extract_candidates import load_and_clean_data, DATA_PATH

    table = pd.read_csv("output/candidate_frequencies.csv")
    cleaned_table = filter_generic_candidates(table)
    cleaned_table.to_csv("output/candidate_frequencies_clean.csv", index=False)
    print(f"Filtered {len(table) - len(cleaned_table)} generic-word candidates")

    df = load_and_clean_data(DATA_PATH)
    injections = build_injection_report(df)
    injections.to_csv("output/flagged_injections.csv", index=False)
    print(f"Flagged {len(injections)} mid-word injection instances for review")


if __name__ == "__main__":
    main()

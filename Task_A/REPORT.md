# Task A: Brand and Attribute Discovery in Luxury-Auto Forum Text

**Corpus:** 6,000 forum posts from 329 users (`data/sample_data.csv`), no brand/model
dictionary supplied.

## 1. Pipeline overview

The pipeline is deliberately split into an automated stage (cheap, deterministic,
runs over the whole corpus) and a semantic-interpretation stage (expensive, run
only where automation can't resolve ambiguity):

| Stage | Script | Automated or semantic? |
|---|---|---|
| Candidate extraction (noun chunks + regex model-code patterns) | `extract_candidates.py` | Automated |
| Generic-English-word filtering (`wordfreq` Zipf threshold) | `cleaning.py` | Automated |
| Known-brand substring anchoring (`bmw`, `audi`, ... 16 anchors) | `canonicalize.py` | Automated |
| Co-occurrence lift, as a *signal* only (not an auto-decision) | `canonicalize.py` | Automated (statistics), feeds semantic step |
| Brand vs. attribute vs. discourse classification | `llm_adjudicate.py` | Semantic (LLM) |
| Corruption genuine-rate estimation | `resolve_ambiguous_brands.py` | Semantic (LLM), sampled |
| Final merge, correction, ranking | `merge_resolved.py`, `finalize_brands.py`, `build_attributes.py` | Automated (given semantic labels) |

### 1.1 Why the 16 brand anchors aren't "a supplied dictionary"

`BRAND_ANCHORS` in `canonicalize.py` looks like a hand-typed list, but it was
derived *from the corpus*, not from an external gazetteer: `bmw`, `toyota`, `acura`,
`audi`, `honda`, `lexus`, `pontiac`, `infiniti`, `mercedes-benz`, `volvo`, `cadillac`,
`nissan`, `gm`, `lincoln`, `mazda`, `volkswagen` are simply the highest-message-reach
proper-noun-shaped tokens in `output/candidate_frequencies_clean.csv`, independently
recognized as real car brands (world knowledge used only to *confirm* a corpus-derived
candidate is a brand, not to go looking for brands the corpus doesn't surface). No
brand outside this discovered set was assumed in advance, and the set is treated as
a *seed for reconciliation*, not the final answer -- most of the actual work is
resolving the much larger set of indirect references (model names, trims, chassis
codes, and slang) back onto these anchors.

### 1.2 How brand variants are reconciled ("model names, abbreviations, variants")

1. **Substring anchoring** (automated): any candidate string containing an anchor
   as a whole word (e.g. "the bmw 3 series") rolls into that anchor's row.
2. **Statistical lift** (automated, advisory only): for every remaining candidate
   with `message_reach >= 30` (273 candidates), we compute lift toward each of the
   16 anchors -- how much more often the candidate co-occurs with a brand than
   chance predicts. We do **not** auto-merge on lift alone: in this corpus "forum"
   showed lift=3.89 toward Lincoln and "fwd" showed lift=3.87 toward Nissan, both
   clearly spurious coincidences at this sample size. Lift is passed to the LLM as
   context, not treated as a decision.
3. **LLM semantic adjudication** (`llm_adjudicate.py`): each of the 273 candidates,
   with up to 5 real forum excerpts as context, is classified in one call as
   `brand` / `attribute` / `other`. This is where "TL" -> Acura, "e46"/"e90" -> BMW,
   "bimmer" -> BMW, "caddy" -> Cadillac, "mb" -> Mercedes-Benz, "sts"/"cts" -> Cadillac,
   "s4"/"2.0t" -> Audi, and GM's constituent marques (Buick, Saab, Chevrolet) -> GM
   get resolved. Only `high`/`medium` confidence calls are merged; `low` confidence
   is discarded rather than silently inflating a brand's count.
4. **Manual override safety net** (`merge_resolved.py`): two LLM calls were
   overturned after spot-checking the model's own stated reasoning against its
   answer:
   - `"ellps"` -> the model called it an Acura model code; it's actually the
     initialism for "Entry-Level Luxury Performance Sedan," a market segment.
   - `"awd"` -> assigned to Acura; it's a generic drivetrain attribute discussed
     across every brand, not brand-specific.
   - `"germans"` -> assigned to BMW at medium confidence, but the model's own
     reasoning said it's a stand-in for German makes generally.
   - `"type"` -> assigned to Lexus at high confidence, but the model's own
     reasoning named Acura (Type-S) and Jaguar (X-Type, not in our anchor list)
     as the more likely referents -- an internally inconsistent answer.

   These are documented, not silently patched, because they show the limits of
   trusting a single LLM call without a verification pass.

### 1.3 Corruption correction (the most consequential finding)

Manual inspection of top candidates surfaced a systematic **mid-word text-injection
corruption** specific to this corpus: brand tokens are spliced into unrelated words,
e.g. `"statistics"` -> `"stat toyota ics"`, `"according"` -> `"ac honda ing"`,
`"unfounded"` -> `"un honda d"`, `"mainstream"` -> `"main honda market"`. `pontiac`
shows a related but distinct artifact (substitution inside model codes like "G35").
Regex substring matching cannot tell a genuine "I love my Toyota" from
"stat**toyota**ics," so raw counts for these three brands are inflated by an unknown,
non-uniform amount.

Rather than hand-writing brittle regex to catch every corruption variant,
`resolve_ambiguous_brands.py` draws a random, reproducible sample (n=100, seed=42)
of raw matches per term and has the LLM classify each occurrence as genuine or
corrupted, using the sample rate as a corrected estimate for the full count:

| Term | Sample genuine rate | Raw message reach | Corrected estimate |
|---|---|---|---|
| toyota | 11% | 1,463 | 161 |
| honda | 30% | 509 | 153 |
| pontiac | 6% | 564 | 34 |

**This changes the answer.** Before correction, Toyota ranks #2 overall and Pontiac
is inside the top 10; Honda is also inside the top 10. After correction, all three
fall well below the top-10 cutoff (which sits at Nissan, reach 197). Reporting the
uncorrected numbers would have meant naming two brands as "important" almost
entirely because of a text-corruption artifact, not because of genuine discussion
volume.

`volkswagen` and `mazda` are also flagged as injection-prone in `cleaning.py`, but
neither is close to the top-10 boundary even at its raw, uncorrected count (137
each), so they weren't sampled -- correcting them cannot change the final answer.

**Stated assumption:** the sampled genuine-rate is applied uniformly to both
`message_reach` and `raw_count` for a term, since the injection mechanism inserts
one spurious token per corrupted message (so the corruption rate in reach and in
raw mentions should track closely). This is a simplification, not a measured fact.

### 1.4 Product attribute discovery

There is no separate attribute pipeline -- `llm_adjudicate.py` classifies every
reviewed candidate as `brand` / `attribute` / `other` in the *same* call that does
brand-variant reconciliation, using a closed set of attribute categories
(`ATTRIBUTE_CATEGORIES`) so canonicalization happens inside the LLM call rather than
as a fragile post-hoc synonym-clustering step. The categories were derived by
inspecting the highest-reach non-brand candidates and grouping surface forms that
clearly describe the same underlying concept (e.g. "mpg"/"gas"/"mileage" -> fuel
economy; "ride"/"brakes"/"suspension" -> handling). `build_attributes.py` filters to
`high`/`medium` confidence and sums counts per category.

Of 273 reviewed candidates: 33 resolved to a brand, 108 to a product attribute, 129
to generic forum discourse (pronouns, filler words, person/place names, ambiguous
tokens) and were discarded from both tables. `output/attribute_term_detail.csv`
lists every candidate term under its category for auditing.

## 2. What counts as "important"

Both brands and attributes are ranked by **message reach** (number of distinct
posts mentioning the entity), not raw mention count. Reach was chosen over raw
count because a handful of users repeating a term many times in one post
shouldn't outweigh the same term appearing once each in many independent posts --
reach better approximates "how much of the conversation touches this," which is
closer to the intuitive meaning of "important" than raw frequency. (`raw_count`
is still reported alongside for transparency.) This choice is also what makes the
corruption correction meaningful: `resolve_ambiguous_brands.py`'s genuine-rate
estimate is defined and applied against reach, not raw count.

## 3. Final results

### 3.1 Top 10 brands (`output/final_brands_top10.csv`)

| Rank | Brand | Message reach | Raw mentions | Distinct surface-form variants merged |
|---|---|---|---|---|
| 1 | BMW | 3,321 | 5,514 | 230 |
| 2 | Acura | 1,214 | 1,833 | 68 |
| 3 | Cadillac | 827 | 1,168 | 20 |
| 4 | Audi | 807 | 1,316 | 66 |
| 5 | Lexus | 719 | 929 | 47 |
| 6 | GM | 437 | 548 | 35 |
| 7 | Infiniti | 400 | 566 | 34 |
| 8 | Mercedes-Benz | 331 | 384 | 22 |
| 9 | Volvo | 266 | 423 | 20 |
| 10 | Nissan | 197 | 305 | 18 |

Just outside the top 10: Lincoln (169), Toyota (corrected, 161), Honda (corrected,
153), Mazda (137), Volkswagen (137), Pontiac (corrected, 34).

Note on GM: forum users referred to Buick, Saab, and Chevrolet mentions as GM
mentions in several LLM-adjudicated cases (Saab and Buick were GM-owned marques
during this corpus's era). This makes "GM" a parent-company rollup rather than a
single nameplate like the other nine -- a modeling choice worth flagging, since it
isn't a perfect peer to "Acura" or "Volvo" as an entity type, but it is how the
corpus itself refers to those sub-brands' owner.

### 3.2 Top product attributes (`output/final_attributes.csv`)

| Rank | Attribute | Message reach | Raw mentions | Example surface forms |
|---|---|---|---|---|
| 1 | Price / value | 1,707 | 2,165 | price, msrp, cost, lease, resale, depreciation, $Xk figures |
| 2 | Performance / powertrain | 1,507 | 1,912 | performance, engine, v6/v8, hp, torque, acceleration |
| 3 | Segment / positioning | 939 | 1,079 | class, segment, luxury, badge, premium, market |
| 4 | Reliability / quality | 866 | 1,014 | problem(s), quality, warranty, reliability, maintenance |
| 5 | Drivetrain configuration | 519 | 649 | awd, rwd, fwd, stick, wheel(s) |
| 6 | Interior / comfort | 505 | 609 | interior, leather, seat, wood, space |
| 7 | Handling / ride | 498 | 550 | tires, handling, suspension, steering, sport |
| 8 | Styling / body type | 375 | 408 | sedan, coupe, size, styling, looks |
| 9 | Technology / features | 303 | 340 | options, features, nav, sunroof, bluetooth |
| 10 | Fuel economy | 102 | 108 | gas, mileage |

No candidate in the reviewed pool resolved to a "safety" category despite it being
offered as an option -- notable in itself: at the frequency levels analyzed, this
luxury-sedan forum's conversation is dominated by price, performance, and brand
positioning, not safety.

## 4. Known limitations

- **Overlapping-span double counting.** A single post like "I love my BMW 3
  series" can independently increment the candidate rows for "bmw," "3 series,"
  and (via the regex model-code pattern) "3 series" again, since noun-chunk
  extraction and regex model-code extraction run over the same text without
  deduplicating spans. After merging, this inflates every brand's reach somewhat.
  It applies fairly uniformly across brands, so relative ranking is more robust
  than any single absolute count -- but absolute counts should be read as
  "at least this much," not exact.
- **Review threshold.** Only candidates with `message_reach >= 30` (273 total)
  were sent to LLM adjudication. Below that, a term is by definition mentioned in
  under 0.5% of posts, so excluding it can't change the top-10 boundary, but it
  does mean some long-tail brand slang or niche attribute terms are not captured.
- **Volkswagen/Mazda corruption unverified.** Flagged as injection-prone by the
  same heuristic that caught Toyota/Honda/Pontiac, but not sampled since neither
  is near the top-10 boundary even uncorrected (see 1.3).
- **Category boundaries aren't perfectly orthogonal.** E.g. "stick"/"manuals"
  (manual transmission) landed in "drivetrain configuration" while "manual"/
  "manual transmission" landed in "performance/powertrain" -- both are defensible
  given each term's specific context, but the two categories overlap conceptually
  for transmission-type terms.
- **Genuine-rate correction is a sampled estimate**, not an exact count (95%
  binomial CI at n=100 is roughly +/-6-9 percentage points depending on the
  observed rate), and assumes the corruption rate is uniform across a term's raw
  count and reach (see 1.3).

# Task B: Co-occurrence Lift Among Task A's Top-10 Brands

**Input:** Task A's final ranked brand list (`../Task_A/output/final_brands_top10.csv`)
and the same forum corpus Task A used (`../Task_A/data/sample_data.csv`, loaded via
Task A's own cleaning pipeline). **Script:** `calculate_lift.py`.

## 1. What "lift" means here

Standard market-basket association measure, computed per unordered brand pair:

```
lift(A, B) = P(A and B) / (P(A) * P(B)) = (N * n_both) / (n_A * n_B)
```

`N` = total cleaned messages, `n_A`/`n_B` = messages mentioning that brand,
`n_both` = messages mentioning both. `lift > 1` means the pair is discussed
together *more* than chance predicts (cross-shopped / compared brands);
`lift < 1` would mean less than chance; `lift ~= 1` means no association. This
is the exact same formula Task A already used and validated internally for its
own candidate-to-brand lift signal (`canonicalize.lift_assign_candidate`), just
applied brand-to-brand instead of candidate-to-brand.

## 2. Method: reuse, don't reinvent

Rather than writing a new presence test, this script imports directly from
Task A:

- `extract_candidates.load_and_clean_data` -- same corpus, same quote-artifact
  stripping and burst-repeat collapsing Task A already validated.
- `canonicalize.compute_brand_presence` -- same `\banchor\b` word-boundary
  regex Task A already used to flag whether a brand's anchor token appears in a
  given message.

A brand is counted as "in" a message only if its literal anchor name appears
as a whole word (case-insensitive). No LLM call, no fuzzy matching -- every
number in this report can be reproduced by re-running `calculate_lift.py` and
is traceable to an exact regex match.

### 2.1 Why anchor-only, not the resolved slang/model-code variants too

Task A's final brand *counts* (`final_brands_top10.csv`) also fold in
LLM-resolved indirect references -- "e46"/"bimmer" -> BMW, "tl" -> Acura,
"mb" -> Mercedes-Benz, "ls"/"is350" -> Lexus, "buick"/"saab"/"chevrolet" -> GM,
etc. Those resolutions were validated by an LLM one candidate at a time, with
forum excerpts as context, specifically for ranking brand importance. Blindly
re-applying an ambiguous short token like "ls" or "mb" as a full-corpus regex
here -- with no per-occurrence context check -- would reintroduce exactly the
spurious-correlation risk Task A's own report warns about (e.g. "forum" showing
lift=3.89 toward Lincoln, "fwd" showing lift=3.87 toward Nissan, both
coincidence). Sticking to one unambiguous anchor token per brand keeps every
co-mention counted here real and auditable, at the cost of undercounting
brands (chiefly BMW) that are frequently referenced by model code or slang
alone. This is a deliberate transparency-over-completeness tradeoff, not an
oversight.

### 2.2 Why the per-brand counts here don't match `final_brands_top10.csv`

| Brand | Task A `message_reach` | Task B `n_messages` (this script) |
|---|---|---|
| bmw | 3,321 | 2,021 |
| acura | 1,214 | 778 |
| audi | 807 | 770 |
| lexus | 719 | 616 |
| infiniti | 400 | 528 |
| mercedes-benz | 331 | 312 |
| volvo | 266 | 282 |
| nissan | 197 | 272 |
| gm | 437 | 269 |
| cadillac | 827 | 258 |

Two different pipelines produce these, and neither is a strict superset of the
other:

- Task A's `message_reach` comes from spaCy noun-chunk **candidate extraction**
  (anchor substring merge + LLM-resolved variants), which (a) adds in slang/model
  variants this script deliberately excludes (inflates BMW, Acura, Cadillac,
  Audi, Lexus, GM, Mercedes-Benz relative to anchor-only), but (b) can *miss*
  a plain mention if spaCy's noun-chunker didn't happen to carve it out as its
  own chunk (Task A's own report flags overlapping-span handling as a known
  limitation). This is likely why Infiniti and Nissan -- brands with few or no
  LLM-resolved slang variants (see `llm_adjudication_results.csv`) -- come out
  *higher* here than in Task A's reach column: Task B's direct full-text regex
  scan catches plain "infiniti"/"nissan" mentions the chunk-based pipeline missed.
- Task B's `n_messages` is a direct, unconditional regex search of the raw
  cleaned message text, independent of noun-chunk parsing.

The brand *ranking* is broadly consistent between the two (BMW is dominant
either way), but absolute counts and even relative order among the smaller
brands shift. Read both tables as directionally informative, not as
interchangeable ground truth for a brand's "true" mention count.

## 3. Results

**Corpus size after cleaning:** 5,991 messages (same cleaning as Task A;
9 empty messages dropped).

**Per-brand message presence** (`output/brand_presence_counts.csv`):

| Brand | Messages | Share of corpus |
|---|---|---|
| bmw | 2,021 | 33.7% |
| acura | 778 | 13.0% |
| audi | 770 | 12.9% |
| lexus | 616 | 10.3% |
| infiniti | 528 | 8.8% |
| mercedes-benz | 312 | 5.2% |
| volvo | 282 | 4.7% |
| nissan | 272 | 4.5% |
| gm | 269 | 4.5% |
| cadillac | 258 | 4.3% |

**Every one of the 45 brand pairs has lift > 1** -- in this forum, any two
luxury brands are mentioned together more than chance would predict, which
makes sense for a comparison-shopping context (posters routinely cross-shop
and cross-reference competing marques in the same message).

Top 5 pairs by lift (`output/brand_pair_lift.csv`, full table there):

| Brand A | Brand B | n_both | lift |
|---|---|---|---|
| Cadillac | GM | 65 | 5.61 |
| Mercedes-Benz | Volvo | 59 | 4.02 |
| Infiniti | Nissan | 82 | 3.42 |
| Lexus | Infiniti | 176 | 3.24 |
| Lexus | Mercedes-Benz | 102 | 3.18 |

Bottom 5 pairs by lift:

| Brand A | Brand B | n_both | lift |
|---|---|---|---|
| Acura | GM | 38 | 1.09 |
| BMW | Nissan | 95 | 1.04 |
| GM | Nissan | 17 | 1.39 |
| GM | Infiniti | 33 | 1.39 |
| BMW | Volvo | 133 | 1.40 |

**Interpretation:**

- **Cadillac-GM (5.61x)** is the strongest pair by a wide margin, and is the
  one result that isn't really "cross-shopping" -- Task A's report already
  notes GM is a parent-company rollup (Buick/Saab/Chevrolet mentions folded
  into "gm"), and Cadillac is itself a GM marque, so this pair is largely
  capturing posts discussing GM's product lineup, not two independent brands
  being compared.
- **Infiniti-Nissan (3.42x)** is a genuine parent/luxury-division pair
  (Infiniti is Nissan's luxury marque), so a high lift here is expected and
  validates the method -- known corporate-family pairs surface at the top.
- **Mercedes-Benz-Volvo, Lexus-Infiniti, Lexus-Mercedes-Benz** are all
  plausible entry-luxury/near-luxury cross-shopping pairs, consistent with the
  segment discussion Task A's attribute analysis found dominant in this forum
  (price/value and segment/positioning were the top two attribute categories).
- **BMW pairs cluster at the low end of lift** (BMW-Nissan 1.04, BMW-Acura
  1.31, BMW-GM 1.49, BMW-Volvo 1.40) not because BMW is discussed *less* with
  these brands in absolute terms (BMW co-occurs with everything a lot, simply
  by being mentioned in a third of all messages), but because lift measures
  co-occurrence *relative to* each brand's own base rate. BMW's presence is so
  broad (33.7% of the corpus) that it approaches statistical independence from
  every other brand -- a large `n_both` can still produce a lift near 1 if
  `n_A` alone is already large. This is the correct reading of lift, not an
  artifact: BMW is the dominant, most-discussed brand overall, but it isn't
  *selectively* paired with any one competitor.

## 4. Known limitations

- **Anchor-only undercounts brands with heavy slang/model-code usage**,
  chiefly BMW (see 2.2). If slang variants were included, BMW's `n_messages`
  and its lift with every other brand would shift upward; the *relative*
  ordering of BMW's pairings against each other would likely be more stable
  than the absolute lift values.
- **"gm" as an anchor is a parent-company rollup**, not a single nameplate
  (inherited directly from Task A's `BRAND_ANCHORS` design choice, see Task A
  REPORT.md section 3.1) -- so GM's pairings, especially GM-Cadillac, should
  be read as "GM-family discussion," not brand-to-brand cross-shopping.
- **No corruption correction applied.** Task A found mid-word text-injection
  corruption for `toyota`/`honda`/`pontiac` specifically; none of those three
  are in the top 10, and none of the ten anchors used here were flagged as
  injection-prone in Task A's cleaning step, so no correction was needed.
- **Small `n_both` cells produce noisier lift estimates.** Pairs like
  Cadillac-Nissan (`n_both`=19) or Volvo-Nissan (`n_both`=19) rest on a small
  absolute number of co-occurring messages; their lift values are more
  sensitive to a handful of messages than pairs like BMW-Audi (`n_both`=408).
  `output/brand_pair_lift.csv` reports `n_both` alongside lift for exactly
  this reason -- read low-`n_both` lift values cautiously.

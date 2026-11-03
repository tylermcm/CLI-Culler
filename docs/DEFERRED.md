# Deferred cases: revisit after all milestones

Unusual cases noticed while tuning, deliberately left until every stage (faces, MANIQA, Q-ReAlign, TinyCLIP,
people coverage) is in, so they are judged against the full pipeline rather than patched one at a time.
Each has a regression test or a measurement to re-check against.

## 1. Coloured light sources count as clipped highlights (China 8-26, `_DSC1055`): **done (option 1)**

- **What:** a portrait framed by a glowing yellow light ring; the face is well exposed. A final pick.
- **Measured** (any channel >= 250, the current rule): 44% of pixels clipped, largest region 24%, top third 53–59%.
  Counting only pixels clipped in all three channels (blown to white): 0.5%, 0.3%, 0.6%. The "clipped"
  pixels are saturated yellow (median R 255, G 243, B 89), not lost detail.
- **Now:** passes because the gates were loosened to `clip_region_reject` 0.35 / `sky_blown` 0.75; it still
  takes a ~0.10 score penalty. Pinned by `test_default_gates_pass_china_picks_with_margin`.
- **Options when revisiting:**
  1. Count only near-white pixels as clipped for the gates; treat saturated colour as a score, not a failure.
  2. Judge exposure on the subject: when MediaPipe finds a well-exposed, unclipped face, highlight gates
     cannot reject (at most a penalty).
  3. Scene type via TinyCLIP (night portrait with practical lights, high-key, silhouette) if 1–2 leave misses.
- **Check:** re-run `evaluate` on China 8-26 and every other shoot with truth; no pick may be gated.
- **Done:** gates count only pixels blown to white (every channel near 255); saturated colour (`sat_region`) costs
  the clip penalty only. `_DSC1055`: 0.8% blown. No pick gated on any shoot. Its scene-mates are no longer
  wrongly gated either, so it now loses its scene on sharpness (RESULTS.md); options 2–3 are not needed for gating.

## 2. Bracketeering warnings send too much to review (Alaska Aug '21): **done**

- **What:** 36% of Alaska's files go to review, almost all because of "partial bracket: N of 7 slots" (and
  "frames don't register"). The spec sends any warned group to review. 19 of Alaska's 20 picks are among them;
  without the rule 18 of 20 would be automatic keeps.
- **Options:** treat "partial bracket" as information, not doubt about the grouping (only low confidence and
  grouping-doubt warnings go to review); or review only when the warned group would be rejected.
- **Check:** Alaska review rate and keep-or-review recall vs plain recall.
- **Done:** `review_warnings: doubt` (default): only a weak bracket candidate, a flagged auto-bracket with no
  partners, or an unknown warning sends a group to review; the rest are noted. Alaska review 35.9% → 2.0%.

## 3. Brackets of a moving subject are alternatives, not HDR sets (Alaska Aug '21): **done**

- **What:** the wildlife was shot with auto-bracketing on. Each pick is one frame of a bracket; `keep_set` keeps
  the whole bracket and `pick_one` keeps the best-exposed frame, which may not be the picked one.
- **Options:** a bracket bracketeering flags "frames don't register (subject or camera moving)" competes like a
  burst (its frames ranked against each other, never on exposure alone); or a `brackets: as_burst` setting.
- **Check:** Alaska recall with picks in `Bears/`, `Eagles/`, `Re-edits/`.
- **Done:** `moving_brackets: as_burst` (default). Alaska recall 5% → 75% (with §2).

## 4. China 8-26 precision against ImageTriage decisions is 18%: **open, needs a decision**

- The engine keeps 46% of the frames you decided on; you accepted 12%. Lone frames (kept with nothing to compare
  against), whole brackets, and the second burst frame (`bursts_keep` 2) make up most of it. Revisit with keep
  counts (§8 question 3) and an absolute-quality floor once Q-ReAlign is in.

- **Measured** (RESULTS.md): lone frames, kept because nothing competes with them, are 39–78% of every shoot's
  keeps; `bursts_keep: 1` gains China 2 points of precision and costs Alaska 10 points of recall.
- **The decision:** "reject only clear failures" keeps every lone frame that passes the gates. Raising precision
  means rejecting some frames that failed nothing, by one of: an absolute quality floor (needs a model: MANIQA or
  Q-ReAlign, so only with `--with`); sending weak lone frames to review instead of keep; or a keep budget per
  shoot (compares across the shoot, against the first rule). None is implemented.

- **Since:** the quality floor is implemented (`gates.quality_reject` / `quality_review`, MANIQA on every frame
  about to be kept), off by default: MANIQA's low end mixes junk with good dark, misty or glare-lit frames
  (RESULTS.md). Next: the photographer's labelled baseline (about 100 frames across 5–10 shoots, `ai-culler
  sample`), then `ai-culler calibrate` for the levels.

## 5. MANIQA judges content, not only quality

- **What:** it scores night scenes, mist, shallow depth of field and water glare low however sharp the frame is
  (Alaska's bear close-ups at 0.145–0.19, under China's lowest ImageTriage keep, 0.228).
- **Tested and ruled out:** pairing a low MANIQA score with a soft frame. On the frames MANIQA scores lowest, neither whole-frame sharpness
  (bears at the 5–6th percentile of their shoot, blurred photos of a screen at 57–62%) nor the sharpest tiles of the frame (bears
  5–17%, screen photos 63%) separates good from bad. A floor relative to the shoot would not help either: the bears are
  in the bottom 0–2% of a shoot that is almost all wildlife, so even a per-category percentile ranks them last.
- **Options after the baseline:** category-relative percentiles (TinyCLIP zero-shot categories, EXIF technique
  flags), compared against a single global level on the labels; or MANIQA sends frames to review only, never rejects.

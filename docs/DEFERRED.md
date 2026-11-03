# Deferred cases: revisit after all milestones

Unusual cases noticed while tuning, deliberately left until every stage (faces, MANIQA, Q-ReAlign, TinyCLIP,
people coverage) is in, so they are judged against the full pipeline rather than patched one at a time.
Each has a regression test or a measurement to re-check against.

## 1. Coloured light sources count as clipped highlights (China 8-26, `_DSC1055`)

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

## 2. Bracketeering warnings send too much to review (Alaska Aug '21)

- **What:** 36% of Alaska's files go to review, almost all because of "partial bracket: N of 7 slots" (and
  "frames don't register"). The spec sends any warned group to review. 19 of Alaska's 20 picks are among them;
  without the rule 18 of 20 would be automatic keeps.
- **Options:** treat "partial bracket" as information, not doubt about the grouping (only low confidence and
  grouping-doubt warnings go to review); or review only when the warned group would be rejected.
- **Check:** Alaska review rate and keep-or-review recall vs plain recall.

## 3. Brackets of a moving subject are alternatives, not HDR sets (Alaska Aug '21)

- **What:** the wildlife was shot with auto-bracketing on. Each pick is one frame of a bracket; `keep_set` keeps
  the whole bracket and `pick_one` keeps the best-exposed frame, which may not be the picked one.
- **Options:** a bracket bracketeering flags "frames don't register (subject or camera moving)" competes like a
  burst (its frames ranked against each other, never on exposure alone); or a `brackets: as_burst` setting.
- **Check:** Alaska recall with picks in `Bears/`, `Eagles/`, `Re-edits/`.

## 4. China 8-26 precision against ImageTriage decisions is 18%

- The engine keeps 46% of the frames you decided on; you accepted 12%. Lone frames (kept with nothing to compare
  against), whole brackets, and the second burst frame (`bursts_keep` 2) make up most of it. Revisit with keep
  counts (§8 question 3) and an absolute-quality floor once Q-ReAlign is in.

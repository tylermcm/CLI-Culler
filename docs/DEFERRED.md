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

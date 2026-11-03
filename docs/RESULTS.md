# Results: China 8-26 (truth: 18 final picks in `_winners/`)

One shoot, 2,304 files, 1,137 captures. With 18 picks each pick is 5.6 points of recall: read differences of
one or two picks as noise. Measured on a V100 (MANIQA) and CPU (faces), `evaluate --truth picks`.

| Run | Models | Recall | Burst hits | Scene hits | Keep rate | Picks gated | Runtime |
|---|---|---|---|---|---|---|---|
| Milestone 1 | none | 61.1% (11/18) | 1/2 | 3/9 | 39.0% | 1 | 68 s |
| + region clipping gates | none | 72.2% (13/18) | 2/2 | 5/9 | 39.3% | 0 | 75 s |
| + upright previews | none | 72.2% (13/18) | 2/2 | 5/9 | 39.2% | 0 | 77 s |
| Milestone 2 | faces (MediaPipe), MANIQA | 66.7% (12/18) | 2/2 | 4/9 | 39.4% | 0 | 158 s (faces 128 s, MANIQA 208 s first run, cached after) |

Not yet measured: Q-ReAlign (milestone 3), TinyCLIP scenes and preference (4), AdaFace people coverage (5);
those models are not installed on this machine (install them with the app's Set Up AI).

**Milestone 2 reading.** Faces and MANIQA swapped one pick for another in scene 91 (both frames were picks)
and lost `_DSC1115` to a burst MANIQA rated higher. The other contest misses (DSC_1360, _DSC0570, _DSC0808,
_DSC1088) are frames the user preferred to a technically better neighbour; MANIQA and face sharpness agree with
whole-frame sharpness against the pick. Choosing a moment over a sharper twin is what the appeal stage
(Q-ReAlign) is for. Of 1,137 captures, 836 are candidates, 615 contested; MANIQA ran on 617 (0.34 s/frame).

## Four shoots (the bracketeering review shoots), no models vs faces + MANIQA

Truth: China 8-26 has two (18 final picks in `_winners/`; 246 ImageTriage decisions in its XMP sidecars, 29
accepted / 217 rejected, so precision is measurable there). Alaska Aug '21: 20 picks in `Bears/`, `Re-edits/`,
`Eagles/` (its card folders and `Tyler's photos/` are copies, not picks). Japan 4-24 and WhiteSands '26 have no
truth: only the engine's own rates are reported.

| Shoot | Truth | Run | Recall | Keep or review | Precision | Bursts | Scenes | Keep rate | Review rate |
|---|---|---|---|---|---|---|---|---|---|
| China 8-26 | picks (18) | none | 72.2% | 72.2% | n/a | 2/2 | 5/9 | 39.2% | 5.7% |
| | | faces + MANIQA | 66.7% | 66.7% | n/a | 2/2 | 4/9 | 39.4% | 5.7% |
| China 8-26 | ImageTriage (29 of 246) | none | 72.4% | 72.4% | 18.3% | 2/5 | 4/8 | 46.8% | 0% |
| | | faces + MANIQA | 72.4% | 72.4% | 18.4% | 2/5 | 5/8 | 46.3% | 0% |
| Alaska Aug '21 | picks (20) | none | 5.0% | **100%** | n/a | – | 0/3 | 36.5% | **35.9%** |
| | | faces + MANIQA | 5.0% | **100%** | n/a | – | 0/3 | 36.6% | **35.9%** |
| Japan 4-24 | – | either | – | – | – | – | – | 66.7% | 17.0% |
| WhiteSands '26 | – | either | – | – | – | – | – | 43.0% | 11.3% |

Runtime with faces + MANIQA (first run; cached after): Alaska 7 min for 2,707 captures (517 contested), WhiteSands
5 min (580), Japan 2 min (126), China 2.5 min (617).

**Reading.** Faces and MANIQA move one or two frames per shoot either way: no measurable gain yet. Where the engine
misses on China it is choosing a technically better neighbour over the moment the user chose (Q-ReAlign's job).
On Alaska no pick was rejected: 19 of 20 went to review, because every pick is a frame of a camera auto-bracket
(wildlife shot with bracketing on) carrying a "partial bracket" warning; see `DEFERRED.md` §2-3.

## Four shoots, core cull (no models) after the DEFERRED §1–3 fixes

Defaults now: no models; `review_warnings: doubt` (§2); `moving_brackets: as_burst` (§3); highlight gates count
only pixels blown to white, saturated colour costs score (§1). Runtime with no models: 1.5 min (China, Japan,
WhiteSands) to 2 min (Alaska) on first run, clipping re-measured; seconds after.

| Shoot | Truth | Before (milestone 2, no models) | Now | Keep rate | Review rate |
|---|---|---|---|---|---|
| China 8-26 | picks (18) | 72.2% | 66.7% | 39.2% → 42.9% | 5.7% → 0.6% |
| China 8-26 | ImageTriage (29 of 246) | recall 72.4%, precision 18.3% | recall 69.0%, precision 17.5% | 46.8% → 46.3% | 0% |
| Alaska Aug '21 | picks (20) | **5.0%** (100% keep or review) | **75.0%** | 36.5% → 56.6% | **35.9% → 2.0%** |
| Japan 4-24 | – | – | – | 66.7% → 81.9% | 17.0% → 0.8% |
| WhiteSands '26 | – | – | – | 43.0% → 50.4% | 11.3% → 2.8% |

No pick or ImageTriage keep is rejected by a gate on any shoot.

**China's lost pick is `_DSC1055`**, and the old result was luck: the four other frames of its scene (181)
share the yellow light ring and were *rejected by the clipping gate* (36–52% "clipped"). With saturated colour
no longer a failure they compete again, and three are sharper and better exposed. It is now the same kind of
miss as the other five (DSC_1360, _DSC0570, _DSC0808, _DSC0822, _DSC1088): a moment the user chose over a
technically better neighbour, which no cheap signal can see (`--with appeal` is for this).

**Alaska**: the moving-bracket rule ranks each wildlife bracket's frames like a burst, and "partial bracket" is
no longer a reason for review. 15 of 20 picks are automatic keeps.

**Keep counts (§4).** `bursts_keep: 1` instead of 2: China precision 17.5% → 19.8% (recall unchanged), but Alaska
recall 75% → 65% (moving brackets keep one frame). The default stays 2. What the engine keeps, by reason:

| Shoot | Kept captures | Lone frame (only unit in its scene) | Bracket (keep_set) | Burst / moving-bracket survivors | Scene winners | Other |
|---|---|---|---|---|---|---|
| China 8-26 | 488 | 53% | 14% | 15% | 18% | – |
| Alaska Aug '21 | 1,533 | 60% | 27% | 7% | 3% | 2% sequences |
| Japan 4-24 | 435 | 78% | 5% | 13% | 3% | – |
| WhiteSands '26 | 462 | 39% | 8% | 36% | 6% | 11% sequences |

Lone frames that pass every gate are kept because nothing competes with them ("reject only clear failures"):
they bound precision. See DEFERRED §4.

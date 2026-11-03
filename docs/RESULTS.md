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

## Quality floor: MANIQA on every competing frame (four shoots)

`maniqa_scope: all`, no floor applied: MANIQA on every competing frame (China 825, Alaska 1,002, Japan 358,
WhiteSands 729; 1–4 min per shoot on the V100). Distribution, and where your own culls sit in it:

| | n | min | p5 | p10 | median |
|---|---|---|---|---|---|
| China 8-26, all | 825 | 0.088 | 0.209 | 0.247 | 0.395 |
| China 8-26, final picks | 18 | 0.288 | | 0.316 | 0.406 |
| China 8-26, ImageTriage keeps | 29 | 0.228 | 0.240 | 0.247 | 0.388 |
| China 8-26, ImageTriage rejects | 216 | 0.184 | 0.212 | 0.243 | 0.374 |
| Alaska Aug '21, all | 1,002 | 0.115 | 0.222 | 0.258 | 0.355 |
| Alaska Aug '21, picks | 12 | 0.337 | | 0.338 | 0.368 |
| Japan 4-24, all | 358 | 0.095 | 0.218 | 0.277 | 0.389 |
| WhiteSands '26, all | 729 | 0.062 | 0.262 | 0.312 | 0.450 |

**Reading.** Keeps and rejects overlap almost completely above ~0.23: MANIQA cannot choose between good frames.
Below, it finds real junk (blurred photos of a screen, a dark out-of-focus interior, an accidental tilted frame),
but also good frames it dislikes for their content: sharp bear close-ups against water glare (DSC_9398, 0.145;
DSC_9815, 0.184), a night portrait (_DSC1739, 0.133), mist (_DSC1694, 0.195), the torii path (DSC_6945, 0.177).
No absolute reject level from these four shoots is safe, so both levels default to off; set them from a
labelled baseline (`ai-culler sample` / `calibrate`).

## Floor modes compared on existing culls (first look, before the labelled baseline)

Library baseline (`ai-culler baseline`) over the four shoots: 2,066 competing frames with MANIQA and TinyCLIP
categories (landscape 752, architecture 357, night 331, street 265, animals 188, interior 148, food_product 145,
portrait 133, macro 117, people 79). Category check by eye: right for portraits, people, night, landscapes,
architecture, streets and interiors; flower fields come out as "macro".

Labels from what already exists: China's ImageTriage decisions and picks, Alaska's picks; 246 of them are frames
the floor can judge (Alaska's picks are almost all bracket frames, which it never judges): 31 keep, 215 reject.

| Mode | Rejects caught, every keep safe | Rejects caught, 95% of keeps safe |
|---|---|---|
| `raw` (A) | 16 of 215 | 21 |
| `shoot` | 16 | 21 |
| `category` (B) | 12 | 25 |
| `technique` (C) | **25** | **26** |

**Reading, with care.** `technique` catches the most with every keep safe, but these are 31 keeps, nearly all from
one shoot: a difference of a few frames is noise. And ImageTriage's rejects are mostly frames that lost to a better
neighbour, not junk: no floor should catch them, which is why every mode catches few. The labelled baseline
(keep / borderline / reject judged frame by frame, spread across categories) is the real test.

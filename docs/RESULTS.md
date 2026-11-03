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

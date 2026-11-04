# Setting up the quality floor

The floor judges every frame a cull is about to keep with MANIQA, Q-ReAlign image quality, or both, and rejects
or reviews the ones below a level you set. It is off until you set levels. This is how to choose them and check
them before letting it reject anything.

```bash
cd ~/Documents/cli-culler
M=/path/to/ImageTriage            # or omit --models when image_triage is installed
DB="--db culler.sqlite"
SHOOTS=("/shoots/A" "/shoots/B" ...)   # 5-10 shoots, as varied as your work
```

## 1. A library baseline (no labels)

```bash
.venv/bin/ai-culler $DB baseline "${SHOOTS[@]}" --qrealign --out library.json --models $M
```

Every competing frame of every shoot: MANIQA, Q-ReAlign quality and aesthetics, TinyCLIP categories, EXIF technique
flags. About 1 s per frame on a V100 the first time; cached after. More shoots make the per-category comparisons
steadier (a category is used once it has 50 frames).

## 2. Label a sample of what the floor will judge

```bash
.venv/bin/ai-culler $DB sample "${SHOOTS[@]}" --n 150 --baseline library.json --settings my_settings.json \
    --out ~/baseline --models $M
```

The sample is drawn from the frames the floor would actually judge in *your* cull (`--settings`): would-be keeps
(lone frames, burst and scene winners) and each set's runner-up, spread across categories. Open
`~/baseline/label.html` and judge each frame on its own, as if it were the only frame of its moment: 1 keep,
2 borderline, 3 reject, 4 hero. Scores are hidden. Export `labels.csv` into `~/baseline`.

## 3. Compare the floors and pick one

```bash
.venv/bin/ai-culler calibrate ~/baseline --baseline library.json --out ~/baseline/calibration.md
```

For MANIQA, Q-ReAlign quality and aesthetics, each compared as the raw score or as a percentile (in its shoot, its
category, its category and technique), and for MANIQA with Q-ReAlign quality (`either`, `both`, `agree` = reject
only when both agree, review when they disagree), the report gives:

- **All keeps safe**: rejects caught at the highest level with none of your keeps below it. Optimistic: the level
  is fitted on the same labels.
- **Held out**: each shoot judged with a level fitted on the others. The honest number. Keeps lost here mean the
  level does not carry over to a new shoot.
- **Review**: what each arm sends to you instead of deciding.
- **Breakdowns** by role (lone frame, winner, runner-up), category and technique.

Prefer the arm with zero keeps lost held out, then the most rejects caught, then the least review. Q-ReAlign
aesthetics is reported for information; leave it to ranking unless it clearly wins. Within-shoot percentiles rank
the weakest frames of a shoot whether or not they are bad: treat that mode as a comparison, not a choice.

**How sure is it?** With n keeps labelled and none lost, the false-reject rate is below about 3/n (95%): 80 keeps
only shows "under 4%"; "under 0.5%" needs about 600. Which is why step 4.

## 4. Put it in your settings, review first

```json
{
  "models": ["maniqa", "appeal"],
  "gates": {
    "quality_signal": "agree",
    "quality_mode": "technique",
    "quality_baseline": "library.json",
    "quality_reject": 0.05, "quality_review": 0.12,
    "qra_reject": 0.05, "qra_review": 0.12,
    "quality_unique": "review"
  }
}
```

(Numbers are placeholders: use calibrate's.) `quality_unique: review` (the default) sends a moment's best frame (a
lone frame, or the best of a burst or scene that all failed) to review instead of rejecting it; only redundant
frames, with a better alternative that passes, are rejected.

## 5. Check it against every shoot you have already culled

Your past decisions are a much larger check than the labelled sample: ImageTriage's keep/reject decisions
(`--truth triage`) and your exported picks (`--truth picks`).

```bash
for s in "${PAST_SHOOTS[@]}"; do
  .venv/bin/ai-culler $DB evaluate "$s" --truth triage --settings my_settings.json --models $M \
      --decisions "out/$(basename "$s").json" --rerun --out "out/$(basename "$s").md"
done
```

Each report has a **Quality floor vs your kept frames** section: how many of your keeps the floor rejected (must be
0), which it sent to review, and how many of your rejects it caught. The one-line summary printed per shoot carries
the same numbers. Look at every keep it rejected or reviewed before trusting it. Only when it has lost none across
many shoots, consider `quality_unique: reject` or tighter levels.

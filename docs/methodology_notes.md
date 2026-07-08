# Methodology Notes

Working notes on decisions made in the preprocessing pipeline, kept separate
from the Milestone 2 report so they can be updated as the pilot session
data comes in without touching the submitted document.

## Face detector choice (Haar cascade vs. MTCNN)

`scripts/face_detect_align.py` currently uses OpenCV's bundled Haar cascade
detector rather than MTCNN. This is a deliberate scaffolding decision, not
the final choice described in the Milestone 2 architecture: Haar has zero
extra dependencies and lets the rest of the pipeline (normalisation, label
mapping, folder structure) be built and tested immediately. Before the
pilot session, swap in an MTCNN-based `detect_face_mtcnn()` function using
`facenet-pytorch`, since Haar is known to be less robust to the off-axis
head poses expected during a coding task (participants looking at a second
monitor, leaning toward the keyboard, etc.).

## Likert-to-class thresholds

The cut points in `scripts/label_mapping.py` (`<=3` low, `4–6` moderate,
`>=7` high) are a starting assumption based on typical 10-point stress
Likert scale conventions, not yet validated against the pilot session
data. After the first 10 pilot participants, re-derive the thresholds by
comparing the Likert distribution against each participant's PSS-10
percentile band and adjust `LIKERT_THRESHOLDS` accordingly, then re-run
label mapping on the full dataset. Document any change here.

## Known limitation: single-annotator self-report

At this stage, only the participant's own concurrent Likert rating is
mapped to a class label. The Milestone 2 report commits to a second,
independent annotator rating a 20% subsample for facial affect, blind to
the self-report score, so inter-annotator agreement (Cohen's kappa) can be
computed. That second annotation stream is not yet implemented in
`label_mapping.py` — it will be added as `--secondary_csv` once the first
pilot batch is annotated.

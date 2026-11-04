"""Preference learning on TinyCLIP embeddings (milestone 4).

The original cli-culler preference model (IncrementalPCA -> StandardScaler -> logistic SGD, alpha 1e-4,
random_state 13) and its leave-one-out diagnostics, trained on TinyCLIP image embeddings with labels from your
past culls (any ``evaluate`` truth source: picks, csv, triage, ...). Classes are balanced, since final picks are
a few percent of a shoot. ``score`` gives the probability you keep a frame; ``decide`` uses it as one more
signal, weighted low (``weights.preference``) until the harness shows it helps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ALPHA = 0.0001
RANDOM_STATE = 13


@dataclass
class PreferenceModel:
    projector: object
    scaler: object
    classifier: object
    embedding_revision: str  # TinyCLIP scorer revision the model was trained on
    examples: int
    positives: int
    diagnostics: dict = field(default_factory=dict)

    def score(self, embeddings: np.ndarray) -> np.ndarray:
        x = self.scaler.transform(self.projector.transform(np.asarray(embeddings, dtype=np.float32)))
        return self.classifier.predict_proba(x)[:, 1]

    def save(self, path: str | Path) -> Path:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: str | Path) -> "PreferenceModel":
        import joblib

        model = joblib.load(Path(path).expanduser())
        if not isinstance(model, PreferenceModel):
            raise ValueError(f"{path} is not an aiculler preference model")
        return model


def train(embeddings: np.ndarray, labels: np.ndarray, *, embedding_revision: str, projected_dim: int = 64) -> PreferenceModel:
    from sklearn.decomposition import IncrementalPCA
    from sklearn.linear_model import SGDClassifier
    from sklearn.preprocessing import StandardScaler

    x = np.asarray(embeddings, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    if len(x) < 4 or len(set(y.tolist())) < 2:
        raise ValueError("need at least 4 labelled frames, with both kept and not-kept examples")
    n_components = min(projected_dim, x.shape[0], x.shape[1])
    projector = IncrementalPCA(n_components=n_components)
    scaler = StandardScaler()
    projected = scaler.fit_transform(projector.fit_transform(x))
    classifier = SGDClassifier(loss="log_loss", penalty="l2", alpha=ALPHA, max_iter=1000, tol=1e-3,
                               random_state=RANDOM_STATE, class_weight="balanced")
    classifier.fit(projected, y)
    model = PreferenceModel(projector, scaler, classifier, embedding_revision, int(len(y)), int(y.sum()))
    model.diagnostics = _diagnostics(projected, y)
    return model


def _diagnostics(projected: np.ndarray, y: np.ndarray) -> dict:
    """Held-out ranking quality: leave-one-out (the existing routine) up to 300 examples, else 10-fold."""
    from sklearn.linear_model import SGDClassifier

    n = len(y)
    held: dict[int, float] = {}
    if n <= 300:
        held = _leave_one_out(projected, y)
        method = "leave-one-out"
    else:
        rng = np.random.default_rng(RANDOM_STATE)
        folds = np.array_split(rng.permutation(n), 10)
        for fold in folds:
            train_idx = np.setdiff1d(np.arange(n), fold)
            if len(set(y[train_idx].tolist())) < 2:
                continue
            clf = SGDClassifier(loss="log_loss", penalty="l2", alpha=ALPHA, max_iter=1000, tol=1e-3,
                                random_state=RANDOM_STATE, class_weight="balanced")
            clf.fit(projected[train_idx], y[train_idx])
            for i, s in zip(fold, clf.decision_function(projected[fold])):
                held[int(i)] = float(s)
        method = "10-fold"
    if not held:
        return {"method": method, "auc": None}
    idx = np.array(sorted(held))
    scores = np.array([held[i] for i in idx])
    return {"method": method, "auc": _auc(scores, y[idx]), "held_out": int(len(idx))}


def _auc(scores: np.ndarray, y: np.ndarray) -> float | None:
    """Probability a random kept frame scores above a random not-kept one (ties count half)."""
    pos, neg = scores[y == 1], scores[y == 0]
    if not len(pos) or not len(neg):
        return None
    greater = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return round(float(greater) / (len(pos) * len(neg)), 4)


def learn_preference(
    shoots: list[str | Path],
    truth: str,
    registry,
    *,
    truth_file: str | Path | None = None,
    session: str | None = None,
    cache_db: str | Path | None = None,
    cache_dir: str | Path | None = None,
    backend: str = "auto",
    progress=None,
) -> tuple[PreferenceModel, list[dict]]:
    """Train a preference model on TinyCLIP embeddings of past culls. Each folder in ``shoots`` is one shoot
    (with its pick/export folders inside it for ``truth="picks"``). Returns the model and one summary per shoot."""
    from aiculler.ingest import PreviewCache, ingest
    from aiculler.scorers import ScoreStore, run_scorers
    from aiculler.scorers.models import TinyClipScorer
    from aiculler.storage import ScoreCache
    from aiculler.truth import load_truth, merge_truths

    if not registry.available("tinyclip"):
        raise ValueError(f"TinyCLIP is needed to learn preferences: {registry.problems().get('tinyclip')}")
    scorer = TinyClipScorer(registry)
    if cache_dir is None:
        cache_dir = Path(cache_db).with_suffix("").as_posix() + "_previews" if cache_db else ".aiculler_cache"
    previews = PreviewCache(cache_dir)
    cache = ScoreCache(cache_db) if cache_db else ScoreCache(":memory:")
    xs: list[list[float]] = []
    ys: list[int] = []
    summary: list[dict] = []
    try:
        for folder in shoots:
            shoot = ingest(folder, backend=backend)
            store = ScoreStore()
            readable = [cid for cid, c in shoot.captures.items() if not c.error]
            run_scorers(store, shoot, readable, [scorer], previews, cache, progress)
            doc = {"files": {p: {"capture": f.capture, "primary": f.primary} for p, f in shoot.files.items()}}
            ts = load_truth(truth, doc, truth_file=truth_file, session=session)
            kept = labelled = 0
            for cid in readable:
                emb = (store.details.get(cid) or {}).get("embedding")
                if emb is None or cid in ts.excluded:
                    continue
                truths = [ts.truth[p] for p in shoot.captures[cid].files if p in ts.truth]
                t = merge_truths(truths) if truths else None
                if t is None or not t.labelled(3):
                    if ts.unlabelled == "exclude":
                        continue
                    label = 0
                else:
                    label = int(t.kept(3))
                xs.append(emb)
                ys.append(label)
                labelled += 1
                kept += label
            summary.append({"shoot": str(folder), "captures": len(shoot.captures), "examples": labelled, "kept": kept,
                            "truth": ts.source})
    finally:
        cache.close()
        registry.close()
    model = train(np.asarray(xs, dtype=np.float32), np.asarray(ys), embedding_revision=scorer.revision)
    return model, summary


def _leave_one_out(projected: np.ndarray, y: np.ndarray) -> dict[int, float]:
    """{example: decision score} with each example held out of its own training (the original routine: no class
    balancing; skipped below 3 examples or when a fold has one class)."""
    from sklearn.linear_model import SGDClassifier

    n = len(y)
    scores: dict[int, float] = {}
    if n < 3:
        return scores
    for i in range(n):
        train = np.arange(n) != i
        if len(set(y[train].tolist())) < 2:
            continue
        clf = SGDClassifier(loss="log_loss", penalty="l2", alpha=ALPHA, max_iter=1000, tol=1e-3, random_state=RANDOM_STATE)
        clf.fit(projected[train], y[train])
        scores[i] = float(clf.decision_function(projected[[i]]).reshape(-1)[0])
    return scores

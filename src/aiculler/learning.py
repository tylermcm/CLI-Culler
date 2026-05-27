from __future__ import annotations

import threading
from typing import Callable

import numpy as np
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import SGDClassifier

from aiculler.storage import SQLiteFeatureStore

ScoresCallback = Callable[[dict[int, float]], None]


class ThreadSafeLearningEngine:
    """Online feedback trainer with non-blocking updates and score callbacks."""

    def __init__(
        self,
        db_connection: SQLiteFeatureStore,
        on_scores_updated_callback: ScoresCallback | None = None,
        projected_dim: int = 64,
    ):
        self.db = db_connection
        self.on_scores_updated = on_scores_updated_callback
        self.projected_dim = int(projected_dim)
        self.projector: IncrementalPCA | None = None
        self.classifier = SGDClassifier(loss="log_loss", penalty="l2", alpha=0.0001)
        self.initialized = False
        self.classes = np.array([0, 1])
        self.lock = threading.RLock()

    def process_user_feedback_async(self, image_id: int, label: int) -> threading.Thread:
        """Spawn a background thread to update the model without blocking callers."""

        thread = threading.Thread(target=self._update_loop, args=(int(image_id), int(label)), daemon=True)
        thread.start()
        return thread

    def process_user_feedback(self, image_id: int, label: int) -> dict[int, float]:
        """Synchronous version for tests, scripts, and deterministic integrations."""

        return self._update_loop(int(image_id), int(label))

    def _update_loop(self, image_id: int, label: int) -> dict[int, float]:
        with self.lock:
            self.db.add_feedback(image_id, label)
            all_ids = self.db.get_all_embedding_ids()
            all_embeddings = self.db.get_all_embeddings()
            if all_embeddings.size == 0:
                return {}

            if not self.initialized:
                n_components = min(self.projected_dim, all_embeddings.shape[0], all_embeddings.shape[1])
                self.projector = IncrementalPCA(n_components=n_components)
                self.projector.partial_fit(all_embeddings)
                self.initialized = True

            if self.projector is None:
                raise RuntimeError("Projector initialization failed")

            raw_embedding = self.db.get_embedding(image_id)
            projected_vector = self.projector.transform(np.atleast_2d(raw_embedding))
            self.classifier.partial_fit(projected_vector, np.array([label]), classes=self.classes)

            projected_all = self.projector.transform(all_embeddings)
            raw_scores = self.classifier.decision_function(projected_all)
            scores = {image_id: float(score) for image_id, score in zip(all_ids, raw_scores)}
            self.db.update_scores(scores)

        if self.on_scores_updated is not None:
            self.on_scores_updated(scores)
        return scores


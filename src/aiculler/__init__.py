"""Headless offline image culling engine."""

__version__ = "0.2.0.dev1"

from aiculler.api import CullResult, CullSettings, FileDecision, GateSettings, cull  # noqa: E402

from aiculler.adapter import RankingAwareAdapter
from aiculler.composite_ranking import CompositeRanker
from aiculler.features import HeadlessFeatureExtractor, IngestionEngine
from aiculler.learning import ThreadSafeLearningEngine
from aiculler.preference_learning import PreferenceLearningScorer
from aiculler.profile_scoring import ProfileScorer
from aiculler.ranking import ActiveQuicksortCuller, GlobalRanker
from aiculler.storage import SQLiteFeatureStore
from aiculler.technical_tags import TechnicalTagScorer
from aiculler.text_scoring import CLIPTextEncoder, TextConditionedScorer

__all__ = [
    "CullResult",
    "CullSettings",
    "FileDecision",
    "GateSettings",
    "cull",
    "ActiveQuicksortCuller",
    "CompositeRanker",
    "GlobalRanker",
    "HeadlessFeatureExtractor",
    "IngestionEngine",
    "RankingAwareAdapter",
    "SQLiteFeatureStore",
    "TechnicalTagScorer",
    "ThreadSafeLearningEngine",
    "CLIPTextEncoder",
    "PreferenceLearningScorer",
    "ProfileScorer",
    "TextConditionedScorer",
]

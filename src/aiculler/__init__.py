"""Headless offline image culling engine for ImageTriage.

``cull(folder, CullSettings(...), models=ModelRegistry.from_app_manifest(app))`` decides keep / reject / review for
every file of a shoot, with reasons; ``outputs.write_decisions`` / ``write_index`` / ``write_xmp`` write the results.
"""

__version__ = "0.3.0.dev1"

from aiculler.api import CullResult, CullSettings, FileDecision, GateSettings, cull  # noqa: E402
from aiculler.models.registry import ModelRegistry, ModelUnavailable  # noqa: E402

__all__ = ["CullResult", "CullSettings", "FileDecision", "GateSettings", "ModelRegistry", "ModelUnavailable", "cull"]

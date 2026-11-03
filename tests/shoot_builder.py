"""Build bracketeering-shaped Shoots by hand, without images, for routing and decision tests."""

from __future__ import annotations

from aiculler.ingest import Capture, ExposureStats, FileInfo, GroupInfo, Shoot


class ShootBuilder:
    def __init__(self) -> None:
        self.groups: dict[str, GroupInfo] = {}
        self.captures: dict[str, Capture] = {}
        self.files: dict[str, FileInfo] = {}
        self.scores: dict[str, dict[str, float]] = {}
        self._n = 0

    def group(
        self,
        kind: str,
        frames: list[dict],
        *,
        scene: int | None = None,
        confidence: float = 1.0,
        warnings: list[str] | None = None,
        representative: int | None = None,
        best_exposed: int | None = None,
    ) -> list[str]:
        """frames: dicts with name, and optional sharpness, exposure, mean, hl, sh, extra_files, error, featureless."""
        self._n += 1
        prefix = {"exposure": "B", "burst": "U", "focus": "F", "timelapse": "T"}.get(kind, "S")
        gid = f"{prefix}{self._n:05d}"
        ids = []
        for pos, fr in enumerate(frames):
            cid = f"cam|sc:{fr['name']}"
            primary = f"/shoot/{fr['name']}.NEF"
            extra = [f"/shoot/{x}" for x in fr.get("extra_files", [])]
            rep = representative if representative is not None else (len(frames) // 2 if kind != "single" else 0)
            role = "single" if kind == "single" else ("representative" if pos == rep else "member")
            self.captures[cid] = Capture(
                id=cid,
                primary=primary,
                files=[primary, *extra],
                group=gid,
                position=pos + 1,
                role=role,
                exposure=ExposureStats(fr.get("hl", 0.0), fr.get("sh", 0.0), fr.get("mean", 0.45), "ok"),
                featureless=fr.get("featureless", False),
                error=fr.get("error"),
            )
            self.files[primary] = FileInfo(primary, cid, True, fr.get("error"))
            for x in extra:
                self.files[x] = FileInfo(x, cid, False)
            values = {}
            if "sharpness" in fr:
                values["sharpness"] = fr["sharpness"]
            values["exposure"] = fr.get("exposure", 1.0)
            for key in ("clip_region", "top_clip", "luma_mean", "luma_spread"):
                if key in fr:
                    values[key] = fr[key]
            self.scores[cid] = values
            ids.append(cid)
        rep = representative if representative is not None else (len(frames) // 2 if kind != "single" else 0)
        self.groups[gid] = GroupInfo(
            id=gid,
            kind=kind,
            confidence=confidence,
            warnings=list(warnings or []),
            captures=ids,
            representative=ids[rep],
            best_exposed=ids[best_exposed] if best_exposed is not None else None,
            scene=scene,
        )
        return ids

    def single(self, name: str, *, scene: int | None = None, **values) -> str:
        return self.group("single", [{"name": name, **values}], scene=scene)[0]

    def build(self) -> Shoot:
        return Shoot(self.groups, self.captures, self.files, "test", {"files": {}})


def run(builder: ShootBuilder, settings=None):
    """Route, gate and decide; returns {primary file name without extension: FileDecision} plus all files."""
    from aiculler.api import CullSettings
    from aiculler.decide import decide
    from aiculler.gates import apply_gates
    from aiculler.routing import route

    settings = settings or CullSettings()
    shoot = builder.build()
    routing = route(shoot, settings)
    gates = apply_gates(shoot, routing, settings, builder.scores)
    files = decide(shoot, routing, gates, builder.scores, settings)
    by_name = {path.rsplit("/", 1)[-1].rsplit(".", 1)[0]: d for path, d in files.items() if d.primary}
    return by_name, files

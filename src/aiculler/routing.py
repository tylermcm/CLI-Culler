"""Routing: which frames compete with which, per group kind and the user's settings.

Two levels of comparison set:

* **Burst sets**: the frames of one burst compete with each other; the top ``bursts_keep`` survive and
  the burst's best frame represents it in its scene.
* **Scene sets**: the *units* of one near-duplicate scene compete: single frames, bracket units (one
  unit per bracket, represented by ``best_exposed`` else ``representative``), burst winners, and focus
  stacks / timelapses when those are set to ``cull``. The top ``scene_keep`` units survive.

A bracket's own frames never compete with each other, with one exception: a bracket bracketeering says does not
register (subject or camera moving) is alternatives of a moment, not an HDR set, and with ``moving_brackets:
as_burst`` its frames compete like a burst's (labelled "moving bracket"). Groups left out of culling (``brackets=ignore``,
``bursts_keep=0``, focus stacks / timelapses on ``keep_all``) are passed through: kept, not ranked.
Nothing is compared across scenes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from aiculler.ingest import Shoot

if TYPE_CHECKING:
    from aiculler.api import CullSettings


@dataclass
class Unit:
    """Something that competes in a scene; its outcome applies to all its captures."""

    id: str  # the group id
    kind: str  # group kind
    captures: list[str]  # captures whose fate follows this unit (capture order)
    rep: str | None  # capture scored for the unit; bursts: filled in by decide() from the burst ranking
    rep_reason: str = ""
    label: str = ""  # how reasons name it ("burst", "moving bracket"); empty: from the kind


@dataclass
class ComparisonSet:
    id: str  # burst group id, or "scene:<n>" / "solo:<group id>" for scenes
    kind: str  # "burst" or "scene"
    members: list[str]  # captures (burst) or unit ids (scene), in capture order
    keep: int | None  # how many survive; None = not reduced (all kept)
    scene: int | None = None
    label: str = "burst"  # how reasons name a burst set: "burst" or "moving bracket"


@dataclass
class Routing:
    passthrough: dict[str, str] = field(default_factory=dict)  # capture -> reason it was kept unranked
    unreadable: dict[str, str] = field(default_factory=dict)  # capture -> error (-> review)
    bursts: list[ComparisonSet] = field(default_factory=list)
    units: dict[str, Unit] = field(default_factory=dict)
    scenes: list[ComparisonSet] = field(default_factory=list)
    pick_one: bool = False

    def scored_captures(self) -> list[str]:
        """Captures that need scores: burst frames and unit representatives. Never passthrough frames."""
        out: list[str] = []
        for b in self.bursts:
            out.extend(b.members)
        for u in self.units.values():
            if u.rep is not None:
                out.append(u.rep)
        return list(dict.fromkeys(out))


def route(shoot: Shoot, settings: "CullSettings") -> Routing:
    r = Routing(pick_one=settings.brackets == "pick_one")
    scene_members: dict[str, list[str]] = {}
    scene_of: dict[str, int | None] = {}

    for g in shoot.groups.values():
        readable = []
        for cid in g.captures:
            err = shoot.captures[cid].error
            if err:
                r.unreadable[cid] = err
            else:
                readable.append(cid)
        if not readable:
            continue

        unit: Unit | None = None
        if ranks_as_burst(g, settings):
            if settings.brackets == "ignore":
                _passthrough(r, readable, "bracket left out of culling (brackets: ignore)")
                continue
            if settings.bursts_keep == 0:
                _passthrough(r, readable, "moving bracket left out of culling (bursts_keep: 0)")
                continue
            r.bursts.append(ComparisonSet(g.id, "burst", readable, settings.bursts_keep, g.scene, label="moving bracket"))
            unit = Unit(g.id, "burst", readable, None, label="moving bracket")
        elif g.kind == "exposure":
            if settings.brackets == "ignore":
                _passthrough(r, readable, "bracket left out of culling (brackets: ignore)")
                continue
            rep = g.best_exposed if g.best_exposed in readable else (g.representative if g.representative in readable else readable[0])
            why = "best exposed" if rep == g.best_exposed else "representative (metered) frame"
            unit = Unit(g.id, g.kind, readable, rep, why)
        elif g.kind == "burst":
            if settings.bursts_keep == 0:
                _passthrough(r, readable, "burst left out of culling (bursts_keep: 0)")
                continue
            r.bursts.append(ComparisonSet(g.id, "burst", readable, settings.bursts_keep, g.scene))
            unit = Unit(g.id, g.kind, readable, None)
        elif g.kind in ("focus", "timelapse"):
            mode = settings.focus_stacks if g.kind == "focus" else settings.timelapses
            label = "focus stack" if g.kind == "focus" else "timelapse"
            if mode == "keep_all":
                setting = "focus_stacks" if g.kind == "focus" else "timelapses"
                _passthrough(r, readable, f"{label}: every frame belongs to the sequence ({setting}: keep_all)")
                continue
            rep = g.representative if g.representative in readable else readable[len(readable) // 2]
            unit = Unit(g.id, g.kind, readable, rep, "representative (middle) frame")
        else:  # single
            unit = Unit(g.id, g.kind, readable, readable[0])

        r.units[unit.id] = unit
        key = f"scene:{g.scene}" if g.scene is not None else f"solo:{g.id}"
        scene_members.setdefault(key, []).append(unit.id)
        scene_of[key] = g.scene

    keep = settings.scene_keep if settings.scene_keep > 0 else None
    for key, members in scene_members.items():
        r.scenes.append(ComparisonSet(key, "scene", members, keep, scene_of[key]))
    return r


MOVING = "frames don't register"


def ranks_as_burst(g, settings: "CullSettings") -> bool:
    """A bracket of a moving subject, culled like a burst (``moving_brackets: as_burst``)."""
    return g.kind == "exposure" and settings.moving_brackets == "as_burst" and any(w.startswith(MOVING) for w in g.warnings)


def _passthrough(r: Routing, captures: list[str], reason: str) -> None:
    for cid in captures:
        r.passthrough[cid] = reason

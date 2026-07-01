"""Shared state objects passed between threads / widgets.

Signal payloads should be dataclasses or primitives — never live PyTorch
tensors or open camera handles.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# 11 classes used by the UI (we drop the SQUAT/squat_extra training artifact).
DISPLAY_CLASSES: tuple[str, ...] = (
    "SQUAT/correct",
    "SQUAT/feet_too_wide",
    "SQUAT/knees_inward",
    "SQUAT/not_low_enough",
    "SQUAT/front_bent",
    "Lunges/correct",
    "Lunges/not_low_enough",
    "Lunges/knee_passes_toe",
    "Plank/correct",
    "Plank/arched_back",
    "Plank/hunch_back",
)

EXERCISES: tuple[str, ...] = ("SQUAT", "Lunges", "Plank")
CORRECT_CLASSES: frozenset[str] = frozenset(
    f"{ex}/correct" for ex in EXERCISES
)


@dataclass(frozen=True)
class Prediction:
    """One inference result, suitable for cross-thread signal payload.

    In gate mode, `label` is the top class within the gate's argmax; in
    per-exercise mode (`selected_mode` set), `label` is the top class
    within the user-selected exercise and the gate is bypassed entirely.

    `gated_exercise` always reflects the exercise the UI should render
    around (= gate argmax in gate mode, = selected_mode otherwise).

    `is_uncertain` is True when the classifier isn't confident enough
    or no real motion is detected. The UI should display a neutral
    state instead of a verdict in that case.
    """
    label: str               # e.g. "SQUAT/knees_inward" (within active exercise)
    confidence: float        # in [0, 1] — confidence of `label`
    probs: np.ndarray        # shape (len(DISPLAY_CLASSES),), float32
    is_correct: bool
    gated_exercise: str      # "SQUAT" | "Lunges" | "Plank"
    gate_probs: np.ndarray   # shape (3,), float32; all-zeros in per-exercise mode
    is_uncertain: bool = False
    selected_mode: str | None = None  # None = gate mode; else the chosen exercise

    @property
    def exercise(self) -> str:
        # The displayed exercise = user pick if in per-exercise mode, else the gate.
        return self.selected_mode or self.gated_exercise

    @property
    def error_name(self) -> str:
        return self.label.split("/", 1)[1]


def labels_for_exercise(ex: str) -> list[str]:
    """Display-order labels for one exercise."""
    return [c for c in DISPLAY_CLASSES if c.startswith(ex + "/")]


def indices_for_exercise(ex: str) -> list[int]:
    """Indices into DISPLAY_CLASSES for one exercise."""
    return [i for i, c in enumerate(DISPLAY_CLASSES) if c.startswith(ex + "/")]

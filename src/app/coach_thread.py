"""Ollama-backed coaching thread for the PyQt app.

Wraps the existing Llama 3.2 3B prompt in a QThread so the GUI never
blocks on `ollama.chat`. Single-flight: a new request replaces any
pending one (we only care about the latest error).
"""
from __future__ import annotations

import queue

import ollama
from PyQt6.QtCore import QThread, pyqtSignal


OLLAMA_MODEL = "llama3.2:3b"
OLLAMA_SYSTEM = (
    "You are a concise strength-training form coach. "
    "Given a single detected form error from a workout pose classifier, "
    "respond with ONE short actionable cue, under 18 words. "
    "Do not greet, do not list multiple cues, do not hedge, do not add quotes."
)


class CoachThread(QThread):
    """Background coaching worker.

    Signals
    -------
    cue_ready(str, str)  : emits (label_the_cue_was_for, cue_text)
    """

    cue_ready = pyqtSignal(str, str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._req_q: "queue.Queue[tuple[str, float]]" = queue.Queue(maxsize=1)
        self._running = True

    def request_cue(self, label: str, confidence: float) -> None:
        """Submit (or replace) a coaching request. Returns immediately."""
        try:
            self._req_q.get_nowait()
        except queue.Empty:
            pass
        try:
            self._req_q.put_nowait((label, confidence))
        except queue.Full:
            pass

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:  # QThread entry point
        while self._running:
            try:
                label, conf = self._req_q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                resp = ollama.chat(
                    model=OLLAMA_MODEL,
                    messages=[
                        {"role": "system", "content": OLLAMA_SYSTEM},
                        {
                            "role": "user",
                            "content": (
                                f"Detected form error: {label} "
                                f"(confidence {conf:.2f})."
                            ),
                        },
                    ],
                    options={"temperature": 0.2, "num_predict": 48},
                )
                cue = resp["message"]["content"].strip()
            except Exception as e:
                cue = f"(coach offline: {type(e).__name__})"
            self.cue_ready.emit(label, cue)

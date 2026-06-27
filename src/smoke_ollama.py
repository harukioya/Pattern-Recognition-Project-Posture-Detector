"""Ollama smoke test: send a structured prompt, time the response, print tok/sec.

Confirms (a) Ollama daemon is running, (b) llama3.2:3b is pulled,
(c) the prompt-shape we plan to use for coaching feedback works.
"""
import json
import time
import ollama

MODEL = "llama3.2:3b"

SYSTEM = (
    "You are a concise strength-training form coach. "
    "Given pose-derived error scores in [0,1] for a single squat rep, "
    "respond with ONE actionable cue under 20 words. "
    "Do not greet, do not list multiple cues, do not hedge."
)

EXAMPLE_PAYLOAD = {
    "exercise": "back_squat",
    "errors": {
        "knees_inward": 0.82,
        "shallow_depth": 0.15,
        "convex_back": 0.10,
        "forward_lean": 0.30,
    },
}


def _call_once(label: str) -> None:
    user_msg = "Errors: " + json.dumps(EXAMPLE_PAYLOAD)
    t0 = time.time()
    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        options={"temperature": 0.2, "num_predict": 60},
    )
    wall = time.time() - t0

    text = response["message"]["content"].strip()
    eval_count = response.get("eval_count", 0)
    eval_duration_ns = response.get("eval_duration", 0)
    load_duration_ns = response.get("load_duration", 0)
    prompt_eval_ns = response.get("prompt_eval_duration", 0)

    eval_s = eval_duration_ns / 1e9
    gen_tok_per_s = eval_count / eval_s if eval_s > 0 else 0.0

    print(f"--- {label} ---")
    print(f"wall            : {wall:.2f} s")
    print(f"load_duration   : {load_duration_ns / 1e9:.2f} s")
    print(f"prompt_eval     : {prompt_eval_ns / 1e9:.2f} s")
    print(f"eval_duration   : {eval_s:.2f} s   ({eval_count} tokens)")
    print(f"gen tok/sec     : {gen_tok_per_s:.1f}   <-- true generation speed")
    print(f"coaching        : {text!r}")
    print()


def main() -> None:
    _call_once("call 1 (cold)")
    _call_once("call 2 (warm)")


if __name__ == "__main__":
    main()

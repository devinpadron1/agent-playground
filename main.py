from __future__ import annotations

import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from typing import List

MODEL = "gemini-3-flash-preview"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"
SYSTEM_PROMPT = "You are a minimal educational agent."


def load_api_key() -> str:
    with open(".env", "r", encoding="utf-8") as env_file:
        for raw_line in env_file:
            key, value = raw_line.strip().split("=", 1)
            key = key.strip()
            value = value.strip().strip("'").strip('"')

            if key and key not in os.environ:
                os.environ[key] = value

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY. Add it to .env.")

    return api_key


def render_history(history: List[str]) -> str:
    return "\n".join(history)


def agent_step(api_key: str, history: List[str]) -> str:
    prompt = render_history(history)
    body = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt,
                    }
                ]
            }
        ]
    }
    request = Request(
        API_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        },
        method="POST",
    )

    try:
        with urlopen(request) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Gemini API HTTP {exc.code}: {error_body}") from exc
    except URLError as exc:
        raise RuntimeError(f"Network error calling Gemini API: {exc.reason}") from exc

    candidates = payload.get("candidates", [])
    if not candidates:
        raise RuntimeError(f"No candidates in Gemini response: {payload}")

    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(part.get("text", "") for part in parts).strip()
    if not text:
        raise RuntimeError(f"Model returned no text: {payload}")

    return text


def run() -> None:
    api_key = load_api_key()
    history: List[str] = [SYSTEM_PROMPT]

    print(f"Minimal agent running with {MODEL}. Type 'exit' to quit.")

    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit"}:
            print("Bye.")
            return

        history.append(f"user: {user_input}")
        assistant_output = agent_step(api_key, history)
        history.append(f"assistant: {assistant_output}")

        print(f"\nAgent: {assistant_output}")


if __name__ == "__main__":
    run()

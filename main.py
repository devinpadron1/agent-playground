from __future__ import annotations

import json
import os
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from typing import List

MODEL = "gemini-3-flash-preview"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"
SYSTEM_PROMPT = "You are a minimal educational agent."

# Get an API
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("Missing GEMINI_API_KEY. Add it to .env.")

# Build history
history: List[str] = [SYSTEM_PROMPT]

print(f"Minimal agent running with {MODEL}. Type 'exit' to quit.")

# Agentic loop
while True:
    user_input = input("\nYou: ").strip()
    if not user_input:
        continue

    history.append(f"user: {user_input}")

    prompt = "\n".join(history)
    body = {"contents": [{"parts": [{"text": prompt}]}]}
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

    candidates = payload.get("candidates", [])
    if not candidates:
        raise RuntimeError(f"No candidates in Gemini response: {payload}")

    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(part.get("text", "") for part in parts).strip()

    history.append(f"assistant: {text}")

    print(f"\nAgent: {text}")


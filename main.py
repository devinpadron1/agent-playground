import json
import os
from urllib.request import Request, urlopen

url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
history = ["You are a minimal educational agent."]

while True:
    history.append("user: " + input("\nYou: "))
    body = {"contents": [{"parts": [{"text": "\n".join(history)}]}]}
    req = Request(
        url,
        json.dumps(body).encode(),
        {"Content-Type": "application/json", "x-goog-api-key": os.environ["GEMINI_API_KEY"]},
    )
    res = json.loads(urlopen(req).read())
    text = res["candidates"][0]["content"]["parts"][0]["text"]
    history.append("assistant: " + text)
    print("\nAgent:", text)

from __future__ import annotations

import json
from dataclasses import dataclass
from urllib import parse, request
import urllib.error


def _http_json(url: str, method: str = "GET", payload: dict | None = None, timeout: int = 30) -> dict:
    headers = {"Accept": "application/json"}
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode("utf-8")

    req = request.Request(url, data=data, headers=headers, method=method)
    try:
        with request.urlopen(req, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "ignore")
        raise RuntimeError(f"{method} {url} failed: HTTP {exc.code} {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"{method} {url} failed: {exc.reason}") from exc


def _http_bytes(url: str, method: str = "GET", payload: dict | None = None, timeout: int = 30) -> bytes:
    headers = {"Accept": "*/*"}
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode("utf-8")

    req = request.Request(url, data=data, headers=headers, method=method)
    try:
        with request.urlopen(req, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "ignore")
        raise RuntimeError(f"{method} {url} failed: HTTP {exc.code} {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"{method} {url} failed: {exc.reason}") from exc


@dataclass(slots=True)
class OllamaClient:
    base_url: str
    model: str
    temperature: float
    max_tokens: int
    timeout_sec: int

    def chat(self, messages: list[dict[str, str]]) -> str:
        url = f"{self.base_url.rstrip('/')}/api/chat"
        payload = {
            "model": self.model,
            "stream": False,
            "messages": messages,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }
        response = _http_json(url, method="POST", payload=payload, timeout=self.timeout_sec)
        message = response.get("message", {})
        text = str(message.get("content", "")).strip()
        if not text:
            raise RuntimeError("Ollama response did not contain message.content")
        return text

    def healthcheck_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/api/tags"


@dataclass(slots=True)
class VoicevoxClient:
    base_url: str
    speaker: int
    speed_scale: float
    timeout_sec: int

    def synthesize(self, text: str) -> bytes:
        base = self.base_url.rstrip("/")
        query_string = parse.urlencode({"text": text, "speaker": self.speaker})
        audio_query_url = f"{base}/audio_query?{query_string}"
        query = _http_json(audio_query_url, method="POST", payload=None, timeout=self.timeout_sec)
        query["speedScale"] = self.speed_scale
        synthesis_url = f"{base}/synthesis?{parse.urlencode({'speaker': self.speaker})}"
        return _http_bytes(synthesis_url, method="POST", payload=query, timeout=self.timeout_sec)

    def healthcheck_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/version"

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol
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


class LlmClient(Protocol):
    def chat(self, messages: list[dict[str, str]]) -> str:
        ...

    def list_models(self) -> list[str]:
        ...

    def warm_up(self) -> None:
        ...

    def healthcheck_url(self) -> str:
        ...


@dataclass(slots=True)
class OllamaClient:
    base_url: str
    model: str
    temperature: float
    max_tokens: int
    timeout_sec: int

    def _chat_request(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
    ) -> dict:
        url = f"{self.base_url.rstrip('/')}/api/chat"
        payload = {
            "model": self.model,
            "stream": False,
            "messages": messages,
            "options": {
                "temperature": self.temperature,
                "num_predict": max_tokens,
            },
        }
        return _http_json(url, method="POST", payload=payload, timeout=self.timeout_sec)

    def chat(self, messages: list[dict[str, str]]) -> str:
        response = self._chat_request(messages, self.max_tokens)
        message = response.get("message", {})
        text = str(message.get("content", "")).strip()
        if not text:
            raise RuntimeError("Ollama response did not contain message.content")
        return text

    def list_models(self) -> list[str]:
        response = _http_json(self.healthcheck_url(), method="GET", timeout=self.timeout_sec)
        models = response.get("models", [])
        names: list[str] = []
        for item in models:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or item.get("model") or "").strip()
            if name and name not in names:
                names.append(name)
        return names

    def warm_up(self) -> None:
        self._chat_request(
            [{"role": "system", "content": "Warm up the model. Do not answer."}],
            max_tokens=1,
        )

    def healthcheck_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/api/tags"


def _normalize_lm_studio_base_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/v1"):
        return normalized
    return normalized + "/v1"


@dataclass(slots=True)
class LmStudioClient:
    base_url: str
    model: str
    temperature: float
    max_tokens: int
    timeout_sec: int

    def _chat_request(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
    ) -> dict:
        url = f"{_normalize_lm_studio_base_url(self.base_url)}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        return _http_json(url, method="POST", payload=payload, timeout=self.timeout_sec)

    def chat(self, messages: list[dict[str, str]]) -> str:
        response = self._chat_request(messages, self.max_tokens)
        choices = response.get("choices", [])
        if not choices:
            raise RuntimeError("LM Studio response did not contain choices")
        message = choices[0].get("message", {})
        text = str(message.get("content", "")).strip()
        if not text:
            raise RuntimeError("LM Studio response did not contain choices[0].message.content")
        return text

    def list_models(self) -> list[str]:
        response = _http_json(self.healthcheck_url(), method="GET", timeout=self.timeout_sec)
        models = response.get("data", [])
        names: list[str] = []
        for item in models:
            if not isinstance(item, dict):
                continue
            name = str(item.get("id") or item.get("model") or "").strip()
            if name and name not in names:
                names.append(name)
        return names

    def warm_up(self) -> None:
        self._chat_request(
            [{"role": "system", "content": "Warm up the model. Do not answer."}],
            max_tokens=1,
        )

    def healthcheck_url(self) -> str:
        return f"{_normalize_lm_studio_base_url(self.base_url)}/models"


@dataclass(slots=True)
class VoicevoxClient:
    base_url: str
    speaker: int
    speed_scale: float
    timeout_sec: int

    def warm_up(self) -> None:
        self.synthesize("warmup")

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


def create_llm_client(
    backend: str,
    base_url: str,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout_sec: int,
) -> LlmClient:
    normalized_backend = backend.casefold()

    if normalized_backend == "ollama":
        return OllamaClient(
            base_url=base_url,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_sec=timeout_sec,
        )

    if normalized_backend == "lm_studio":
        return LmStudioClient(
            base_url=base_url,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_sec=timeout_sec,
        )

    raise RuntimeError(f"Unsupported LLM backend: {backend}")

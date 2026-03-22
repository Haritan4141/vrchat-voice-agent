from __future__ import annotations

from unittest import mock
import unittest

from vrchat_ai_tool.services import LmStudioClient, OllamaClient, create_llm_client


class ServicesTests(unittest.TestCase):
    def test_create_llm_client_supports_ollama_and_lm_studio(self) -> None:
        ollama = create_llm_client(
            backend="ollama",
            base_url="http://127.0.0.1:11434",
            model="gemma3:12b",
            temperature=0.7,
            max_tokens=120,
            timeout_sec=60,
        )
        lm_studio = create_llm_client(
            backend="lm_studio",
            base_url="http://127.0.0.1:1234/v1",
            model="local-model",
            temperature=0.7,
            max_tokens=120,
            timeout_sec=60,
        )

        self.assertIsInstance(ollama, OllamaClient)
        self.assertIsInstance(lm_studio, LmStudioClient)

    def test_lm_studio_chat_uses_openai_compatible_endpoint(self) -> None:
        client = LmStudioClient(
            base_url="http://127.0.0.1:1234",
            model="local-model",
            temperature=0.5,
            max_tokens=64,
            timeout_sec=30,
        )

        with mock.patch("vrchat_ai_tool.services._http_json") as http_json:
            http_json.return_value = {
                "choices": [
                    {
                        "message": {
                            "content": "hello from lm studio",
                        }
                    }
                ]
            }
            text = client.chat([{"role": "user", "content": "hello"}])

        self.assertEqual(text, "hello from lm studio")
        http_json.assert_called_once()
        args, kwargs = http_json.call_args
        self.assertEqual(args[0], "http://127.0.0.1:1234/v1/chat/completions")
        self.assertEqual(kwargs["method"], "POST")
        self.assertEqual(kwargs["payload"]["model"], "local-model")
        self.assertEqual(kwargs["payload"]["max_tokens"], 64)

    def test_lm_studio_healthcheck_uses_models_endpoint(self) -> None:
        client = LmStudioClient(
            base_url="http://127.0.0.1:1234",
            model="local-model",
            temperature=0.5,
            max_tokens=64,
            timeout_sec=30,
        )

        self.assertEqual(client.healthcheck_url(), "http://127.0.0.1:1234/v1/models")

    def test_ollama_list_models_uses_tags_endpoint(self) -> None:
        client = OllamaClient(
            base_url="http://127.0.0.1:11434",
            model="gemma3:12b",
            temperature=0.7,
            max_tokens=120,
            timeout_sec=30,
        )

        with mock.patch("vrchat_ai_tool.services._http_json") as http_json:
            http_json.return_value = {
                "models": [
                    {"name": "gemma3:12b"},
                    {"name": "qwen3:14b"},
                ]
            }
            names = client.list_models()

        self.assertEqual(names, ["gemma3:12b", "qwen3:14b"])
        args, kwargs = http_json.call_args
        self.assertEqual(args[0], "http://127.0.0.1:11434/api/tags")
        self.assertEqual(kwargs["method"], "GET")

    def test_lm_studio_list_models_uses_models_endpoint(self) -> None:
        client = LmStudioClient(
            base_url="http://127.0.0.1:1234/v1",
            model="local-model",
            temperature=0.5,
            max_tokens=64,
            timeout_sec=30,
        )

        with mock.patch("vrchat_ai_tool.services._http_json") as http_json:
            http_json.return_value = {
                "data": [
                    {"id": "google/gemma-3-12b"},
                    {"id": "bartowski/qwen2.5-14b"},
                ]
            }
            names = client.list_models()

        self.assertEqual(names, ["google/gemma-3-12b", "bartowski/qwen2.5-14b"])
        args, kwargs = http_json.call_args
        self.assertEqual(args[0], "http://127.0.0.1:1234/v1/models")
        self.assertEqual(kwargs["method"], "GET")


if __name__ == "__main__":
    unittest.main()

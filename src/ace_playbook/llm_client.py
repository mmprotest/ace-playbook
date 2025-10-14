"""OpenAI-compatible LLM client."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional

try:  # pragma: no cover - optional dependency
    import httpx
except ImportError:  # pragma: no cover
    httpx = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from tenacity import retry, stop_after_attempt, wait_exponential
except ImportError:  # pragma: no cover
    def retry(*args, **kwargs):  # type: ignore
        def decorator(func):
            return func

        return decorator

    def stop_after_attempt(*args, **kwargs):  # type: ignore
        return None

    def wait_exponential(*args, **kwargs):  # type: ignore
        return None

from .config import ACEConfig


class LLMError(RuntimeError):
    """Raised when an LLM request fails."""


class ChatClient:
    """Minimal OpenAI-compatible chat client with retry logic."""

    def __init__(self, config: ACEConfig):
        self.config = config
        if httpx is None:  # pragma: no cover
            self._client = None
        else:
            self._client = httpx.AsyncClient(base_url=config.base_url, timeout=config.request_timeout)

    async def _request(
        self,
        method: str,
        url: str,
        json_payload: Dict[str, Any],
        api_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        if self._client is None:
            raise LLMError("httpx is required for network operations")
        headers = {"Content-Type": "application/json"}
        key = api_key or self.config.api_key()
        if key:
            headers["Authorization"] = f"Bearer {key}"

        response = await self._client.request(method, url, json=json_payload, headers=headers)
        if response.status_code >= 400:
            raise LLMError(f"LLM request failed: {response.status_code} {response.text}")
        return response.json()

    @retry(reraise=True, stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        model: Optional[str] = None,
        tools: Optional[Iterable[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        **gen_params: Any,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": model or self.config.model,
            "messages": messages,
        }
        payload.update(
            {
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "max_tokens": self.config.max_tokens,
            }
        )
        payload.update(gen_params)
        if tools is not None:
            payload["tools"] = list(tools)
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice

        return await self._request("POST", "/chat/completions", payload)

    async def embeddings(
        self,
        texts: List[str],
        *,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> List[List[float]]:
        payload = {"model": model or self.config.embedding_model, "input": texts}
        url = "/embeddings"
        if base_url:
            if httpx is None:
                raise LLMError("httpx is required for network operations")
            async with httpx.AsyncClient(base_url=base_url, timeout=self.config.request_timeout) as client:
                response = await client.post(url, json=payload, headers=self._build_headers(api_key))
                if response.status_code >= 400:
                    raise LLMError(f"Embedding request failed: {response.status_code} {response.text}")
                data = response.json()
        else:
            data = await self._request("POST", url, payload, api_key=api_key)
        return [item["embedding"] for item in data.get("data", [])]

    def _build_headers(self, api_key: Optional[str]) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        key = api_key or self.config.embedding_api_key()
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return headers

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()


class SyncChatClient:
    """Synchronous wrapper around :class:`ChatClient`."""

    def __init__(self, config: ACEConfig):
        self._config = config

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, Any]:
        import asyncio

        async def _run() -> Dict[str, Any]:
            client = ChatClient(self._config)
            try:
                return await client.chat(messages, **kwargs)
            finally:
                await client.aclose()

        return asyncio.run(_run())

    def embeddings(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
        import asyncio

        async def _run() -> List[List[float]]:
            client = ChatClient(self._config)
            try:
                return await client.embeddings(texts, **kwargs)
            finally:
                await client.aclose()

        return asyncio.run(_run())


def dump_messages(messages: List[Dict[str, str]]) -> str:
    """Debug helper for pretty-printing conversation messages."""

    return json.dumps(messages, indent=2)

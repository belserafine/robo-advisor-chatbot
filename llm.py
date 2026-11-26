# ------------------------------------------
# LLM ACCESS (single place that talks to the model provider)
# ------------------------------------------
# Provider: Anthropic Claude API, model Claude Haiku 5.5.
# Both agents call only generate_text() / generate_structured(), so switching provider
# means changing this file only.
#
# Notes for Claude Haiku 5.5:
#   * temperature / top_p / top_k are not accepted (non-default values return a 400), so none are sent.
#   * Adaptive thinking is on by default; `effort` sets how much it thinks. Responses can start with
#     thinking blocks, so text is read by block type, never by position.
#   * Thinking counts toward max_tokens, so caps leave room for it.
from __future__ import annotations

import os
from typing import Optional, Type, TypeVar

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

PROVIDER = "Anthropic"
MODEL = os.environ.get("LLM_MODEL", "claude-haiku-5-5")
EFFORT_EXTRACT = "low"     # structured extraction / checks: simple, high-volume
EFFORT_WRITE = "medium"    # plain-language explanations
SETTINGS = {"provider": PROVIDER, "model": MODEL, "thinking": "adaptive (default)",
            "effort_extract": EFFORT_EXTRACT, "effort_write": EFFORT_WRITE,
            "sampling": "provider defaults (temperature not configurable on this model)"}

T = TypeVar("T", bound=BaseModel)
_client: Optional[anthropic.Anthropic] = None


class LLMRefusal(RuntimeError):
    """The model declined the request (stop_reason == "refusal")."""


def client() -> anthropic.Anthropic:
    """Lazy client: reads ANTHROPIC_API_KEY from the environment / .env on first use."""
    global _client
    if _client is None:
        if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            raise RuntimeError("ANTHROPIC_API_KEY is not set - add ANTHROPIC_API_KEY=... to the .env file")
        _client = anthropic.Anthropic()
    return _client


def generate_text(prompt: str, system: Optional[str] = None, effort: str = EFFORT_WRITE,
                  max_tokens: int = 8000) -> str:
    kwargs = {"system": system} if system else {}
    response = client().messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        output_config={"effort": effort},
        messages=[{"role": "user", "content": prompt}],
        **kwargs,
    )
    if response.stop_reason == "refusal":
        raise LLMRefusal(getattr(response.stop_details, "explanation", None) or "request declined")
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    if not text:
        raise RuntimeError(f"empty response (stop_reason={response.stop_reason})")
    return text


def generate_structured(prompt: str, schema: Type[T], system: Optional[str] = None,
                        effort: str = EFFORT_EXTRACT, max_tokens: int = 4000) -> Optional[T]:
    """Returns a validated `schema` instance, or None if the model declined or ran out of tokens."""
    kwargs = {"system": system} if system else {}
    response = client().messages.parse(
        model=MODEL,
        max_tokens=max_tokens,
        output_config={"effort": effort},
        output_format=schema,
        messages=[{"role": "user", "content": prompt}],
        **kwargs,
    )
    if response.stop_reason in ("refusal", "max_tokens"):
        return None
    return response.parsed_output


if __name__ == "__main__":
    print(SETTINGS)
    print(generate_text("Reply with 'API test successful' and one short sentence about Toronto.", effort="low"))

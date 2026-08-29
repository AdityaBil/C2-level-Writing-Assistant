"""
llm_client.py
=============

Modular, provider-agnostic LLM client for the C2 English Writing Assistant.

Supports:
- Groq (Ultra low-latency Llama-3.3-70B, Llama-3.1-8B)
- OpenAI (GPT-4o-mini, GPT-4o)
- OpenRouter / Custom OpenAI-compatible endpoints (Ollama, Together, DeepSeek, vLLM)

Zero local GPU requirements. Loads credentials securely from .env.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from dotenv import load_dotenv

# Automatically load .env if present
load_dotenv()


def get_config_val(key: str, default: str = "") -> str:
    """Resolve configuration value from environment variables or Streamlit secrets."""
    # 1. Direct environment variable
    val = os.getenv(key)
    if val:
        return val.strip()

    # 2. Check alternative provider-specific env vars for API keys
    if key == "LLM_API_KEY":
        for alt in ["GROQ_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY"]:
            alt_val = os.getenv(alt)
            if alt_val:
                return alt_val.strip()

    # 3. Streamlit Cloud secrets (st.secrets)
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            if key in st.secrets:
                return str(st.secrets[key]).strip()
            if key.lower() in st.secrets:
                return str(st.secrets[key.lower()]).strip()
            if key == "LLM_API_KEY":
                for alt in ["GROQ_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY", "groq_api_key", "openai_api_key"]:
                    if alt in st.secrets:
                        return str(st.secrets[alt]).strip()
    except Exception:
        pass

    return default


# --------------------------------------------------------------------------- #
# Custom Exceptions for Clean User-Facing Error Messages
# --------------------------------------------------------------------------- #

class LLMClientError(Exception):
    """Base error for LLM client failures."""


class MissingAPIKeyError(LLMClientError):
    """Raised when no API key is provided."""


class AuthenticationError(LLMClientError):
    """Raised when the API key is rejected."""


class RateLimitError(LLMClientError):
    """Raised when provider rate limits or quotas are hit."""


class ModelNotFoundError(LLMClientError):
    """Raised when model ID is invalid or unavailable."""


class NetworkTimeoutError(LLMClientError):
    """Raised when API connection times out."""


# --------------------------------------------------------------------------- #
# Provider Presets & Default Models
# --------------------------------------------------------------------------- #

DEFAULT_MODELS: Dict[str, str] = {
    "groq": "qwen/qwen3.8-27b",
    "openai": "gpt-4o-mini",
    "openrouter": "meta-llama/llama-3.3-70b-instruct",
    "custom": "default",
    "demo": "c2-demonstration-engine",
}

DEFAULT_BASE_URLS: Dict[str, Optional[str]] = {
    "groq": "https://api.groq.com/openai/v1",
    "openai": None,
    "openrouter": "https://openrouter.ai/api/v1",
    "custom": None,
    "demo": None,
}


DEMO_RESPONSES = {
    "enhance": (
        "### C2 VERSION\n"
        "This constitutes an exceptionally auspicious opportunity to consolidate our strategic objectives.\n\n"
        "### VOCABULARY & SYNTAX NOTE\n"
        "- *Constitutes*: Formal copula replacement providing syntactic authority.\n"
        "- *Auspicious*: Elevates 'ideal' by conveying favorable prospects for future success.\n"
        "- *Consolidate*: Adds analytical precision to the operational intent."
    ),
    "synonyms": (
        "### C2 SYNONYMS\n\n"
        "1. **OPTIMAL**\n"
        "- **Nuance & Register:** Best or most favourable under particular conditions; technical and analytical register.\n"
        "- **Example:** \"Under optimal meteorological conditions, the aircraft completed its transatlantic crossing ahead of schedule.\"\n\n"
        "2. **EXEMPLARY**\n"
        "- **Nuance & Register:** Serving as an outstanding model or benchmark; carries strong commendatory tone.\n"
        "- **Example:** \"Her exemplary dedication to archival scholarship earned universal acclaim from the faculty.\"\n\n"
        "3. **QUINTESSENTIAL**\n"
        "- **Nuance & Register:** Representing the purest, most characteristic embodiment of a type or quality.\n"
        "- **Example:** \"The opening monologue was the quintessential expression of tragic melancholy.\"\n\n"
        "4. **CONSUMMATE**\n"
        "- **Nuance & Register:** Showing the ultimate degree of skill, craftsmanship, or perfection.\n"
        "- **Example:** \"He handled the diplomatic crisis with consummate tact and intellectual composure.\"\n\n"
        "5. **AUSPICIOUS**\n"
        "- **Nuance & Register:** Suggesting favourable circumstances or high likelihood of future success.\n"
        "- **Example:** \"The signing of the accord marked an auspicious commencement to bilateral relations.\"\n\n"
        "### CONTEXTUAL INTERCHANGEABILITY NOTE\n"
        "While 'optimal' pertains strictly to functional efficiency or empirical conditions, 'exemplary' carries a normative or moral standard. 'Quintessential' describes typological purity rather than practical suitability. These terms cannot be substituted interchangeably without altering the core semantic dimension."
    ),
    "polish": (
        "### POLISHED VERSION\n"
        "Although I fundamentally disagree with this philosophy, engaging with it nevertheless taught me the discipline of embracing boredom.\n\n"
        "### REFINEMENT HIGHLIGHTS\n"
        "- Substituted 'Even though' with 'Although' for tighter syntactic cohesion.\n"
        "- Replaced 'studying it' with 'engaging with it' to heighten intellectual depth.\n"
        "- Maintained the original conversational honesty while elevating cadence and flow."
    ),
}


@dataclass
class GenerationResult:
    """Encapsulates response text, metadata, and latency metrics."""
    text: str
    provider: str
    model: str
    latency_seconds: float
    usage: Optional[Dict[str, int]] = None


# --------------------------------------------------------------------------- #
# LLMClient Class
# --------------------------------------------------------------------------- #

class LLMClient:
    """Provider-agnostic client handling API calls with robust error translation."""

    def __init__(
        self,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
    ):
        self.provider = (provider or get_config_val("LLM_PROVIDER", "groq")).lower().strip()
        self.api_key = api_key or get_config_val("LLM_API_KEY", "").strip()
        self.model = model or get_config_val("LLM_MODEL", DEFAULT_MODELS.get(self.provider, "llama-3.3-70b-versatile")).strip()
        self.base_url = base_url or get_config_val("LLM_BASE_URL", DEFAULT_BASE_URLS.get(self.provider) or "")
        self.timeout = timeout

        # Clean empty strings
        if self.base_url == "":
            self.base_url = None

    def _get_openai_compatible_client(self, api_key: str, base_url: Optional[str] = None):
        """Instantiate OpenAI client configured with appropriate credentials and endpoint."""
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMClientError("The 'openai' library is required. Run 'pip install openai'.") from exc

        return OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=self.timeout,
            max_retries=2,
        )

    def _get_groq_client(self, api_key: str):
        """Instantiate official Groq client if available, else fallback to OpenAI-compatible base_url."""
        try:
            from groq import Groq
            return Groq(api_key=api_key, timeout=self.timeout, max_retries=2)
        except ImportError:
            # Fallback to OpenAI SDK pointed at Groq endpoint
            return self._get_openai_compatible_client(api_key=api_key, base_url="https://api.groq.com/openai/v1")

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 768,
        top_p: float = 0.9,
    ) -> GenerationResult:
        """Execute chat completion request with latency measurement and error handling."""
        start_time = time.perf_counter()

        if self.provider == "demo":
            time.sleep(0.3)  # Simulate instant sub-second latency
            user_content = messages[-1].get("content", "").lower()
            if "synonym" in user_content or "c2 alternatives" in user_content:
                text = DEMO_RESPONSES["synonyms"]
            elif "polish" in user_content or "authentic personal voice" in user_content:
                text = DEMO_RESPONSES["polish"]
            else:
                text = DEMO_RESPONSES["enhance"]

            return GenerationResult(
                text=text,
                provider="demo",
                model="c2-demo-engine",
                latency_seconds=time.perf_counter() - start_time,
                usage={"prompt_tokens": 50, "completion_tokens": 120, "total_tokens": 170},
            )

        if not self.api_key:
            raise MissingAPIKeyError(
                f"No API key provided for '{self.provider}'. Please set LLM_API_KEY in your .env file "
                "or provide it in the sidebar."
            )

        try:
            if self.provider == "groq":
                client = self._get_groq_client(self.api_key)
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                )
            else:
                client = self._get_openai_compatible_client(self.api_key, self.base_url)
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                )

            latency = time.perf_counter() - start_time
            content = response.choices[0].message.content or ""

            usage = None
            if hasattr(response, "usage") and response.usage:
                usage = {
                    "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(response.usage, "completion_tokens", 0),
                    "total_tokens": getattr(response.usage, "total_tokens", 0),
                }

            return GenerationResult(
                text=content.strip(),
                provider=self.provider,
                model=self.model,
                latency_seconds=latency,
                usage=usage,
            )

        except Exception as exc:
            self._handle_api_exception(exc)

    def _handle_api_exception(self, exc: Exception) -> None:
        """Translate third-party SDK and HTTP errors into clear, actionable exceptions."""
        err_msg = str(exc)
        err_type = type(exc).__name__.lower()

        if "auth" in err_msg.lower() or "invalid_api_key" in err_msg.lower() or "unauthorized" in err_msg.lower() or "401" in err_msg:
            raise AuthenticationError(
                f"Authentication failed for {self.provider.upper()}. Please verify that your API key is correct."
            ) from exc

        if "rate_limit" in err_msg.lower() or "429" in err_msg or "quota" in err_msg.lower():
            raise RateLimitError(
                f"Rate limit or quota exceeded on {self.provider.upper()}. Please wait a moment or check your account tier."
            ) from exc

        if "model_not_found" in err_msg.lower() or "does not exist" in err_msg.lower() or "404" in err_msg:
            raise ModelNotFoundError(
                f"Model '{self.model}' was not found on {self.provider.upper()}. Please verify the model name."
            ) from exc

        if "timeout" in err_type or "timed out" in err_msg.lower():
            raise NetworkTimeoutError(
                f"Request to {self.provider.upper()} timed out after {self.timeout}s. Please check your internet connection."
            ) from exc

        # General error fallback
        raise LLMClientError(f"API Error ({self.provider.upper()}): {err_msg}") from exc

    def test_connection(self) -> Tuple[bool, str]:
        """Send a lightweight test message to verify API credentials and connectivity."""
        if self.provider == "demo":
            return True, "Demo Mode active (Instant offline evaluation)"

        if not self.api_key:
            return False, "Missing API Key"

        try:
            test_messages = [
                {"role": "system", "content": "Respond only with OK."},
                {"role": "user", "content": "Ping"},
            ]
            res = self.generate(test_messages, max_tokens=10, temperature=0.1)
            return True, f"Connected to {self.provider.upper()} ({self.model}) in {res.latency_seconds:.2f}s"
        except Exception as exc:
            return False, str(exc)


# --------------------------------------------------------------------------- #
# High-Level Convenience Function
# --------------------------------------------------------------------------- #

def generate_response(
    messages: List[Dict[str, str]],
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 768,
    top_p: float = 0.9,
) -> GenerationResult:
    """Top-level generation function callable directly without manual client management."""
    client = LLMClient(
        provider=provider,
        api_key=api_key,
        model=model,
        base_url=base_url,
    )
    return client.generate(
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
    )

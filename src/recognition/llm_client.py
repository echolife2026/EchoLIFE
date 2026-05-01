from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
except Exception:
    ChatOpenAI = None
    ChatPromptTemplate = None

from .config import RecognitionConfig
from .usage import UsageTracker

_LLM_CACHE: Any = None


def get_llm(cfg: RecognitionConfig) -> Optional[Any]:
    global _LLM_CACHE
    if _LLM_CACHE is not None:
        return _LLM_CACHE
    if ChatOpenAI is None or ChatPromptTemplate is None:
        return None
    if not cfg.openai_api_key:
        return None
    kwargs = {
        "model": cfg.llm_model,
        "temperature": cfg.llm_temperature,
        "api_key": cfg.openai_api_key,
    }
    if cfg.openai_base_url:
        kwargs["base_url"] = cfg.openai_base_url
    _LLM_CACHE = ChatOpenAI(**kwargs)
    return _LLM_CACHE


def strip_json_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json\n"):
            text = text.replace("json\n", "", 1)
    return text.strip()


def invoke_json(
    llm: Any,
    system_msg: str,
    human_msg: str,
    variables: Dict[str, Any],
    tracker: UsageTracker,
    stage: str,
    day: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict[str, Any]]:
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        ("human", human_msg),
    ])
    raw = (prompt | llm).invoke(variables)
    tracker.add(
        stage=stage,
        day=day,
        usage_metadata=getattr(raw, "usage_metadata", None),
        response_metadata=getattr(raw, "response_metadata", None),
        extra=extra,
    )
    text = raw.content if hasattr(raw, "content") else str(raw)
    cleaned = strip_json_fence(text)
    return cleaned, json.loads(cleaned)

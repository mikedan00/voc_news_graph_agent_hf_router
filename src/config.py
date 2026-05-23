from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL_ID = "google/gemma-4-26B-A4B-it"
DEFAULT_ROUTER_MODEL = "google/gemma-4-26B-A4B-it:deepinfra"
DEFAULT_CANDIDATES = (
    "google/gemma-4-26B-A4B-it:deepinfra,"
    "google/gemma-4-26B-A4B-it:novita,"
    "google/gemma-4-31B-it:deepinfra,"
    "google/gemma-4-31B-it:together,"
    "Qwen/Qwen3.5-9B:together,"
    "Qwen/Qwen2.5-7B-Instruct:together"
)

DEFAULT_CATEGORIES = [
    "제품불만",
    "서비스불만",
    "가격/프로모션",
    "경쟁사/시장",
    "기술/AI",
    "보안/개인정보",
    "정책/규제",
    "투자/주가",
    "기타",
]


@dataclass
class AppConfig:
    hf_token: str = ""
    hf_model_id: str = DEFAULT_MODEL_ID
    hf_router_model: str = DEFAULT_ROUTER_MODEL
    hf_model_candidates: List[str] = field(default_factory=lambda: parse_candidates(DEFAULT_CANDIDATES))
    categories: List[str] = field(default_factory=lambda: DEFAULT_CATEGORIES.copy())
    max_items: int = 80
    request_timeout: int = 30

    neo4j_uri: str = ""
    neo4j_user: str = ""
    neo4j_password: str = ""


def parse_candidates(value: str | List[str] | None) -> List[str]:
    if isinstance(value, list):
        return [v.strip() for v in value if str(v).strip()]
    if not value:
        return parse_candidates(DEFAULT_CANDIDATES)
    return [v.strip() for v in str(value).split(",") if v.strip()]


def load_config(
    hf_token: str | None = None,
    model_candidates: str | List[str] | None = None,
    categories: str | List[str] | None = None,
    max_items: int | None = None,
    neo4j_uri: str | None = None,
    neo4j_user: str | None = None,
    neo4j_password: str | None = None,
) -> AppConfig:
    if isinstance(categories, str):
        category_list = [c.strip() for c in categories.split(",") if c.strip()]
    elif isinstance(categories, list):
        category_list = [str(c).strip() for c in categories if str(c).strip()]
    else:
        category_list = DEFAULT_CATEGORIES.copy()

    return AppConfig(
        hf_token=hf_token or os.getenv("HF_TOKEN", ""),
        hf_model_id=os.getenv("HF_MODEL_ID", DEFAULT_MODEL_ID),
        hf_router_model=os.getenv("HF_ROUTER_MODEL", DEFAULT_ROUTER_MODEL),
        hf_model_candidates=parse_candidates(model_candidates or os.getenv("HF_MODEL_CANDIDATES", DEFAULT_CANDIDATES)),
        categories=category_list or DEFAULT_CATEGORIES.copy(),
        max_items=max_items or int(os.getenv("MAX_ITEMS", "80")),
        neo4j_uri=neo4j_uri or os.getenv("NEO4J_URI", ""),
        neo4j_user=neo4j_user or os.getenv("NEO4J_USER", ""),
        neo4j_password=neo4j_password or os.getenv("NEO4J_PASSWORD", ""),
    )

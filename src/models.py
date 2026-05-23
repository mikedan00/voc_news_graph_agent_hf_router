from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Any
import hashlib
import re


def stable_id(*parts: str, prefix: str = "item") -> str:
    raw = "|".join([p or "" for p in parts])
    h = hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()[:12]
    return f"{prefix}_{h}"


@dataclass
class TextItem:
    id: str
    source: str
    title: str
    text: str
    url: str = ""
    published: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ClassifiedItem:
    id: str
    source: str
    title: str
    text: str
    url: str = ""
    published: str = ""
    category: str = "기타"
    sentiment: str = "neutral"
    priority: int = 3
    keywords: List[str] = None
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["keywords"] = self.keywords or []
        return d


def clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", str(text or ""))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def truncate(text: str, n: int = 700) -> str:
    text = clean_text(text)
    return text if len(text) <= n else text[:n] + "..."

from __future__ import annotations

import re
from typing import Iterable, List, Dict, Any
from urllib.parse import quote_plus

import feedparser
import pandas as pd
import requests
from bs4 import BeautifulSoup

from .models import TextItem, clean_text, stable_id


def google_news_rss_url(query: str, lang: str = "ko", region: str = "KR") -> str:
    q = quote_plus(query.strip())
    return f"https://news.google.com/rss/search?q={q}&hl={lang}&gl={region}&ceid={region}:{lang}"


def collect_google_news_queries(queries: Iterable[str], max_per_query: int = 10) -> List[TextItem]:
    urls = [google_news_rss_url(q) for q in queries if q and q.strip()]
    return collect_rss_urls(urls, max_per_source=max_per_query, default_source="Google News")


def collect_rss_urls(urls: Iterable[str], max_per_source: int = 20, default_source: str = "RSS") -> List[TextItem]:
    items: List[TextItem] = []
    for url in [u.strip() for u in urls if u and str(u).strip()]:
        try:
            feed = feedparser.parse(url)
            source_name = getattr(feed.feed, "title", "") or default_source
            for entry in feed.entries[:max_per_source]:
                title = clean_text(getattr(entry, "title", ""))
                summary = clean_text(getattr(entry, "summary", "") or getattr(entry, "description", ""))
                link = getattr(entry, "link", "")
                published = getattr(entry, "published", "") or getattr(entry, "updated", "")
                text = f"{title}\n{summary}".strip()
                if not text:
                    continue
                items.append(
                    TextItem(
                        id=stable_id(source_name, title, link, prefix="news"),
                        source=source_name,
                        title=title or "(no title)",
                        text=text,
                        url=link,
                        published=published,
                    )
                )
        except Exception:
            continue
    return deduplicate_items(items)


def collect_from_web_pages(urls: Iterable[str], max_chars: int = 3500) -> List[TextItem]:
    """
    단순 HTML 페이지 수집. JavaScript 렌더링 사이트는 제한될 수 있습니다.
    """
    headers = {"User-Agent": "Mozilla/5.0 VOCNewsGraphAgent/1.0"}
    items: List[TextItem] = []
    for url in [u.strip() for u in urls if u and str(u).strip()]:
        try:
            r = requests.get(url, headers=headers, timeout=15)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()
            title = clean_text(soup.title.string if soup.title else url)
            text = clean_text(soup.get_text(" "))
            text = text[:max_chars]
            if text:
                items.append(TextItem(id=stable_id(url, title, prefix="web"), source="WEB", title=title, text=text, url=url))
        except Exception:
            continue
    return deduplicate_items(items)


def items_from_free_text(text: str, source: str = "VOC") -> List[TextItem]:
    """
    빈 줄 또는 줄바꿈 단위로 VOC/메모를 아이템화.
    """
    text = text or ""
    chunks = [c.strip() for c in re.split(r"\n\s*\n|[\r\n]+", text) if c.strip()]
    items: List[TextItem] = []
    for i, chunk in enumerate(chunks, 1):
        title = clean_text(chunk[:60])
        items.append(TextItem(id=stable_id(source, str(i), chunk, prefix="voc"), source=source, title=title, text=chunk))
    return items


def safe_cell(value: Any) -> str:
    """CSV/엑셀 셀 값을 안전하게 문자열로 변환합니다. NaN/None/숫자/날짜/혼합 타입 대응."""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return clean_text(str(value))


def combine_row_values(row: pd.Series, max_chars: int = 2500) -> str:
    """
    본문 컬럼을 찾지 못했을 때 행 전체를 안전하게 하나의 텍스트로 합칩니다.
    기존 df.astype(str).agg(" | ".join, axis=1)는 일부 pandas/Python 조합에서 TypeError가 날 수 있어 사용하지 않습니다.
    """
    parts: List[str] = []
    for col, val in row.items():
        s = safe_cell(val)
        if s:
            parts.append(f"{col}: {s}")
    return " | ".join(parts)[:max_chars]


def items_from_dataframe(df: pd.DataFrame, default_source: str = "CSV") -> List[TextItem]:
    if df is None or df.empty:
        return []

    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    lower_cols = {str(c).strip().lower(): c for c in df.columns}
    title_col = first_existing(lower_cols, ["title", "제목", "subject", "headline", "뉴스제목", "게시글제목"])
    text_col = first_existing(lower_cols, ["text", "본문", "content", "내용", "voc", "message", "description", "summary", "댓글", "의견"])
    source_col = first_existing(lower_cols, ["source", "출처", "channel", "site", "매체", "언론사"])
    url_col = first_existing(lower_cols, ["url", "link", "링크", "주소"])
    pub_col = first_existing(lower_cols, ["published", "date", "datetime", "작성일", "날짜", "시간", "등록일"])

    if text_col is None:
        df["_combined_text"] = df.apply(combine_row_values, axis=1)
        text_col = "_combined_text"

    items: List[TextItem] = []
    for idx, row in df.iterrows():
        text = safe_cell(row.get(text_col, ""))
        if not text:
            continue

        title = safe_cell(row.get(title_col, "")) if title_col else ""
        if not title:
            title = clean_text(text[:70]) or f"row-{idx}"

        source = safe_cell(row.get(source_col, default_source)) if source_col else default_source
        url = safe_cell(row.get(url_col, "")) if url_col else ""
        published = safe_cell(row.get(pub_col, "")) if pub_col else ""

        items.append(
            TextItem(
                id=stable_id(source, title, text, url, prefix="csv"),
                source=source or default_source,
                title=title,
                text=text,
                url=url,
                published=published,
            )
        )
    return deduplicate_items(items)

def first_existing(lower_cols: Dict[str, Any], candidates: List[str]):
    for c in candidates:
        if c.lower() in lower_cols:
            return lower_cols[c.lower()]
    return None


def deduplicate_items(items: List[TextItem]) -> List[TextItem]:
    seen = set()
    out = []
    for it in items:
        key = (clean_text(it.title).lower(), clean_text(it.url).lower(), clean_text(it.text)[:120].lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out

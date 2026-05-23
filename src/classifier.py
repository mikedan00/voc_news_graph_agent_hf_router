from __future__ import annotations

import json
import re
from collections import Counter
from typing import List

from .hf_router import HFRouterClient
from .models import ClassifiedItem, TextItem, truncate


KEYWORD_RULES = {
    "제품불만": ["불량", "오류", "버그", "발열", "배터리", "카메라", "느림", "고장", "품질", "업데이트"],
    "서비스불만": ["서비스센터", "상담", "대기", "수리", "as", "고객센터", "응대", "환불", "교환"],
    "가격/프로모션": ["가격", "할인", "프로모션", "쿠폰", "요금", "비싸", "혜택", "구독"],
    "경쟁사/시장": ["애플", "아이폰", "구글", "샤오미", "경쟁", "시장", "점유율", "출하량"],
    "기술/AI": ["ai", "llm", "gemma", "온디바이스", "인공지능", "모델", "에이전트", "기술"],
    "보안/개인정보": ["보안", "개인정보", "해킹", "유출", "권한", "인증", "암호화"],
    "정책/규제": ["정부", "규제", "법", "정책", "과징금", "제재", "가이드라인"],
    "투자/주가": ["주가", "투자", "매출", "영업이익", "실적", "증권", "상장", "전망"],
}


def classify_items_llm(
    items: List[TextItem],
    categories: List[str],
    router: HFRouterClient | None = None,
    batch_size: int = 12,
) -> List[ClassifiedItem]:
    if not items:
        return []
    if router is None:
        return [heuristic_classify(it, categories) for it in items]

    classified: List[ClassifiedItem] = []
    for start in range(0, len(items), batch_size):
        batch = items[start : start + batch_size]
        prompt_items = [
            {
                "id": it.id,
                "source": it.source,
                "title": it.title,
                "text": truncate(it.text, 650),
                "published": it.published,
                "url": it.url,
            }
            for it in batch
        ]
        system = (
            "너는 VOC/뉴스 분석가다. 사용자가 제공한 카테고리 중 하나로 각 아이템을 분류하고, "
            "감성(sentiment: positive/neutral/negative/mixed), 중요도(priority: 1~5, 5가 가장 중요), "
            "핵심 키워드 3~6개, 분류 이유를 JSON 배열로만 반환한다."
        )
        user = {
            "categories": categories,
            "output_schema": [
                {
                    "id": "원본 id",
                    "category": "카테고리 하나",
                    "sentiment": "positive|neutral|negative|mixed",
                    "priority": 1,
                    "keywords": ["키워드"],
                    "reason": "짧은 이유",
                }
            ],
            "items": prompt_items,
        }
        try:
            res = router.json_chat(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
                ],
                temperature=0.1,
                max_tokens=2500,
            )
            arr = res.get("json")
            if isinstance(arr, dict) and "items" in arr:
                arr = arr["items"]
            parsed_by_id = {str(x.get("id")): x for x in arr if isinstance(x, dict)}
            for it in batch:
                x = parsed_by_id.get(it.id)
                if not x:
                    classified.append(heuristic_classify(it, categories))
                    continue
                classified.append(
                    ClassifiedItem(
                        id=it.id,
                        source=it.source,
                        title=it.title,
                        text=it.text,
                        url=it.url,
                        published=it.published,
                        category=normalize_category(str(x.get("category", "기타")), categories),
                        sentiment=normalize_sentiment(str(x.get("sentiment", "neutral"))),
                        priority=safe_priority(x.get("priority", 3)),
                        keywords=clean_keywords(x.get("keywords", []), it.text),
                        reason=str(x.get("reason", ""))[:200],
                    )
                )
        except Exception:
            classified.extend([heuristic_classify(it, categories) for it in batch])
    return classified


def heuristic_classify(item: TextItem, categories: List[str]) -> ClassifiedItem:
    text = f"{item.title} {item.text}".lower()
    scores = Counter()
    for cat, kws in KEYWORD_RULES.items():
        for kw in kws:
            if kw.lower() in text:
                scores[cat] += 1
    category = scores.most_common(1)[0][0] if scores else "기타"
    category = normalize_category(category, categories)

    neg_words = ["불만", "문제", "오류", "고장", "비싸", "느림", "유출", "발열", "부족", "악화"]
    pos_words = ["개선", "성장", "호평", "상승", "출시", "강화", "성공"]
    neg = sum(w in text for w in neg_words)
    pos = sum(w in text for w in pos_words)
    sentiment = "negative" if neg > pos else "positive" if pos > neg else "neutral"
    priority = min(5, max(1, 2 + neg + scores[category] if category in scores else 3))
    keywords = extract_keywords(item.text, top_n=5)
    return ClassifiedItem(
        id=item.id,
        source=item.source,
        title=item.title,
        text=item.text,
        url=item.url,
        published=item.published,
        category=category,
        sentiment=sentiment,
        priority=priority,
        keywords=keywords,
        reason="키워드 규칙 기반 분류",
    )


def extract_keywords(text: str, top_n: int = 6) -> List[str]:
    tokens = re.findall(r"[가-힣A-Za-z0-9][가-힣A-Za-z0-9_\-]{1,}", text or "")
    stop = {
        "그리고", "하지만", "대한", "관련", "최근", "이번", "있는", "없는", "합니다", "입니다",
        "the", "and", "for", "with", "this", "that", "from", "have",
    }
    cnt = Counter(t.strip() for t in tokens if t.lower() not in stop and len(t.strip()) >= 2)
    return [w for w, _ in cnt.most_common(top_n)]


def normalize_category(cat: str, categories: List[str]) -> str:
    if cat in categories:
        return cat
    for c in categories:
        if c in cat or cat in c:
            return c
    return "기타" if "기타" in categories else categories[-1]


def normalize_sentiment(s: str) -> str:
    s = s.lower()
    for v in ["positive", "neutral", "negative", "mixed"]:
        if v in s:
            return v
    return "neutral"


def safe_priority(v) -> int:
    try:
        return max(1, min(5, int(float(v))))
    except Exception:
        return 3


def clean_keywords(v, text: str) -> List[str]:
    if isinstance(v, str):
        arr = re.split(r"[,/|]", v)
    elif isinstance(v, list):
        arr = v
    else:
        arr = []
    out = []
    for x in arr:
        x = str(x).strip()
        if x and x not in out:
            out.append(x[:30])
    return out[:6] or extract_keywords(text, top_n=5)

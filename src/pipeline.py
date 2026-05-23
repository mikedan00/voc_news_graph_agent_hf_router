from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .classifier import classify_items_llm
from .config import AppConfig
from .graphdb import build_graph, graph_to_cypher, graph_to_json, render_pyvis_html
from .hf_router import HFRouterClient
from .models import ClassifiedItem, TextItem, truncate


def make_router(config: AppConfig) -> Optional[HFRouterClient]:
    if not config.hf_token:
        return None
    return HFRouterClient(config.hf_token, config.hf_model_candidates)


def run_analysis(
    items: List[TextItem],
    config: AppConfig,
    use_llm_classification: bool = True,
    use_llm_summary: bool = True,
    output_dir: str | Path = "exports",
) -> Dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    items = items[: config.max_items]
    router = make_router(config) if (config.hf_token and (use_llm_classification or use_llm_summary)) else None

    classified = classify_items_llm(
        items,
        config.categories,
        router=router if use_llm_classification else None,
    )

    g = build_graph(classified)
    html_path = output_dir / "graph.html"
    html = render_pyvis_html(g, html_path)

    df = pd.DataFrame([x.to_dict() for x in classified])
    csv_path = output_dir / "classified_items.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    graph_json = graph_to_json(g)
    graph_json_path = output_dir / "graph.json"
    graph_json_path.write_text(json.dumps(graph_json, ensure_ascii=False, indent=2), encoding="utf-8")

    cypher = graph_to_cypher(classified)
    cypher_path = output_dir / "graph_import.cypher"
    cypher_path.write_text(cypher, encoding="utf-8")

    summary = summarize_items(classified, config, router if use_llm_summary else None)

    summary_path = output_dir / "summary_insights.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "items": classified,
        "df": df,
        "graph": g,
        "graph_html": html,
        "summary": summary,
        "paths": {
            "html": str(html_path),
            "csv": str(csv_path),
            "graph_json": str(graph_json_path),
            "cypher": str(cypher_path),
            "summary": str(summary_path),
        },
    }


def summarize_items(items: List[ClassifiedItem], config: AppConfig, router: Optional[HFRouterClient] = None) -> Dict:
    if not items:
        return {"executive_summary": "분석할 아이템이 없습니다.", "insights": [], "risks": [], "actions": []}

    category_counts = Counter([x.category for x in items])
    sentiment_counts = Counter([x.sentiment for x in items])
    high_priority = sorted(items, key=lambda x: x.priority, reverse=True)[:10]

    base_stats = {
        "total_items": len(items),
        "category_counts": dict(category_counts),
        "sentiment_counts": dict(sentiment_counts),
        "top_priority_items": [
            {
                "title": x.title,
                "category": x.category,
                "sentiment": x.sentiment,
                "priority": x.priority,
                "keywords": x.keywords or [],
                "url": x.url,
                "text": truncate(x.text, 240),
            }
            for x in high_priority
        ],
    }

    if router is None:
        return fallback_summary(base_stats)

    system = (
        "너는 VOC/뉴스 전략 분석 에이전트다. "
        "입력된 분류 결과를 바탕으로 경영진이 바로 읽을 수 있는 요약, 핵심 인사이트, 중요 리스크, 실행 액션을 도출한다. "
        "반드시 JSON 객체만 반환한다."
    )
    user = {
        "task": "VOC/뉴스 요약 및 insight 도출",
        "required_json_schema": {
            "executive_summary": "5~8문장 한국어 요약",
            "key_insights": ["중요 인사이트 5~8개"],
            "critical_issues": ["즉시 확인해야 할 중요사항 3~6개"],
            "recommended_actions": ["실행 액션 5~8개"],
            "category_analysis": [
                {"category": "카테고리", "signal": "관찰 신호", "implication": "사업/제품상 의미", "action": "권장 조치"}
            ],
            "watch_keywords": ["추적 키워드"],
            "confidence_notes": "데이터 한계와 해석상 주의점",
        },
        "stats_and_items": base_stats,
    }
    try:
        res = router.json_chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
            ],
            temperature=0.2,
            max_tokens=2600,
        )
        out = res.get("json")
        if isinstance(out, dict):
            out["_model_used"] = res.get("model")
            out["_base_stats"] = base_stats
            return out
    except Exception as e:
        fb = fallback_summary(base_stats)
        fb["llm_error"] = str(e)
        return fb

    return fallback_summary(base_stats)


def fallback_summary(stats: Dict) -> Dict:
    cats = stats.get("category_counts", {})
    sents = stats.get("sentiment_counts", {})
    top_cat = max(cats.items(), key=lambda x: x[1])[0] if cats else "없음"
    neg = sents.get("negative", 0)
    total = stats.get("total_items", 0)
    return {
        "executive_summary": (
            f"총 {total}건의 VOC/뉴스를 분석했습니다. 가장 많이 등장한 카테고리는 '{top_cat}'입니다. "
            f"부정 감성 아이템은 {neg}건으로 집계되었습니다. 중요도 상위 아이템을 우선 검토하고, 반복 키워드를 기준으로 원인 분석을 진행하는 것이 좋습니다."
        ),
        "key_insights": [
            f"최다 카테고리: {top_cat}",
            "부정 감성과 중요도가 동시에 높은 아이템은 즉시 확인 대상입니다.",
            "키워드 그래프에서 여러 아이템과 연결된 노드는 반복 이슈 또는 시장 신호일 가능성이 큽니다.",
        ],
        "critical_issues": [x["title"] for x in stats.get("top_priority_items", [])[:5]],
        "recommended_actions": [
            "중요도 4~5 아이템을 담당 부서별로 배정합니다.",
            "반복 키워드 기준으로 원인/영향/대응상태를 관리합니다.",
            "동일 이슈가 뉴스와 VOC 양쪽에서 동시에 나타나는지 추적합니다.",
        ],
        "category_analysis": [
            {"category": c, "signal": f"{n}건 감지", "implication": "추가 검토 필요", "action": "상세 원문 확인"}
            for c, n in cats.items()
        ],
        "watch_keywords": [],
        "confidence_notes": "LLM 요약을 사용하지 않았거나 실패하여 규칙 기반 요약을 생성했습니다.",
        "_base_stats": stats,
    }

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.collectors import (
    collect_google_news_queries,
    collect_rss_urls,
    collect_from_web_pages,
    items_from_dataframe,
    items_from_free_text,
)
from src.config import load_config
from src.pipeline import run_analysis


def main():
    parser = argparse.ArgumentParser(description="VOC/News Graph Intelligence Agent - HF Router")
    parser.add_argument("--voc-file", type=str, default="", help="VOC 텍스트 파일(.txt)")
    parser.add_argument("--csv-file", type=str, default="", help="VOC/뉴스 CSV 파일")
    parser.add_argument("--rss", type=str, default="", help="RSS URL 여러 개를 콤마로 구분")
    parser.add_argument("--query", type=str, default="", help="Google News 검색어 여러 개를 콤마로 구분")
    parser.add_argument("--web", type=str, default="", help="일반 웹페이지 URL 여러 개를 콤마로 구분")
    parser.add_argument("--categories", type=str, default="", help="카테고리 콤마 구분")
    parser.add_argument("--max-items", type=int, default=80)
    parser.add_argument("--no-llm-classify", action="store_true")
    parser.add_argument("--no-llm-summary", action="store_true")
    parser.add_argument("--out", type=str, default="exports")
    args = parser.parse_args()

    config = load_config(categories=args.categories or None, max_items=args.max_items)

    items = []

    if args.voc_file:
        text = Path(args.voc_file).read_text(encoding="utf-8")
        items += items_from_free_text(text, source="VOC")

    if args.csv_file:
        df = pd.read_csv(args.csv_file)
        items += items_from_dataframe(df)

    if args.rss:
        items += collect_rss_urls(args.rss.split(","), max_per_source=20)

    if args.query:
        items += collect_google_news_queries(args.query.split(","), max_per_query=10)

    if args.web:
        items += collect_from_web_pages(args.web.split(","))

    if not items:
        print("분석할 데이터가 없습니다. --query, --rss, --csv-file, --voc-file 중 하나를 입력하세요.")
        return

    result = run_analysis(
        items,
        config,
        use_llm_classification=not args.no_llm_classify,
        use_llm_summary=not args.no_llm_summary,
        output_dir=args.out,
    )

    print("\n=== 완료 ===")
    print(f"분석 아이템: {len(result['items'])}건")
    print("생성 파일:")
    for k, v in result["paths"].items():
        print(f"- {k}: {v}")
    print("\n요약:")
    print(result["summary"].get("executive_summary", ""))


if __name__ == "__main__":
    main()

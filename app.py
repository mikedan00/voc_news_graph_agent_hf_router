from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.collectors import (
    collect_google_news_queries,
    collect_rss_urls,
    collect_from_web_pages,
    items_from_dataframe,
    items_from_free_text,
)
from src.config import DEFAULT_CANDIDATES, DEFAULT_CATEGORIES, load_config
from src.graphdb import write_to_neo4j
from src.pipeline import run_analysis


st.set_page_config(
    page_title="VOC/News Graph Intelligence Agent",
    page_icon="🧠",
    layout="wide",
)


def get_secret_or_env(name: str, default: str = "") -> str:
    try:
        if name in st.secrets:
            return st.secrets.get(name, default)
    except Exception:
        pass
    return os.getenv(name, default)


def init_state():
    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None
    if "raw_items_count" not in st.session_state:
        st.session_state.raw_items_count = 0


init_state()

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; max-width: 1420px;}
    .metric-card {
        background: #111827; border: 1px solid #374151; border-radius: 18px;
        padding: 16px; color: #F9FAFB;
    }
    .small-muted {color:#9CA3AF; font-size: 0.90rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🧠 VOC / News Graph Intelligence Agent")
st.caption("HF Router + Gemma/Qwen 후보 모델 + VOC/뉴스 수집 + 카테고리 분류 + GraphDB 시각화 + 요약/인사이트 도출")

with st.sidebar:
    st.header("⚙️ 설정")

    default_token = get_secret_or_env("HF_TOKEN", "")
    hf_token = st.text_input(
        "HF_TOKEN",
        value=default_token,
        type="password",
        help="로컬은 .env, 배포는 Streamlit Secrets 사용 권장. 임시 테스트용으로 직접 입력도 가능합니다.",
    )

    default_candidates = get_secret_or_env("HF_MODEL_CANDIDATES", DEFAULT_CANDIDATES)
    candidates_text = st.text_area(
        "HF Router 모델 후보",
        value=default_candidates,
        height=130,
        help="앞 모델이 실패하면 다음 모델로 자동 fallback 됩니다.",
    )

    categories_text = st.text_area(
        "분류 카테고리",
        value=", ".join(DEFAULT_CATEGORIES),
        height=100,
    )

    max_items = st.slider("최대 분석 아이템 수", min_value=10, max_value=250, value=80, step=10)
    use_llm_classification = st.toggle("LLM 분류 사용", value=True)
    use_llm_summary = st.toggle("LLM 요약/인사이트 사용", value=True)

    st.divider()
    st.subheader("선택: Neo4j GraphDB")
    neo4j_uri = st.text_input("NEO4J_URI", value=get_secret_or_env("NEO4J_URI", ""))
    neo4j_user = st.text_input("NEO4J_USER", value=get_secret_or_env("NEO4J_USER", ""))
    neo4j_password = st.text_input("NEO4J_PASSWORD", value=get_secret_or_env("NEO4J_PASSWORD", ""), type="password")

    st.info("Streamlit Cloud에서는 Settings > Secrets에 HF_TOKEN을 저장하세요.", icon="🔐")

tab_collect, tab_result, tab_graph, tab_summary, tab_export = st.tabs(
    ["① 수집/실행", "② 분류 결과", "③ GraphDB 시각화", "④ 요약·Insight", "⑤ 내보내기/Neo4j"]
)

with tab_collect:
    col_left, col_right = st.columns([1.1, 1])

    with col_left:
        st.subheader("VOC / 직접 입력")
        voc_text = st.text_area(
            "VOC, 사용자 의견, 커뮤니티 글, 내부 메모를 붙여넣으세요. 줄 단위 또는 빈 줄 단위로 아이템화됩니다.",
            height=220,
            placeholder="예: 업데이트 후 배터리 소모가 심해졌다는 불만...\n예: 서비스센터 대기 시간이 너무 길다는 VOC...",
        )

        uploaded = st.file_uploader("CSV 업로드", type=["csv"])
        st.caption("지원 컬럼 예: title/제목, text/내용/content/voc, source/출처, url/link, published/date")

        if uploaded:
            try:
                csv_df = pd.read_csv(uploaded)
                st.dataframe(csv_df.head(20), use_container_width=True)
            except Exception as e:
                st.error(f"CSV 읽기 실패: {e}")
                csv_df = None
        else:
            csv_df = None

    with col_right:
        st.subheader("뉴스 / 웹 수집")
        google_queries = st.text_area(
            "Google News 검색어",
            value="AI 스마트폰, 삼성 갤럭시 VOC, 온디바이스 AI",
            height=90,
            help="각 줄 또는 콤마 단위로 여러 검색어 입력",
        )
        rss_urls = st.text_area(
            "RSS URL",
            height=110,
            placeholder="https://news.google.com/rss/search?q=...\nhttps://example.com/rss",
        )
        web_urls = st.text_area(
            "일반 웹페이지 URL",
            height=90,
            placeholder="https://example.com/article",
            help="단순 HTML 텍스트 수집입니다. JS 렌더링 페이지는 제한될 수 있습니다.",
        )

        st.subheader("실행")
        run_btn = st.button("🚀 수집 → 분류 → 그래프 → 요약 실행", type="primary", use_container_width=True)

    if run_btn:
        if not hf_token and (use_llm_classification or use_llm_summary):
            st.warning("HF_TOKEN이 없습니다. LLM 기능은 실패할 수 있으므로, 토큰을 입력하거나 LLM 옵션을 끄세요.")

        with st.status("데이터 수집 및 분석 중...", expanded=True) as status:
            items = []

            if voc_text.strip():
                st.write("VOC 직접 입력 수집")
                items += items_from_free_text(voc_text, source="VOC")

            if csv_df is not None:
                st.write("CSV 아이템 변환")
                items += items_from_dataframe(csv_df)

            q_list = [x.strip() for x in google_queries.replace("\n", ",").split(",") if x.strip()]
            if q_list:
                st.write(f"Google News RSS 검색: {len(q_list)}개 검색어")
                items += collect_google_news_queries(q_list, max_per_query=10)

            rss_list = [x.strip() for x in rss_urls.replace("\n", ",").split(",") if x.strip()]
            if rss_list:
                st.write(f"RSS 수집: {len(rss_list)}개")
                items += collect_rss_urls(rss_list, max_per_source=15)

            web_list = [x.strip() for x in web_urls.replace("\n", ",").split(",") if x.strip()]
            if web_list:
                st.write(f"웹페이지 수집: {len(web_list)}개")
                items += collect_from_web_pages(web_list)

            st.session_state.raw_items_count = len(items)

            if not items:
                st.error("수집된 데이터가 없습니다.")
                status.update(label="중단됨", state="error")
            else:
                st.write(f"수집 아이템: {len(items)}건")
                config = load_config(
                    hf_token=hf_token,
                    model_candidates=candidates_text,
                    categories=categories_text,
                    max_items=max_items,
                    neo4j_uri=neo4j_uri,
                    neo4j_user=neo4j_user,
                    neo4j_password=neo4j_password,
                )
                result = run_analysis(
                    items,
                    config,
                    use_llm_classification=use_llm_classification,
                    use_llm_summary=use_llm_summary,
                    output_dir="exports",
                )
                st.session_state.analysis_result = result
                status.update(label="분석 완료", state="complete")

        if st.session_state.analysis_result:
            st.success("분석이 완료되었습니다. 상단 탭에서 결과를 확인하세요.")

with tab_result:
    result = st.session_state.analysis_result
    if not result:
        st.info("먼저 ① 수집/실행 탭에서 분석을 실행하세요.")
    else:
        df = result["df"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("수집 아이템", st.session_state.raw_items_count)
        c2.metric("분석 아이템", len(df))
        c3.metric("카테고리 수", df["category"].nunique() if "category" in df else 0)
        c4.metric("고중요도(4~5)", int((df["priority"] >= 4).sum()) if "priority" in df else 0)

        st.subheader("분류 결과 테이블")
        st.dataframe(df, use_container_width=True, height=520)

        st.subheader("카테고리/감성 분포")
        col_a, col_b = st.columns(2)
        with col_a:
            st.bar_chart(df["category"].value_counts())
        with col_b:
            st.bar_chart(df["sentiment"].value_counts())

with tab_graph:
    result = st.session_state.analysis_result
    if not result:
        st.info("먼저 분석을 실행하세요.")
    else:
        st.subheader("GraphDB 시각화")
        st.caption("파란색=카테고리, 초록색=아이템, 분홍색=키워드, 보라색=출처. 노드를 드래그하거나 확대/축소할 수 있습니다.")
        components.html(result["graph_html"], height=800, scrolling=True)

with tab_summary:
    result = st.session_state.analysis_result
    if not result:
        st.info("먼저 분석을 실행하세요.")
    else:
        summary = result["summary"]
        st.subheader("Executive Summary")
        st.write(summary.get("executive_summary", ""))

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("핵심 인사이트")
            for x in summary.get("key_insights", []):
                st.markdown(f"- {x}")

            st.subheader("중요사항")
            for x in summary.get("critical_issues", []):
                st.markdown(f"- {x}")

        with col2:
            st.subheader("권장 액션")
            for x in summary.get("recommended_actions", []):
                st.markdown(f"- {x}")

            st.subheader("추적 키워드")
            kws = summary.get("watch_keywords", [])
            st.write(", ".join(kws) if kws else "없음")

        st.subheader("카테고리별 분석")
        cat_analysis = summary.get("category_analysis", [])
        if cat_analysis:
            st.dataframe(pd.DataFrame(cat_analysis), use_container_width=True)
        else:
            st.info("카테고리별 분석 데이터가 없습니다.")

        with st.expander("원본 JSON 보기"):
            st.json(summary)

with tab_export:
    result = st.session_state.analysis_result
    if not result:
        st.info("먼저 분석을 실행하세요.")
    else:
        paths = result["paths"]
        st.subheader("파일 다운로드")

        def download_file(label, path, mime):
            p = Path(path)
            if p.exists():
                st.download_button(label, p.read_bytes(), file_name=p.name, mime=mime, use_container_width=True)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            download_file("CSV 다운로드", paths["csv"], "text/csv")
        with col2:
            download_file("Graph JSON", paths["graph_json"], "application/json")
        with col3:
            download_file("Neo4j Cypher", paths["cypher"], "text/plain")
        with col4:
            download_file("Summary JSON", paths["summary"], "application/json")

        st.divider()
        st.subheader("Neo4j GraphDB로 저장")
        st.caption("Neo4j Aura 또는 로컬 Neo4j가 있으면 Item-Category-Keyword-Source 관계를 실제 GraphDB에 저장합니다.")
        if st.button("Neo4j에 저장", use_container_width=True):
            ok, msg = write_to_neo4j(
                result["items"],
                neo4j_uri,
                neo4j_user,
                neo4j_password,
            )
            if ok:
                st.success(msg)
            else:
                st.error(msg)

        st.subheader("Cypher 미리보기")
        cypher_text = Path(paths["cypher"]).read_text(encoding="utf-8")
        st.code(cypher_text[:6000], language="cypher")

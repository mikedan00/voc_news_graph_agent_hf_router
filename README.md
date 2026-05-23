# VOC / News Graph Intelligence Agent - HF Router

VOC, 원하는 뉴스, RSS, 웹페이지, CSV를 수집한 뒤 카테고리별로 분류하고, GraphDB 형태로 시각화하며, 최종적으로 요약/Insight/중요사항/권장 액션을 도출하는 에이전트입니다.

기본 모델 설정은 요청값을 반영했습니다.

```env
HF_MODEL_ID=google/gemma-4-26B-A4B-it
HF_ROUTER_MODEL=google/gemma-4-26B-A4B-it:deepinfra
HF_MODEL_CANDIDATES=google/gemma-4-26B-A4B-it:deepinfra,google/gemma-4-26B-A4B-it:novita,google/gemma-4-31B-it:deepinfra,google/gemma-4-31B-it:together,Qwen/Qwen3.5-9B:together,Qwen/Qwen2.5-7B-Instruct:together
```

HF Router 호출은 OpenAI 호환 방식의 `https://router.huggingface.co/v1` 엔드포인트를 사용합니다.

---

## 1. 전체 구조

```text
voc_news_graph_agent_hf_router/
├─ app.py                         # Streamlit UI
├─ cli.py                         # VS Code/터미널 로컬 CLI 실행
├─ requirements.txt
├─ .env.example
├─ .streamlit/
│  └─ secrets.toml.example
├─ data/
│  ├─ sample_voc.csv
│  └─ rss_sources.example.csv
├─ exports/                       # 실행 후 결과 생성 폴더
└─ src/
   ├─ config.py
   ├─ hf_router.py
   ├─ collectors.py
   ├─ classifier.py
   ├─ graphdb.py
   ├─ models.py
   └─ pipeline.py
```

---

## 2. VS Code 로컬 실행

### 2-1. 압축 해제 후 폴더 열기

```powershell
cd C:\0MyWork1
# ZIP 압축 해제 후
cd voc_news_graph_agent_hf_router
code .
```

### 2-2. 가상환경 생성

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

PowerShell 실행 정책 문제 발생 시:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### 2-3. HF_TOKEN 설정

`.env.example` 파일을 `.env`로 복사합니다.

```powershell
copy .env.example .env
notepad .env
```

`.env` 안에 실제 Hugging Face 토큰을 입력합니다.

```env
HF_TOKEN=hf_실제토큰
```

---

## 3. Streamlit 로컬 실행

```powershell
streamlit run app.py
```

브라우저에서 보통 아래 주소가 열립니다.

```text
http://localhost:8501
```

앱 사이드바에서 `HF_TOKEN`이 비어 있으면 직접 입력할 수 있습니다. 로컬에서는 `.env`를 읽도록 되어 있습니다.

---

## 4. CLI 실행 예시

Google News 검색어 기반 실행:

```powershell
python cli.py --query "AI 스마트폰,삼성 갤럭시 VOC,온디바이스 AI"
```

CSV 파일 기반 실행:

```powershell
python cli.py --csv-file data/sample_voc.csv
```

VOC 텍스트 파일 기반 실행:

```powershell
python cli.py --voc-file my_voc.txt
```

RSS 직접 입력:

```powershell
python cli.py --rss "https://news.google.com/rss/search?q=AI&hl=ko&gl=KR&ceid=KR:ko"
```

LLM 분류 없이 규칙 기반으로만 테스트:

```powershell
python cli.py --query "AI 스마트폰" --no-llm-classify --no-llm-summary
```

---

## 5. Streamlit Cloud 배포

### 5-1. GitHub에 업로드

```powershell
git init
git add .
git commit -m "Initial commit: VOC news graph agent HF Router"
git branch -M main
git remote add origin https://github.com/사용자명/voc_news_graph_agent_hf_router.git
git push -u origin main
```

### 5-2. Streamlit Cloud에서 배포

1. Streamlit Cloud 접속
2. New app 선택
3. GitHub repo 선택
4. Main file path: `app.py`
5. Advanced settings 또는 App settings > Secrets에 아래 내용 입력

```toml
HF_TOKEN = "hf_실제토큰"
HF_MODEL_ID = "google/gemma-4-26B-A4B-it"
HF_ROUTER_MODEL = "google/gemma-4-26B-A4B-it:deepinfra"
HF_MODEL_CANDIDATES = "google/gemma-4-26B-A4B-it:deepinfra,google/gemma-4-26B-A4B-it:novita,google/gemma-4-31B-it:deepinfra,google/gemma-4-31B-it:together,Qwen/Qwen3.5-9B:together,Qwen/Qwen2.5-7B-Instruct:together"
```

토큰은 절대 GitHub에 커밋하지 마세요.

---

## 6. GraphDB 구성

이 프로젝트는 두 가지 방식을 제공합니다.

### 방식 A: Streamlit 내부 GraphDB 시각화

- `networkx`로 Item-Category-Keyword-Source 관계 그래프 생성
- `pyvis`로 브라우저에서 시각화
- 별도 서버 없이 Streamlit에서 바로 확인 가능

### 방식 B: Neo4j GraphDB 저장

Neo4j Aura 또는 로컬 Neo4j를 사용하는 경우 사이드바에 아래 값을 입력합니다.

```env
NEO4J_URI=neo4j+s://xxxxx.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=비밀번호
```

Streamlit의 `⑤ 내보내기/Neo4j` 탭에서 `Neo4j에 저장`을 누르면 다음 관계가 저장됩니다.

```text
(Item)-[:IN_CATEGORY]->(Category)
(Item)-[:HAS_KEYWORD]->(Keyword)
(Item)-[:FROM_SOURCE]->(Source)
```

Neo4j Browser에서 Cypher 파일을 직접 실행할 수도 있습니다.

---

## 7. 입력 데이터 CSV 형식

지원 컬럼명은 자동 감지됩니다.

| 의미 | 지원 컬럼명 |
|---|---|
| 제목 | title, 제목, subject, headline |
| 본문 | text, 본문, content, 내용, voc, message, description, summary |
| 출처 | source, 출처, channel, site |
| URL | url, link, 링크 |
| 날짜 | published, date, datetime, 작성일, 날짜 |

샘플:

```csv
source,title,text,url,published
VOC,갤럭시 발열 문의,업데이트 후 영상 촬영 시 발열이 심하다는 의견,,
News,AI 폰 경쟁 심화,온디바이스 AI 기능 경쟁이 심화되고 있음,https://example.com,2026-05-23
```

---

## 8. 결과 파일

분석 후 `exports/` 폴더에 생성됩니다.

```text
exports/
├─ classified_items.csv      # 분류 결과
├─ graph.html                # PyVis 그래프
├─ graph.json                # 노드/엣지 JSON
├─ graph_import.cypher       # Neo4j import용 Cypher
└─ summary_insights.json     # 요약/Insight 결과
```

---

## 9. 운영상 주의점

1. HF Router 후보 모델 중 일부 provider/model 조합은 계정, 지역, provider 활성화, 라이선스 동의 상태에 따라 실패할 수 있습니다.
2. 코드에는 fallback 로직이 있어 첫 번째 모델이 실패하면 다음 후보 모델을 순서대로 시도합니다.
3. Streamlit Cloud에서는 대형 모델을 직접 다운로드하지 않습니다. 오직 HF Router API만 호출합니다.
4. RSS/웹 수집은 공개 페이지와 RSS 기반입니다. 로그인 필요한 커뮤니티, 동적 JavaScript 페이지는 별도 크롤러가 필요합니다.
5. VOC에 개인정보가 포함될 수 있으므로 외부 LLM으로 전송하기 전 마스킹 정책을 추가하는 것을 권장합니다.

---

## 10. 다음 확장 아이디어

- Samsung Members, 네이버 카페, 텔레그램 등 소스별 커넥터 추가
- 개인정보/전화번호/이메일 자동 마스킹
- 매일 자동 수집 스케줄러
- 중요도 급상승 알림: Telegram/Email
- Neo4j Aura 기반 대시보드
- RAG 검색: 과거 VOC와 현재 VOC의 유사 이슈 자동 매칭

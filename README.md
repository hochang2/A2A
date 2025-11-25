📚 A2A Book Recommendation System

LLM + 콘텐츠 기반( SBERT ) + 협업필터링 기반 하이브리드 추천 시스템

이 프로젝트는 사용자의 자연어 입력(기분·상황·취향) 을 분석하여
가장 적합한 책을 추천하는 하이브리드 추천 시스템입니다.

사용자 입력 → LLM 파싱 → 콘텐츠 기반 후보 생성 (SBERT)
                           → CF 후보 집계
                           → LLM Re-rank → 자연어 설명 생성

✨ Features
1) LLM 기반 전략/취향 파싱 (LLM Decider)

사용자 입력 문장에서 다음을 추출:

mood / 감정 (sad, excited…)

desired feeling

genres

content mood

전략(by_mood, by_genre, hybrid)

JSON 형식으로만 응답하도록 강제 → 안정적인 파서 설계

2) SBERT 기반 콘텐츠 기반 추천 (TF-IDF → 업그레이드 완료)

전체 책(GoodBooks-10k)을 SBERT 임베딩

내용/장르/저자 기반 의미적 유사도 계산

초기 계산만 40초 → 이후 캐싱(npy)하면 즉시 불러오기

3) ALS/Implicit 기반 협업 필터링(CF)

사용자의 과거 평점 기반 추천

현재는 cold-start → fallback 구조로 작동

향후 user 로그 학습 시 쉽게 확장 가능

4) LLM Reranker + 자연어 추천 설명

후보 top-N에 대해 LLM이:

사용자 mood에 맞게 재랭킹

자연어 해설문 생성 (고급 UX)

5) LangGraph 기반 파이프라인 설계

Node 구성:

llm_decider_node

generate_candidates_node (SBERT + CF)

rerank_node

natural_output_node

그래프로 pipeline 실행 → 유지보수/확장 용이

6) Sanity-check 스크립트 (debug_sanity.py)

여러 사용자 입력을 자동으로 테스트하여:

LLM 파싱 결과

후보 리스트 top-5

자연어 설명문

을 한눈에 검증할 수 있음.

7) 비한국어(아랍어 등) 제목 필터링

다음 범위(\u0600–\u06FF)의 문자가 포함된 책 제목은 자동 제외.

def is_non_korean_preferred(book):
    for ch in book["title"]:
        if '\u0600' <= ch <= '\u06FF':
            return False
    return True


콘텐츠 기반 + CF 전체 결과에 적용됨.

🗂 프로젝트 구조
a2a/
├── data/
│   ├── books.csv
│   ├── book_embs.npy      # SBERT 캐싱
│
├── src/
│   ├── book/
│   │   ├── recommender.py        # SBERT 기반 content rec
│   │   ├── cf_recommender.py     # Collaborative Filtering
│   │   ├── llm_decider.py        # LLM 취향 파서
│   │   ├── llm_reranker.py       # LLM 재랭커
│   │   ├── graph_book.py         # LangGraph 파이프라인
│   │   ├── debug_sanity.py       # 품질 검증 스크립트
│   │   ├── run_chat_llm_demo.py  # 실사용 demo
│   │
│   ├── common/
│   │   ├── state_types.py
│   │
├── README.md
├── requirements.txt
└── .env

🚀 실행 방법
1) 환경 구성
pip install -r requirements.txt


필수 라이브러리:

sentence-transformers

langgraph

openai

pandas, numpy

implicit (옵션)

2) SBERT 임베딩 생성 + 캐싱
python -m src.book.build_embeddings


생성된 임베딩은 자동으로 data/book_embs.npy에 저장됩니다.

3) demo 실행
python -m src.book.run_chat_llm_demo

4) sanity 테스트
python -m src.book.debug_sanity

📈 향후 개선 로드맵

BM25 기반 장르·설명 sparse vector 추가(Hybrid)

User 로그를 활용한 협업필터링 fully 활성화

LLM output grounding (hallucination 감소)

영화/음악/음식 도메인으로 확장 (A2A Multi-domain)

서버형 FastAPI backend로 전환 → 프론트 연결
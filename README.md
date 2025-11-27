```markdown
# 📚 A2A Book Recommendation System

**LLM + SBERT 기반 콘텐츠 추천 + 협업필터링(ALS/Implicit) + LLM 재랭킹**  
→ 감정 기반 하이브리드 책 추천 시스템

사용자의 **자연어 입력(기분·상황·취향)** 을 분석하여  
가장 적합한 책을 추천하는 고급 추천 엔진입니다.

파이프라인:

> **사용자 입력 → LLM 파싱 → SBERT 콘텐츠 기반 후보 → CF 후보 → LLM Re-rank → 자연어 설명 생성**

---

## ✨ Features

---

## 🔍 1. LLM 기반 전략·취향 파싱 (LLM Decider)

사용자 입력 문장에서 자동으로 다음 정보를 추출합니다:

- `mood / 감정` (예: sad, lonely, happy)
- `desired_feeling` (예: comforted, motivated)
- `genres` / `genres_en`
- `content_mood`
- `strategy` (by_mood / by_genre / hybrid)

LLM이 **항상 JSON으로만 답하도록 강제**하여  
파싱 오류를 최소화했습니다.

---

## 🧠 2. SBERT 기반 콘텐츠 기반 추천 (Content-based)

> **TF-IDF → SBERT 임베딩 기반**으로 업그레이드 완료

- 책의 **설명 / 제목 / 저자 / 장르**를 기반으로 SBERT 임베딩 생성
- 전체 GoodBooks-10k 코퍼스를 **벡터화 후 NPY 캐싱**
- 코사인 유사도로 의미 기반 추천 수행
- 최초 임베딩 생성에는 약 40초  
  → 이후에는 즉시 로딩

---

## 👥 3. ALS / Implicit 기반 협업필터링 (CF)

- 사용자의 과거 평점 기반 추천(ALS)
- cold-start 시 popularity fallback
- 실제 환경에서는  
  - user 로그 누적  
  - implicit feedback 활용  
  - 개인화 추천 강화  
  가능하도록 설계됨

---

## 🏆 4. LLM Reranker + 자연어 설명 생성

SBERT/CF 후보 Top-N에 대해:

### 1) LLM Reranking
- LLM이 `mood / desired_feeling / genres` 맥락에 맞춰  
  기존 hybrid score를 크게 훼손하지 않는 범위에서 재정렬

### 2) 자연어 추천 설명 생성
- “왜 이 책을 추천했는지”를 한국어로 부드럽고 자연스럽게 설명
- 실제 UX에 큰 효과

---

## 🧭 5. LangGraph 기반 파이프라인 구성

추천 시스템 전체를 LangGraph의 노드로 구성:

- `llm_decider_node`
- `generate_candidates_node`
- `rerank_node`
- `natural_output_node`

장점:

- 구조 분리 + 디버깅 편리
- 모듈 추가/확장에 최적화
- Multi-domain 확장 용이

---

## 🧪 6. Sanity-check 스크립트 (`debug_sanity.py`)

샘플 입력 여러 개를 한 번에 테스트 → 결과 종합 출력:

- LLM 파싱(JSON)
- SBERT/CF hybrid 후보
- Top-5 최종 결과
- 자연어 설명

```bash
python -m src.book.debug_sanity
```

---

## 🌍 7. 비한국어(아랍어 등) 제목 자동 필터링

SBERT 후보·CF 후보 모두에 대해  
아랍어/히브리어 범위를 포함한 책을 자동 필터링합니다.

```python
def is_non_korean_preferred(book: dict) -> bool:
    title = book["title"]
    for ch in title:
        if "\u0600" <= ch <= "\u06FF":  # Arabic/Hebrew block
            return False
    return True
```

→ 한국어/영어 독서 사용자에게 노이즈 제거 효과

---

# 🗂 프로젝트 구조

```plaintext
a2a/
├── data/
│   ├── books.csv                           # 책 메타데이터 (GoodBooks-10k)
│   ├── book_embs.npy                       # SBERT 임베딩 캐시
│   ├── ratings.csv
│   ├── to_read.csv                         # 암묵적 피드백 (rating = 1.0)
│   ├── tags.csv
│   ├── book_tags.csv
│   ├── book_genres.json
│   └── my_ratings.csv                      # 현재 CF에서는 미사용
│
├── src/
│   ├── book/
│   │   ├── recommender.py                  # SBERT Content-based 추천
│   │   ├── cf_recommender.py               # ALS/Implicit 기반 CF
│   │   ├── llm_decider.py                  # LLM 감정·전략·장르 파서
│   │   ├── llm_reranker.py                 # LLM 재랭킹 + 설명 생성
│   │   ├── graph_book.py                   # LangGraph 파이프라인 정의
│   │   └── run_chat_llm_demo.py            # 대화형 데모 실행
│   │
│   └── common/
│       └── state_types.py                  # 공통 상태 타입
│
├── requirements.txt
├── README.md
└── .env                                    # API keys (Git 업로드 금지)
```

---

# 🚀 실행 방법

## 1) 패키지 설치
```bash
pip install -r requirements.txt
```

필수 라이브러리:

- sentence-transformers
- langgraph
- openai
- pandas / numpy
- implicit (CF 사용시)

---

## 2) SBERT 임베딩 생성 + 캐싱
```bash
python -m src.book.build_embeddings
```

생성 파일:
```
data/book_embs.npy
```

→ 이후에는 로딩만 수행 (빠름)

---

## 3) 데모 실행
```bash
python -m src.book.run_chat_llm_demo
```

입력 예시:

```
지금 읽고 싶은 책/기분/취향을 자유롭게 적어보세요:
> 심심한데 설레고 싶다
```



# 📈 향후 개선 로드맵

### 🔹 모델링 강화
- BM25 sparse + SBERT dense Hybrid 개선  
- CF fully 활성화(ALS + implicit feedback 응용)  
- 사용자 장기 취향 기반 personalization 강화

### 🔹 LLM 개선
- grounding 강화 → hallucination 감소  
- 결과 정합성 검증 layer 추가

### 🔹 도메인 확장
- 영화 / 음악 / 음식 추천까지 확장  
- A2A Multi-Domain Recommender로 확장

### 🔹 서비스화/MLOps
- FastAPI 기반 Backend  
- Web UI 연동  
- 임베딩 버전 관리 / 캐시 자동화

---
```


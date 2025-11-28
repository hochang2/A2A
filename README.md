# 📚 A2A Book Recommendation System

**LLM + SBERT 콘텐츠 기반 추천 + 협업필터링(ALS/implicit) + LLM 재랭킹**  
→ 감정·상황·취향까지 반영하는 **하이브리드 책 추천 시스템**

사용자의 **자연어 입력(기분·상황·취향)** 을 분석하여  
가장 적합한 책을 추천하는 고급 추천 엔진입니다.

파이프라인 개요:

> **사용자 입력 → LLM 파싱 → SBERT+TF-IDF 콘텐츠 후보 → CF(ALS) 후보 → Hybrid 결합 → LLM Re-rank → 자연어 설명 생성**

---

# 🏗 전체 아키텍처 & 코드 흐름 (my_ratings 제거 버전)

A2A 추천 흐름을 **코드 기준**으로 정리하면 (✔ *my_ratings 관련 로직 제외*):

---

## 1. `run_chat_llm_demo.py` — 엔트리 포인트

### ✔ STEP 0: 초기 추천 (선택적)  
- 사용자에게 user_id 입력
- `get_initial_recommendations(user_id)` 호출  
  → ALS 기반 CF 추천  
  → CF 불가능하면 popularity fallback  
- 사용자가 고른 책을 **my_ratings.csv에 저장하지 않음(제거됨)**  
  → 현재는 단순히 추천만 보여주는 구조

> **나중에 "초기 추천"을 없애고 싶다면**  
> → `run_chat_llm_demo.py`의 "초기 추천" 섹션을 통째로 삭제하면 됨  
> 파이프라인은 그대로 정상 동작합니다.

### ✔ STEP 1: 감정 기반 자연어 추천 루프

- 사용자 입력 받기
- `run_book_recommendation(user_input, user_id)` 호출
- 결과 구성:
  - LLM 파싱(JSON)
  - SBERT/CF hybrid Top-N
  - LLM 재랭킹 결과
  - 자연어 추천 설명 출력

---

## 2. `graph_book.py` — LangGraph 파이프라인

`run_book_recommendation()` 내부에서 LangGraph 구성:

### ✔ BookState
- `user_input`
- `user_id`
- `decision` (LLM 파싱 JSON)
- `candidates` (hybrid 후보)
- `reranked` (LLM 재랭킹 결과)
- `natural_output` (설명문)

➡ my_ratings 관련 필드는 없음.

### ✔ 그래프 노드 구성

1. **`llm_decider_node`**  
   → 감정·장르·전략 JSON 생성

2. **`generate_candidates_node`**  
   - SBERT/TF-IDF 콘텐츠 기반 추천 생성  
   - ALS CF 추천 생성  
   - hybrid score로 결합  
   - `state["candidates"]` 채움

3. **`rerank_node`**  
   - LLM 기반 재랭킹  
   - 감정·장르 맥락 반영  
   - `state["reranked"]` 채움

4. **`natural_output_node`**  
   - LLM이 추천 이유를 자연스럽게 생성  
   - `state["natural_output"]` 저장

---

## 3. `recommender.py` — SBERT + TF-IDF 콘텐츠 기반 추천

- `books.csv`, `book_genres.json`, `tags.csv`, `book_tags.csv` 사용
- `full_text` 구성: 제목 + 작가 + 장르 + 태그 + 설명  
- TF-IDF 학습 후 캐시 저장  
- SBERT 임베딩도 캐시로 저장  
- 이후 실행에서는 로드만 수행 → 매우 빠름

### Hybrid scoring

```python
score = 0.5 * sbert_similarity + 0.5 * tfidf_similarity

# 📚 A2A Book Recommendation System — ALS 적용 버전 (my_ratings 제거)

---

## 4. `cf_recommender.py` — ALS 기반 협업필터링

### ✔ 사용 데이터
- `ratings.csv`
- `to_read.csv` (implicit → rating=1.0으로 자동 변환)

### ✔ 핵심 기능
- ALS 학습 후  
  → **user_factors / item_factors** 캐시로 저장  
- 매 실행 시 캐시 자동 로드 (학습 불필요)
- `recommend_for_user(user_id, top_k)` 제공  
- cold-start → **popularity fallback**

---

### ✔ my_ratings 관련 기능 완전 제거
- 사용자 선택 책을 기록하지 않음  
- “읽은 책 제외하기” 기능 비활성화  
- 완전한 **비상호작용형 협업필터링 구조**

---

## 5. `llm_decider.py` / `llm_reranker.py`

- 자연어 입력 → 감정·전략·장르 JSON 파싱
- SBERT/CF Hybrid 후보 리스트를 LLM이 재랭킹
- 사용자의 현재 감정/목적에 맞춘 **설명문 자동 생성**
- JSON 출력 강제 + grounding 기법으로 안정성 강화

---

## 6. `debug_sanity.py` — 전체 파이프라인 일괄 테스트

- 여러 자연어 입력을 자동 테스트
- 수행 흐름:
  1) **LLM 파싱(JSON)**  
  2) **SBERT/CF Hybrid 후보 생성**  
  3) **LLM 재랭킹**  
  4) **최종 설명문 출력**  

- my_ratings은 사용하지 않음

---

# 🌍 비한국어 제목 자동 필터링

- SBERT·CF 후보 모두에 대해  
  **아랍어·히브리어 유니코드 범위** 감지하여 제외
- 한국어/영어 사용자에게 노이즈 감소

---

# 🗂 프로젝트 구조 (my_ratings 제거 버전)

a2a/
├── data/
│   ├── books.csv
│   ├── ratings.csv
│   ├── to_read.csv
│   ├── tags.csv
│   ├── book_tags.csv
│   ├── book_genres.json
│   ├── book_embs_*.npy
│   ├── tfidf_vectorizer_fulltext_*.joblib
│   ├── tfidf_matrix_fulltext_*.npz
│   └── als_model_f*_r*_it*_a*.npz
│
├── src/
│   ├── book/
│   │   ├── recommender.py
│   │   ├── cf_recommender.py
│   │   ├── llm_decider.py
│   │   ├── llm_reranker.py
│   │   ├── graph_book.py
│   │   ├── run_chat_llm_demo.py
│   │   └── debug_sanity.py
│   │
│   └── common/
│       └── state_types.py
│
├── requirements.txt
└── .env

### ✔ CF는 ALS + implicit feedback 기반으로만 작동  

---

# 🚀 실행 방법

## 1) 패키지 설치
```bash
pip install -r requirements.txt

## 2) SBERT 임베딩 생성
```bash
python -m src.book.build_embeddings

## 2) 데모 실행
```bash
python -m src.book.run_chat_llm_demo


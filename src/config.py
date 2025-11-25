# src/config.py (새로 추가)
BOOK_TFIDF_MAX_FEATURES = 20_000

HYBRID_ALPHA_CONTENT = 0.7  # content vs CF 비율

LLM_MODEL_DECIDER = "gpt-4.1-mini"
LLM_MODEL_RERANKER = "gpt-4.1-mini"

MAX_CANDIDATES_FOR_LLM = 30

# Embedding 관련 설정
EMBEDDING_MODEL_NAME = "text-embedding-3-small"  # OpenAI 임베딩 모델 이름
MAX_EMBEDDING_CANDIDATES = 50  # 한 번에 임베딩으로 점수 계산할 후보 수

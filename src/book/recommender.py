"""
BookRecommender (SBERT 버전)

GoodBooks-10k 책 메타데이터를 기반으로
**Sentence-BERT 임베딩**을 사용한 콘텐츠 기반 추천을 제공하는 모듈.

역할
----
1) books.csv 로딩 및 전처리
2) title + authors + genres + description 을 합친 full_text를 SBERT로 임베딩
3) 자연어 취향(preference_text) 혹은 LLM decider 결과(llm_decision)를 받아
   임베딩 cosine similarity 기반으로 후보 리스트 반환

외부에서 주로 사용하는 메서드
-----------------------------
- recommend_with_preferences(preference_text, mood_keywords, genres, top_k)
- recommend_from_llm_decision(llm_decision, top_k)
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from src.config import BOOK_TFIDF_MAX_FEATURES  # max_features는 지금 안 쓰지만, 호환용으로 유지

# SBERT 모델 이름 (필요하면 config로 뺄 수 있음)
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# 언어 필터: 아랍어(히브리어 포함) 제목은 제외
def is_non_korean_preferred(book) -> bool:
    title = str(book.get("title", ""))

    for ch in title:
        # 아랍어 / 페르시아 / 우르두 등
        if '\u0600' <= ch <= '\u06FF':
            return False
        # 아랍어 확장 영역 (여유로 추가해도 됨)
        if '\u0750' <= ch <= '\u077F':
            return False
    return True

class BookRecommender:
    """
    GoodBooks-10k 기반 **임베딩 기반 콘텐츠 추천 엔진**.

    필수 컬럼
    ---------
    - book_id
    - title
    - authors 또는 author
    - genres
    - description

    내부적으로 full_text = title + authors + genres + description
    에 대해 SBERT 임베딩을 구성하여 cosine similarity 기반으로 추천을 수행한다.
    """

    def __init__(
        self,
        csv_path: Optional[str] = None,
        max_features: Optional[int] = None,  # 예전 TF-IDF 시대의 파라미터(호환용, 지금은 사용 안 함)
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        """
        Parameters
        ----------
        csv_path : str, optional
            책 메타데이터 CSV 경로.
            None이면 프로젝트 루트 기준 data/goodbooks-10k/books.csv 를 기본값으로 사용.
        max_features : int, optional
            (이전 TF-IDF용 파라미터. 지금은 사용하지 않지만 시그니처 호환을 위해 남김.)
        embedding_model_name : str
            SentenceTransformer 모델 이름.
        """
        if csv_path is None:
            # 프로젝트 루트 기준 data/goodbooks-10k/books.csv
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            csv_path = os.path.join(base_dir, "data", "goodbooks-10k", "books.csv")

        self.csv_path = csv_path
        self.embedding_model_name = embedding_model_name

        # 책 메타데이터
        self.df: pd.DataFrame = self._load_and_prepare_df(csv_path)

        # SBERT 모델 로드
        self.model: SentenceTransformer = SentenceTransformer(self.embedding_model_name)

        # 책 full_text 임베딩 (n_books, dim)
        self.embeddings: np.ndarray = self._build_book_embeddings()

    # --------------------------------------------------------
    # 1-1. 데이터 로딩 & 전처리
    # --------------------------------------------------------

    def _load_and_prepare_df(self, csv_path: str) -> pd.DataFrame:
        """
        CSV를 로드하고 필수 컬럼을 정리한 뒤 full_text 컬럼을 생성한다.
        """
        df = pd.read_csv(csv_path)

        # book_id 존재 여부 체크
        if "book_id" not in df.columns:
            raise ValueError("books.csv 에 'book_id' 컬럼이 필요합니다.")

        # title
        if "title" not in df.columns:
            raise ValueError("books.csv 에 'title' 컬럼이 필요합니다.")
        df["title"] = df["title"].fillna("").astype(str)

        # authors / author 처리
        if "authors" in df.columns:
            df["authors"] = df["authors"].fillna("").astype(str)
        elif "author" in df.columns:
            df["authors"] = df["author"].fillna("").astype(str)
        else:
            raise ValueError("books.csv 에 'authors' 또는 'author' 컬럼이 필요합니다.")

        # genres
        if "genres" in df.columns:
            df["genres"] = df["genres"].fillna("").astype(str)
        else:
            df["genres"] = ""

        # description
        if "description" in df.columns:
            df["description"] = df["description"].fillna("").astype(str)
        else:
            df["description"] = ""

        # full_text 생성: 추천에 사용할 통합 텍스트
        df["full_text"] = (
            df["title"].astype(str)
            + " "
            + df["authors"].astype(str)
            + " "
            + df["genres"].astype(str)
            + " "
            + df["description"].astype(str)
        ).str.strip()
        self.book_genre_text: dict[int, str] = {}
        for _, row in df.iterrows():
            bid = int(row["book_id"])
            text = (
                str(row["genres"]) + " " +
                str(row["title"]) + " " +
                str(row["authors"])
            ).lower()
            self.book_genre_text[bid] = text

        return df

    # --------------------------------------------------------
    # 1-2. SBERT 임베딩 생성, 저장
    # --------------------------------------------------------
    def _get_embedding_cache_path(self) -> str:
        """
        SBERT 책 임베딩을 캐싱할 .npy 파일 경로를 생성한다.
        모델 이름에 따라 파일명을 다르게 해서, 모델을 바꾸면 자동으로 새 파일을 쓰게 함.
        """
        # 프로젝트 루트 기준
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        cache_dir = os.path.join(base_dir, "data", "goodbooks-10k")
        os.makedirs(cache_dir, exist_ok=True)

        # 모델 이름에서 / 같은 문자 제거/치환
        model_tag = self.embedding_model_name.replace("/", "__")
        filename = f"book_embs_{model_tag}.npy"

        return os.path.join(cache_dir, filename)

    def _build_book_embeddings(self) -> np.ndarray:
        """
        full_text 에 대해 SBERT 임베딩을 생성한다.
        - 먼저 캐시(.npy)가 있으면 로드 시도
        - 없거나, 행 개수가 안 맞으면 새로 계산 후 저장
        """
        cache_path = self._get_embedding_cache_path()

        # 1) 캐시가 있으면 먼저 로드 시도
        if os.path.exists(cache_path):
            try:
                embs = np.load(cache_path)
                # 책 개수(df 길이)와 임베딩 행 개수가 같으면 그대로 사용
                if embs.shape[0] == len(self.df):
                    return embs.astype(np.float32)
            except Exception:
                # 로드 실패 시에는 그냥 새로 계산
                pass

        # 2) 캐시가 없거나, 사이즈가 안 맞으면 새로 계산
        texts = self.df["full_text"].tolist()
        embeddings = self.model.encode(texts, batch_size=64, show_progress_bar=True)
        embeddings = np.asarray(embeddings, dtype=np.float32)

        # 3) 계산 결과 캐싱
        np.save(cache_path, embeddings)

        return embeddings


    # --------------------------------------------------------
    # 1-3. 내부 유틸
    # --------------------------------------------------------

    def _build_query_text(
        self,
        preference_text: Optional[str],
        mood_keywords: Optional[List[str]] = None,
        genres: Optional[List[str]] = None,
    ) -> str:
        """
        자연어 취향(preference_text) + mood_keywords + genres 를 합쳐
        하나의 쿼리 텍스트로 만든다.
        """
        tokens: List[str] = []

        if preference_text:
            tokens.append(str(preference_text))

        if mood_keywords:
            tokens.extend([str(m) for m in mood_keywords if m])

        if genres:
            tokens.extend([str(g) for g in genres if g])

        query = " ".join(tokens).strip()
        return query

    def _score_by_embedding(
        self,
        query_text: str,
        top_k: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        SBERT 임베딩 기반 cosine similarity 로 상위 top_k 책을 반환한다.

        반환 형식
        --------
        [
            {
                "book_id": int,
                "title": str,
                "authors": str,
                "score": float (0~1 정규화)
            },
            ...
        ]
        """
        if not query_text:
            return []

        # 쿼리 임베딩
        query_vec = self.model.encode([query_text])  # (1, dim)
        # 코사인 유사도 계산
        sims = cosine_similarity(query_vec, self.embeddings)[0]  # (n_books,)

        # 상위 top_k 인덱스
        top_indices = np.argsort(sims)[::-1][:top_k]

        results: List[Dict[str, Any]] = []
        for idx in top_indices:
            row = self.df.iloc[idx]
            results.append(
                {
                    "book_id": int(row["book_id"]),
                    "title": str(row["title"]),
                    "authors": str(row["authors"]),
                    "score": float(sims[idx]),
                }
            )

        # 0~1 정규화
        if results:
            max_score = max(r["score"] for r in results)
            min_score = min(r["score"] for r in results)
            if max_score > min_score:
                for r in results:
                    r["score"] = (r["score"] - min_score) / (max_score - min_score)
            else:
                # 모든 점수가 동일한 경우 → 전부 1.0 처리
                for r in results:
                    r["score"] = 1.0

        return results
    def _filter_and_reorder_by_genre(
        self,
        results: List[Dict[str, Any]],
        required_genres_en: List[str],
        top_k: int,
        hard_filter_top_n: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        SBERT로 뽑은 results에 대해:
        - required_genres_en에 해당하는 장르가 메타데이터에 들어있는 책들을 우선 선택
        - 최소 hard_filter_top_n 권 이상이면, 그 중에서 top_k까지 사용
        - 부족하면: 장르 매칭된 책 + 나머지 결과를 섞어서 top_k까지 채움
        """
        if not results or not required_genres_en:
            return results[:top_k]

        # 모두 소문자로
        required_genres_en = [g.lower() for g in required_genres_en if g]

        genre_matched: List[Dict[str, Any]] = []
        genre_unmatched: List[Dict[str, Any]] = []

        for r in results:
            bid = int(r["book_id"])
            meta_text = self.book_genre_text.get(bid, "")
            meta_text_lower = meta_text.lower()

            # 하나라도 포함되면 genre 매칭으로 봄
            if any(g in meta_text_lower for g in required_genres_en):
                genre_matched.append(r)
            else:
                genre_unmatched.append(r)

        # 1) 장르 매칭된 책이 충분히 많으면, 그 안에서 top_k
        if len(genre_matched) >= hard_filter_top_n:
            return genre_matched[:top_k]

        # 2) 부족하면: 매칭된 책들을 우선 넣고, 나머지는 기존 순서대로 채우기
        merged: List[Dict[str, Any]] = []
        merged.extend(genre_matched)
        for r in genre_unmatched:
            if len(merged) >= top_k:
                break
            merged.append(r)

        return merged[:top_k]

    # ========================================================
    # 2. 외부 인터페이스 (추천 API)
    # ========================================================

    def recommend_with_preferences(
        self,
        preference_text: Optional[str],
        mood_keywords: Optional[List[str]] = None,
        genres: Optional[List[str]] = None,
        top_k: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        자연어로 표현된 취향/상황(preference_text)와 mood_keywords, genres를 받아
        SBERT 임베딩 기반 콘텐츠 추천을 수행한다.

        예시
        ----
        preference_text = "잔잔한 감성의 성장소설"
        mood_keywords = ["calm", "healing"]
        genres = ["young adult", "contemporary"]
        """
        query_text = self._build_query_text(
            preference_text=preference_text,
            mood_keywords=mood_keywords,
            genres=genres,
        )

        return self._score_by_embedding(query_text=query_text, top_k=top_k)

    def recommend_from_llm_decision(
        self,
        llm_decision: Dict[str, Any],
        top_k: int = 50,
        user_input: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if llm_decision is None:
            llm_decision = {}

        # 1) 쿼리 텍스트 재료 모으기 (기존 그대로)
        preference_tokens: List[str] = []

        mentioned_titles = llm_decision.get("mentioned_titles") or []
        preference_tokens.extend([str(t) for t in mentioned_titles if t])

        extra_constraints = llm_decision.get("extra_constraints") or []
        preference_tokens.extend([str(c) for c in extra_constraints if c])

        current_emotion = llm_decision.get("current_emotion") or []
        desired_feeling = llm_decision.get("desired_feeling") or []
        content_mood = llm_decision.get("content_mood") or []
        preference_tokens.extend([str(e) for e in current_emotion if e])
        preference_tokens.extend([str(e) for e in desired_feeling if e])
        preference_tokens.extend([str(m) for m in content_mood if m])

        mood_keywords = llm_decision.get("mood_keywords") or []

        genres_ko = llm_decision.get("genres") or []
        genres_en = llm_decision.get("genres_en") or []
        genres = []
        genres.extend([str(g) for g in genres_ko if g])
        genres.extend([str(g) for g in genres_en if g])

        preference_text = " ".join(preference_tokens).strip()

        # 2) SBERT(또는 TF-IDF)로 1차 후보 뽑기
        #    - top_k 보다 넉넉히 뽑아서 장르 필터를 적용할 여유를 둠
        base_top_k = max(top_k * 3, 50)
        raw_results = self.recommend_with_preferences(
            preference_text=preference_text,
            mood_keywords=mood_keywords,
            genres=genres,
            top_k=base_top_k,
        )

        # 3) genres_en 이 있으면 장르 필터/보정 적용
        required_genres_en = [str(g) for g in genres_en if g]
        final_results = self._filter_and_reorder_by_genre(
            results=raw_results,
            required_genres_en=required_genres_en,
            top_k=top_k,
            hard_filter_top_n=3,  # 최소 이 정도는 같은 장르로 맞추자
        )

        return final_results

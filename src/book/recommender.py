"""
BookRecommender (SBERT 버전, 최종 정리본)

GoodBooks-10k 책 메타데이터 기반으로
Sentence-BERT 임베딩을 이용한 콘텐츠 기반 추천을 수행하는 엔진.

📌 포함된 기능
------------------------------------------
1) books.csv 로딩 & full_text 생성
2) SBERT 임베딩 계산 + 캐싱
3) LLM Decider 결과 기반 추천
4) exclude_book_ids 처리
5) 장르 필터링 후 재정렬
6) 한국어/아랍어 등 특정 언어 필터링 제거

외부에서 사용하는 핵심 메서드
------------------------------------------
- recommend_with_preferences(preference_text, mood_keywords, genres, top_k)
- recommend_from_llm_decision(llm_decision, top_k, user_input, exclude_book_ids)
"""

from __future__ import annotations

import os
import json  # ⬅️ 새로 추가
from typing import Any, Dict, List, Optional, Set
import logging

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from src.config import BOOK_TFIDF_MAX_FEATURES


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# =========================================================
# 언어 필터: 아랍어/히브리어 제거
# =========================================================
def is_non_korean_preferred(book) -> bool:
    title = str(book.get("title", ""))
    for ch in title:
        if '\u0600' <= ch <= '\u06FF':  # Arabic block
            return False
        if '\u0750' <= ch <= '\u077F':  # Arabic supplement
            return False
    return True


# =========================================================
# SBERT 콘텐츠 기반 추천 엔진
# =========================================================
class BookRecommender:
    """
    SBERT 임베딩 기반 콘텐츠 추천 엔진.
    """

    def __init__(
        self,
        csv_path: Optional[str] = None,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
    ) -> None:

        # 기본 books.csv 경로 설정
        if csv_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            csv_path = os.path.join(base_dir, "data", "goodbooks-10k", "books.csv")

        self.csv_path = csv_path
        self.embedding_model_name = embedding_model_name

        # 1) 데이터 로드 + 전처리
        self.df: pd.DataFrame = self._load_and_prepare_df(csv_path)
        
        # 2) TF-IDF 벡터라이저 + 매트릭스
        self.tfidf_vectorizer: TfidfVectorizer
        self.tfidf_matrix: sparse.spmatrix
        self.tfidf_vectorizer, self.tfidf_matrix = self._build_tfidf_matrix()

        # 3) SBERT 모델 로드
        self.model: SentenceTransformer = SentenceTransformer(self.embedding_model_name)

        # 4) 책 임베딩 생성/로드
        self.embeddings: np.ndarray = self._build_book_embeddings()

    # --------------------------------------------------------
    # 1. 데이터 로딩 + 전처리
    # --------------------------------------------------------
    def _load_and_prepare_df(self, csv_path: str) -> pd.DataFrame:
        # 0) books.csv 로드
        df = pd.read_csv(csv_path)

        # 프로젝트 루트 기준 base_dir
        base_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )

        # 1) 필수 컬럼 보정 ----------------------------------------
        if "title" not in df.columns:
            df["title"] = ""
        else:
            df["title"] = df["title"].fillna("").astype(str)

        if "authors" not in df.columns:
            df["authors"] = ""
        else:
            df["authors"] = df["authors"].fillna("").astype(str)

        if "genres" not in df.columns:
            df["genres"] = ""
        else:
            df["genres"] = df["genres"].fillna("").astype(str)

        if "genres_en" not in df.columns:
            df["genres_en"] = ""
        else:
            df["genres_en"] = df["genres_en"].fillna("").astype(str)

        if "description" not in df.columns:
            df["description"] = (
                df["title"].astype(str)
                + " "
                + df["authors"].astype(str)
                + " "
                + df["genres"].astype(str)
            ).str.strip()
        else:
            df["description"] = df["description"].fillna("").astype(str)

        # 2) book_genres.json 로드해서 genres 보강 ------------------
        try:
            genres_json_path = os.path.join(
                base_dir, "data", "goodbooks-10k", "book_genres.json"
            )
            with open(genres_json_path, "r", encoding="utf-8") as f:
                genres_raw = json.load(f)  # {"1": ["fantasy", ...], ...}

            # key를 int(book_id)로 매핑
            genre_map: Dict[int, List[str]] = {
                int(k): (v or []) for k, v in genres_raw.items()
            }

            def _genres_from_json(bid: Any) -> str:
                try:
                    lst = genre_map.get(int(bid), [])
                except Exception:
                    lst = []
                if not lst:
                    return ""
                return " ".join(str(x) for x in lst)

            df["genres_from_json"] = df["book_id"].map(_genres_from_json).fillna("")
        except Exception as e:
            logging.getLogger(__name__).warning(
                "[BookRec] book_genres.json 로드 실패: %s", e
            )
            df["genres_from_json"] = ""

        # 3) book_tags.csv + tags.csv 로 태그 텍스트 만들기 ---------
        try:
            tags_path = os.path.join(base_dir, "data", "goodbooks-10k", "tags.csv")
            book_tags_path = os.path.join(
                base_dir, "data", "goodbooks-10k", "book_tags.csv"
            )

            tags_df = pd.read_csv(tags_path)            # tag_id, tag_name
            book_tags_df = pd.read_csv(book_tags_path)  # goodreads_book_id, tag_id, count

            # tag_id -> tag_name
            tag_name_map: Dict[int, str] = dict(
                zip(tags_df["tag_id"].astype(int), tags_df["tag_name"].astype(str))
            )

            book_tags_df["tag_name"] = book_tags_df["tag_id"].map(tag_name_map)
            book_tags_df = book_tags_df[book_tags_df["tag_name"].notna()]

            def _is_meaningful_tag(name: str) -> bool:
                name = str(name).strip()
                if not name:
                    return False
                # 전부 숫자/기호면 버리기
                if all((not ch.isalpha()) for ch in name):
                    return False
                return True

            book_tags_df = book_tags_df[
                book_tags_df["tag_name"].map(_is_meaningful_tag)
            ]

            # 각 책별 count 기준 상위 N개 태그만 사용
            TOP_N_TAGS = 5

            # goodreads_book_id가 books.csv의 book_id와 같다고 가정
            book_tags_df["goodreads_book_id"] = book_tags_df[
                "goodreads_book_id"
            ].astype(int)

            book_tags_df = book_tags_df.sort_values(
                ["goodreads_book_id", "count"], ascending=[True, False]
            )

            top_tags_df = book_tags_df.groupby("goodreads_book_id").head(TOP_N_TAGS)

            tags_agg = (
                top_tags_df.groupby("goodreads_book_id")["tag_name"]
                .apply(lambda xs: " ".join(str(t) for t in xs))
            )

            # df["book_id"] 기준 매핑 (book_id == goodreads_book_id 가정)
            df["tags_text"] = df["book_id"].astype(int).map(tags_agg).fillna("")
        except Exception as e:
            logging.getLogger(__name__).warning(
                "[BookRec] tags/book_tags 로드 실패: %s", e
            )
            df["tags_text"] = ""

        # 4) genres_text + full_text 구성 ---------------------------
        df["genres_text"] = (
            df["genres"].fillna("") + " " + df["genres_from_json"].fillna("")
        ).str.strip()

        df["full_text"] = (
            df["title"].fillna("")
            + " "
            + df["authors"].fillna("")
            + " "
            + df["genres_text"].fillna("")
            + " "
            + df["tags_text"].fillna("")
            + " "
            + df["description"].fillna("")
        ).str.strip()

        # 5) book_genre_text (장르/태그 기반 boost용) ---------------
        self.book_genre_text: Dict[int, str] = {}
        for _, row in df.iterrows():
            bid = int(row["book_id"])
            meta_text = (
                str(row.get("genres_text", ""))
                + " "
                + str(row.get("tags_text", ""))
                + " "
                + str(row["title"])
                + " "
                + str(row["authors"])
            ).lower()
            self.book_genre_text[bid] = meta_text

        # 6) 언어 필터 적용 -----------------------------------------
        df = df[df.apply(is_non_korean_preferred, axis=1)].reset_index(drop=True)

        # ✅ 반드시 df를 반환해야 self.df가 None이 되지 않습니다.
        return df


    def _build_tfidf_matrix(self) -> tuple[TfidfVectorizer, sparse.spmatrix]:
        """
        full_text 기준 TF-IDF 행렬 생성.
        - 캐시 없이 매 실행 시 다시 학습 (속도 크게 문제될 정도는 아님)
        """
        vectorizer = TfidfVectorizer(
            max_features=BOOK_TFIDF_MAX_FEATURES,
            ngram_range=(1, 2),
            stop_words="english",
        )
        texts = self.df["full_text"].tolist()
        tfidf_matrix = vectorizer.fit_transform(texts)
        return vectorizer, tfidf_matrix


    # --------------------------------------------------------
    # 2. SBERT 임베딩 로드/생성
    # --------------------------------------------------------
    def _get_embedding_cache_path(self) -> str:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        cache_dir = os.path.join(base_dir, "data", "goodbooks-10k")
        os.makedirs(cache_dir, exist_ok=True)

        tag = self.embedding_model_name.replace("/", "__")
        filename = f"book_embs_{tag}.npy"
        return os.path.join(cache_dir, filename)

    def _build_book_embeddings(self) -> np.ndarray:
        cache_path = self._get_embedding_cache_path()

        # 캐시 로드
        if os.path.exists(cache_path):
            try:
                embs = np.load(cache_path)
                if embs.shape[0] == len(self.df):
                    return embs.astype(np.float32)
            except:
                pass  # 실패하면 새로 계산

        # 새로 생성
        texts = self.df["full_text"].tolist()
        embeddings = self.model.encode(texts, batch_size=64, show_progress_bar=True)
        embeddings = np.asarray(embeddings, dtype=np.float32)

        np.save(cache_path, embeddings)
        return embeddings

    # --------------------------------------------------------
    # 3. 내부 유틸
    # --------------------------------------------------------
    def _build_query_text(
        self,
        preference_text: Optional[str],
        mood_keywords: Optional[List[str]],
        genres: Optional[List[str]],
    ) -> str:
        tokens = []
        if preference_text:
            tokens.append(preference_text)

        if mood_keywords:
            tokens.extend(mood_keywords)

        if genres:
            tokens.extend(genres)

        return " ".join(tokens).strip()

    def _score_by_embedding(
        self,
        query_text: str,
        top_k: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        SBERT + TF-IDF 하이브리드 콘텐츠 스코어링.

        - SBERT: self.embeddings (full_text 임베딩)
        - TF-IDF: self.tfidf_vectorizer, self.tfidf_matrix (full_text 기반)
        두 점수를 0.5 : 0.5 로 단순 가중 평균합니다.
        """
        if not query_text:
            return []

        # -----------------------------
        # 1) SBERT similarity
        # -----------------------------
        query_emb = self.model.encode([query_text])
        sims_sbert = cosine_similarity(query_emb, self.embeddings)[0]  # (num_books,)

        # -----------------------------
        # 2) TF-IDF similarity (있으면)
        # -----------------------------
        sims_tfidf = None
        if getattr(self, "tfidf_vectorizer", None) is not None and getattr(self, "tfidf_matrix", None) is not None:
            try:
                q_tfidf = self.tfidf_vectorizer.transform([query_text])
                sims_tfidf = cosine_similarity(q_tfidf, self.tfidf_matrix)[0]  # (num_books,)
            except Exception:
                # 혹시라도 에러 나면 SBERT만 사용
                sims_tfidf = None

        # -----------------------------
        # 3) 두 스코어 합치기
        # -----------------------------
        if sims_tfidf is not None:
            # 간단히 0.5 : 0.5 평균
            sims = 0.5 * sims_sbert + 0.5 * sims_tfidf
        else:
            sims = sims_sbert

        # -----------------------------
        # 4) 상위 top_k 뽑기 + 0~1 정규화
        # -----------------------------
        top_idx = np.argsort(sims)[::-1][:top_k]

        results = []
        for idx in top_idx:
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
            scores = [r["score"] for r in results]
            mx, mn = max(scores), min(scores)
            if mx > mn:
                for r in results:
                    r["score"] = (r["score"] - mn) / (mx - mn)
            else:
                for r in results:
                    r["score"] = 1.0

        return results



    # --------------------------------------------------------
    # 4. 장르 필터링(LLM 장르 우선 적용)
    # --------------------------------------------------------
    def _filter_and_reorder_by_genre(
        self,
        results: List[Dict[str, Any]],
        required_genres_en: List[str],
        top_k: int,
        hard_filter_top_n: int = 3,
    ) -> List[Dict[str, Any]]:

        if not results or not required_genres_en:
            return results[:top_k]

        required_genres_en = [g.lower() for g in required_genres_en]

        matched = []
        unmatched = []

        for r in results:
            bid = int(r["book_id"])
            meta = self.book_genre_text.get(bid, "")
            if any(g in meta for g in required_genres_en):
                matched.append(r)
            else:
                unmatched.append(r)

        # 충분하면 matched만
        if len(matched) >= hard_filter_top_n:
            return matched[:top_k]

        # 부족하면 unmatched 섞기
        out = matched.copy()
        for r in unmatched:
            if len(out) >= top_k:
                break
            out.append(r)

        return out[:top_k]

    # ========================================================
    # 5. 외부 API (핵심)
    # ========================================================
    def recommend_with_preferences(
        self,
        preference_text: Optional[str],
        mood_keywords: Optional[List[str]],
        genres: Optional[List[str]],
        top_k: int = 50,
    ) -> List[Dict[str, Any]]:

        query = self._build_query_text(preference_text, mood_keywords, genres)
        return self._score_by_embedding(query, top_k)

    def recommend_from_llm_decision(
        self,
        llm_decision: Dict[str, Any],
        top_k: int = 50,
        user_input: Optional[str] = None,
        exclude_book_ids: Optional[Set[int]] = None,
    ) -> List[Dict[str, Any]]:

        if llm_decision is None:
            llm_decision = {}

        exclude_book_ids = exclude_book_ids or set()

        # ------------------------------
        # ① LLM 토큰 조합하여 query 만들기
        # ------------------------------
        preference_tokens = []

        preference_tokens.extend(llm_decision.get("mentioned_titles", []) or [])
        preference_tokens.extend(llm_decision.get("extra_constraints", []) or [])
        preference_tokens.extend(llm_decision.get("current_emotion", []) or [])
        preference_tokens.extend(llm_decision.get("desired_feeling", []) or [])
        preference_tokens.extend(llm_decision.get("content_mood", []) or [])

        preference_text = " ".join(preference_tokens).strip()
        mood_keywords = llm_decision.get("mood_keywords") or []

        genres_ko = llm_decision.get("genres") or []
        genres_en = llm_decision.get("genres_en") or []
        genres = genres_ko + genres_en

        # ------------------------------
        # ② SBERT로 넉넉히 후보 뽑기
        # ------------------------------
        base_k = max(top_k * 3, 50)
        raw_results = self.recommend_with_preferences(
            preference_text=preference_text,
            mood_keywords=mood_keywords,
            genres=genres,
            top_k=base_k,
        )

        # ------------------------------
        # ③ exclude_book_ids 적용
        # ------------------------------
        if exclude_book_ids:
            raw_results = [
                r for r in raw_results if int(r["book_id"]) not in exclude_book_ids
            ]

        # ------------------------------
        # ④ 장르 필터링 (LLM 장르 우선)
        # ------------------------------
        required_genres_en = [g.lower() for g in genres_en if g]
        final = self._filter_and_reorder_by_genre(
            results=raw_results,
            required_genres_en=required_genres_en,
            top_k=top_k,
            hard_filter_top_n=3,
        )

        return final

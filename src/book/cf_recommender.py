# src/book/cf_recommender.py

"""
CFRecommender

GoodBooks-10k의 ratings.csv + to_read.csv를 기반으로
- 사용자-아이템 상호작용(interaction) 행렬을 만들고
- item-based CF 스코어 또는 popularity 스코어로 추천을 생성하는 모듈.

외부에서 주로 사용하는 메서드
-----------------------------
- load_data()
- build_interaction_matrix()
- compute_item_similarity()   # item-based CF용
- recommend_for_user(user_id, top_k, filter_read_items)
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import sparse
from implicit.als import AlternatingLeastSquares
from implicit.cpu.als import AlternatingLeastSquares


np.random.seed(42)
logger = logging.getLogger(__name__)


class CFRecommender:
    """
    GoodBooks-10k 상호작용(평점 + to_read) 기반 협업필터링 엔진.

    파이프라인 개요
    ---------------
    1) load_data()
       - ratings.csv 로딩
       - to_read.csv 로딩 후 rating=1.0 으로 간주해서 합치기
       - min_ratings_per_user / min_ratings_per_item 기준으로 필터링
       - valid_book_ids 가 주어지면, 그 book_id들만 남김

    2) build_interaction_matrix()
       - user_id / book_id를 내부 index로 매핑
       - CSR user-item 상호작용 행렬을 생성

    3) compute_item_similarity()
       - item-based CF를 위한 item-item similarity 행렬 계산

    4) recommend_for_user()
       - item_similarity가 있으면 item-based CF 점수로 추천
       - 없으면 단순 popularity(아이템별 상호작용 수) 기반 추천
    """

    def __init__(
        self,
        ratings_csv_path: Optional[str] = None,
        to_read_csv_path: Optional[str] = None,
        min_ratings_per_user: int = 5,
        min_ratings_per_item: int = 5,
        max_items_for_similarity: Optional[int] = None,  # 현재는 미사용(placeholder)
        valid_book_ids: Optional[set[int]] = None,
    ) -> None:
        """
        Parameters
        ----------
        ratings_csv_path : str, optional
            평점 CSV 경로. None이면 프로젝트 루트 기준
            data/goodbooks-10k/ratings.csv 를 기본값으로 사용.
        to_read_csv_path : str, optional
            to_read CSV 경로. None이면
            data/goodbooks-10k/to_read.csv 를 기본값으로 사용.
        min_ratings_per_user : int
            이 값보다 적게 상호작용(평점+to_read)을 남긴 유저는 제거.
        min_ratings_per_item : int
            이 값보다 적게 상호작용을 받은 책은 제거.
        max_items_for_similarity : int, optional
            (현재 구현에서는 사용하지 않지만, 향후 확장용 placeholder)
        valid_book_ids : set[int], optional
            유효한 book_id 집합. (BookRecommender의 df 기준)
            ratings/to_read에 존재하지만 books.csv에 없는 항목들을 제거하기 위해 사용.
        """
        base_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        data_dir = os.path.join(base_dir, "data", "goodbooks-10k")

        if ratings_csv_path is None:
            ratings_csv_path = os.path.join(data_dir, "ratings.csv")
        if to_read_csv_path is None:
            to_read_csv_path = os.path.join(data_dir, "to_read.csv")

        self.ratings_csv_path = ratings_csv_path
        self.to_read_csv_path = to_read_csv_path

        self.min_ratings_per_user = min_ratings_per_user
        self.min_ratings_per_item = min_ratings_per_item
        self.max_items_for_similarity = max_items_for_similarity
        self.valid_book_ids = valid_book_ids

        # 내부 상태
        # 👉 이제 의미상 "interactions_df"이지만, 외부 영향 줄이려고 이름은 ratings_df 유지
        self.ratings_df: Optional[pd.DataFrame] = None

        # user_id ↔ index 매핑
        self.user_to_index: Dict[int, int] = {}
        self.index_to_user: Dict[int, int] = {}

        # book_id ↔ index 매핑
        self.item_to_index: Dict[int, int] = {}
        self.index_to_item: Dict[int, int] = {}

        # 상호작용 행렬 (users x items)
        self.interaction_matrix: Optional[sparse.csr_matrix] = None

        # ALS 모델 (implicit 라이브러리)
        self.als_model: Optional[AlternatingLeastSquares] = None

        # ALS 캐시 파일 경로를 만들 때 쓸 기본 데이터 디렉토리
        self._data_dir = os.path.dirname(self.ratings_csv_path)


    # --------------------------------------------------------
    # 1-1. 데이터 로딩 (ratings + to_read 통합)
    # --------------------------------------------------------

    def load_data(self) -> None:
        """
        ratings_csv_path에서 평점 데이터를 로딩하고,
        to_read_csv_path가 있으면 rating=1.0 으로 간주하여 합친 뒤,
        min_ratings_per_user / min_ratings_per_item 기준으로 필터링한다.
        """
        # 1) ratings.csv 로드
        logger.info("[CF] Loading ratings from %s", self.ratings_csv_path)
        ratings = pd.read_csv(self.ratings_csv_path)

        # 필요한 컬럼만 사용
        if not {"user_id", "book_id", "rating"}.issubset(ratings.columns):
            raise ValueError(
                f"ratings.csv에 user_id, book_id, rating 컬럼이 모두 필요합니다. "
                f"현재 컬럼: {ratings.columns.tolist()}"
            )

        ratings = ratings[["user_id", "book_id", "rating"]].dropna()
        ratings["user_id"] = ratings["user_id"].astype(int)
        ratings["book_id"] = ratings["book_id"].astype(int)
        ratings["rating"] = ratings["rating"].astype(float)

        # 2) to_read.csv 로드 (있으면) → rating = 1.0 implicit feedback
        interactions = ratings.copy()
        try:
            if self.to_read_csv_path and os.path.exists(self.to_read_csv_path):
                logger.info("[CF] Loading to_read from %s", self.to_read_csv_path)
                to_read = pd.read_csv(self.to_read_csv_path)

                if not {"user_id", "book_id"}.issubset(to_read.columns):
                    logger.warning(
                        "[CF] to_read.csv에 user_id, book_id 컬럼이 없습니다. 무시합니다."
                    )
                else:
                    to_read = to_read[["user_id", "book_id"]].dropna()
                    to_read["user_id"] = to_read["user_id"].astype(int)
                    to_read["book_id"] = to_read["book_id"].astype(int)
                    to_read["rating"] = 1.0  # 암묵적 positive feedback

                    interactions = pd.concat([interactions, to_read], ignore_index=True)
            else:
                logger.warning(
                    "[CF] to_read.csv (%s)를 찾지 못했습니다. ratings.csv만 사용합니다.",
                    self.to_read_csv_path,
                )
        except Exception as e:
            logger.warning("[CF] to_read.csv 로드 중 오류: %s", e)

        # 3) 같은 (user_id, book_id) 쌍이 여러 번 있을 수 있으므로 하나로 합치기
        #    여기서는 가장 큰 rating(=가장 강한 positive)만 남긴다.
        interactions = (
            interactions.groupby(["user_id", "book_id"])["rating"]
            .max()
            .reset_index()
        )

        # 4) valid_book_ids 기준 필터
        if self.valid_book_ids is not None:
            interactions = interactions[
                interactions["book_id"].isin(self.valid_book_ids)
            ]

        # 5) 최소 상호작용 수 기준 필터링 (유저)
        user_counts = interactions["user_id"].value_counts()
        valid_users = user_counts[user_counts >= self.min_ratings_per_user].index
        interactions = interactions[interactions["user_id"].isin(valid_users)]

        # 6) 최소 상호작용 수 기준 필터링 (아이템)
        item_counts = interactions["book_id"].value_counts()
        valid_items = item_counts[item_counts >= self.min_ratings_per_item].index
        interactions = interactions[interactions["book_id"].isin(valid_items)]

        interactions = interactions.reset_index(drop=True)
        self.ratings_df = interactions  # 이름만 ratings_df, 실제로는 interactions_df 개념

        logger.info(
            "[CF] Loaded interactions (ratings + to_read): %d rows, %d users, %d items",
            len(interactions),
            interactions["user_id"].nunique(),
            interactions["book_id"].nunique(),
        )

    # --------------------------------------------------------
    # 1-2. 상호작용 행렬 구성
    # --------------------------------------------------------

    def build_interaction_matrix(self) -> None:
        """
        ratings_df(interactions)를 기반으로 user-item CSR 상호작용 행렬을 생성한다.
        """
        if self.ratings_df is None:
            raise RuntimeError("먼저 load_data()를 호출해야 합니다.")

        df = self.ratings_df

        unique_users = df["user_id"].unique()
        unique_items = df["book_id"].unique()

        self.user_to_index = {u: idx for idx, u in enumerate(unique_users)}
        self.index_to_user = {idx: u for u, idx in self.user_to_index.items()}

        self.item_to_index = {i: idx for idx, i in enumerate(unique_items)}
        self.index_to_item = {idx: i for i, idx in self.item_to_index.items()}

        rows = df["user_id"].map(self.user_to_index).to_numpy()
        cols = df["book_id"].map(self.item_to_index).to_numpy()
        data = df["rating"].astype(float).to_numpy()

        num_users = len(unique_users)
        num_items = len(unique_items)

        matrix = sparse.csr_matrix(
            (data, (rows, cols)), shape=(num_users, num_items), dtype=np.float32
        )

        self.interaction_matrix = matrix

        logger.info(
            "[CF] Built interaction matrix: shape=%s, nnz=%d",
            matrix.shape,
            matrix.nnz,
        )

    # --------------------------------------------------------
    # 1-3. ALS 캐시 경로 유틸
    # --------------------------------------------------------
    def _get_als_cache_path(
        self,
        factors: int,
        regularization: float,
        iterations: int,
        alpha: float,
    ) -> str:
        """
        ALS 학습 결과(user_factors, item_factors)를 저장/로드할 캐시 파일 경로를 만든다.
        하이퍼파라미터에 따라 파일 이름이 달라지도록 구성.
        """
        # 하이퍼파라미터를 간단한 정수 형태로 인코딩
        reg_tag = int(regularization * 1000)
        alpha_tag = int(alpha)

        filename = f"als_model_f{factors}_r{reg_tag}_it{iterations}_a{alpha_tag}.npz"
        return os.path.join(self._data_dir, filename)

    # --------------------------------------------------------
    # 1-4. ALS 학습 및 캐시 로드
    # --------------------------------------------------------
    def fit_als(
        self,
        factors: int = 64,
        regularization: float = 0.01,
        iterations: int = 15,
        alpha: float = 40.0,
        force_retrain: bool = False,
    ) -> None:
        """
        implicit ALS 모델을 학습하거나, 기존 학습 결과를 캐시에서 로드한다.

        - self.interaction_matrix: (num_users, num_items) CSR
        - implicit ALS: (num_users, num_items) user-item matrix를 입력으로 받는다고 생각하면 됨
          → "행(row)이 user" 라고만 맞춰주면 돼.
        """
        if self.interaction_matrix is None:
            raise RuntimeError("먼저 load_data()와 build_interaction_matrix()를 호출해야 합니다.")

        num_users, num_items = self.interaction_matrix.shape
        logger = logging.getLogger(__name__)

        cache_path = self._get_als_cache_path(
            factors=factors,
            regularization=regularization,
            iterations=iterations,
            alpha=alpha,
        )

        # ----------------------------------------------------
        # 1) 캐시 있으면 로드 시도
        # ----------------------------------------------------
        if (not force_retrain) and os.path.exists(cache_path):
            try:
                logger.info("[CF/ALS] 캐시에서 ALS 모델을 로드합니다: %s", cache_path)
                data = np.load(cache_path)
                user_factors = data["user_factors"]
                item_factors = data["item_factors"]

                if user_factors.shape[0] != num_users or item_factors.shape[0] != num_items:
                    logger.warning(
                        "[CF/ALS] 캐시의 user/item 크기와 현재 interaction_matrix 크기가 맞지 않습니다. "
                        "캐시를 무시하고 새로 학습합니다. "
                        "(cache users=%d, items=%d / current users=%d, items=%d)",
                        user_factors.shape[0],
                        item_factors.shape[0],
                        num_users,
                        num_items,
                    )
                else:
                    model = AlternatingLeastSquares(
                        factors=factors,
                        regularization=regularization,
                        iterations=iterations,
                        random_state=42,
                    )
                    model.user_factors = user_factors
                    model.item_factors = item_factors

                    self.als_model = model
                    logger.info(
                        "[CF/ALS] ALS 모델 로드 완료 (users=%d, items=%d)",
                        user_factors.shape[0],
                        item_factors.shape[0],
                    )
                    return
            except Exception as e:
                logger.warning(
                    "[CF/ALS] ALS 캐시 로드 실패(%s). 새로 학습을 수행합니다.", e
                )

        # ----------------------------------------------------
        # 2) 새로 학습
        # ----------------------------------------------------
        logger.info(
            "[CF/ALS] ALS 모델을 새로 학습합니다 (factors=%d, reg=%.4f, it=%d, alpha=%.1f)",
            factors,
            regularization,
            iterations,
            alpha,
        )

        # ✅ 여기서 더 이상 transpose 하지 말기!
        user_item_matrix = self.interaction_matrix.tocsr().astype("float32")
        # user_item_matrix.shape == (num_users, num_items)

        model = AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=iterations,
            random_state=42,
        )

        model.fit(user_item_matrix)

        logger.info(
            "[CF/ALS] 학습 완료: user_factors=%s, item_factors=%s",
            model.user_factors.shape,
            model.item_factors.shape,
        )

        # 이제는 반대로 나와야 정상:
        # user_factors.shape[0] == num_users(44783), item_factors.shape[0] == num_items(812)
        assert model.user_factors.shape[0] == num_users, "ALS user_factors 크기가 num_users와 다릅니다"
        assert model.item_factors.shape[0] == num_items, "ALS item_factors 크기가 num_items와 다릅니다"

        self.als_model = model

        # ----------------------------------------------------
        # 3) 캐시 저장
        # ----------------------------------------------------
        try:
            os.makedirs(self._data_dir, exist_ok=True)
            np.savez_compressed(
                cache_path,
                user_factors=model.user_factors,
                item_factors=model.item_factors,
            )
            logger.info(
                "[CF/ALS] ALS 학습 결과를 캐시에 저장했습니다: %s", cache_path
            )
        except Exception as e:
            logger.warning("[CF/ALS] ALS 캐시 저장 실패: %s", e)

    # --------------------------------------------------------
    # 1-4. 추천 함수
    # --------------------------------------------------------

    def _get_user_index(self, user_id: int) -> Optional[int]:
        return self.user_to_index.get(int(user_id))

    def _get_seen_item_indices(self, user_idx: int) -> np.ndarray:
        """
        해당 user_idx가 이미 상호작용한 아이템 인덱스 리스트를 반환.
        """
        if self.interaction_matrix is None:
            return np.array([], dtype=np.int64)

        user_row = self.interaction_matrix.getrow(user_idx)
        return user_row.indices

    def recommend_for_user(
        self,
        user_id: int,
        top_k: int = 20,
        filter_read_items: bool = True,
    ) -> List[Dict[str, float]]:
        """
        특정 user_id에 대해 ALS 기반 상위 top_k 추천을 반환.

        - 기본: implicit ALS 모델(self.als_model)을 사용
        - als_model이 없으면: popularity 기반으로 fallback
        """
        if self.interaction_matrix is None:
            raise RuntimeError("먼저 load_data()와 build_interaction_matrix()를 호출해야 합니다.")

        user_idx = self._get_user_index(user_id)
        if user_idx is None:
            logger.warning(
                "[CF/ALS] Unknown user_id=%s (cold-start). popularity 기반 추천으로 대체합니다.",
                user_id,
            )
            # cold-start 사용자는 popularity만 쓸 수 있음
            return self._recommend_by_popularity(top_k=top_k)

        user_idx = int(user_idx)

        # ----------------------------------------------------
        # 1) ALS 모델이 있으면 ALS 기반 추천
        # ----------------------------------------------------
        if self.als_model is not None:
            try:
                user_items = self.interaction_matrix[user_idx]

                item_indices, scores = self.als_model.recommend(
                    user_idx,
                    user_items,
                    N=top_k,
                    filter_already_liked_items=filter_read_items,
                )

                results: List[Dict[str, float]] = []
                for idx, score in zip(item_indices, scores):
                    book_id = int(self.index_to_item[int(idx)])
                    results.append(
                        {
                            "book_id": book_id,
                            "score": float(score),
                            "title": None,
                            "authors": None,
                        }
                    )

                return results[:top_k]
            except Exception as e:
                logger.exception(
                    "[CF/ALS] ALS 기반 추천 중 오류 발생 (user_id=%s): %s. popularity로 fallback 합니다.",
                    user_id,
                    e,
                )

        # ----------------------------------------------------
        # 2) ALS 모델이 없거나 실패한 경우 → popularity fallback
        # ----------------------------------------------------
        return self._recommend_by_popularity(
            top_k=top_k,
            user_idx=user_idx if filter_read_items else None,
        )

    # --------------------------------------------------------
    # 1-5. popularity 기반 추천 (fallback용 내부 유틸)
    # --------------------------------------------------------
    def _recommend_by_popularity(
        self,
        top_k: int = 20,
        user_idx: Optional[int] = None,
    ) -> List[Dict[str, float]]:
        """
        아이템별 상호작용 수(=popularity)를 기준으로 상위 top_k 아이템을 추천.
        user_idx가 주어지면, 해당 유저가 이미 본 아이템은 제외한다.
        """
        if self.interaction_matrix is None:
            return []

        item_popularity = np.asarray(self.interaction_matrix.sum(axis=0)).ravel()

        if user_idx is not None:
            seen_idx = self._get_seen_item_indices(user_idx)
            item_popularity[seen_idx] = -np.inf

        top_indices = np.argsort(item_popularity)[::-1][:top_k]

        results: List[Dict[str, float]] = []
        for idx in top_indices:
            if item_popularity[idx] == -np.inf:
                continue
            book_id = int(self.index_to_item[int(idx)])
            score = float(item_popularity[idx])

            results.append(
                {
                    "book_id": book_id,
                    "score": score,
                    "title": None,
                    "authors": None,
                }
            )

        return results[:top_k]

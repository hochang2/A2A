# src/book/cf_recommender.py

"""
CFRecommender

GoodBooks-10k의 ratings.csv를 기반으로
- 사용자-아이템 상호작용 행렬을 만들고
- (선택) implicit ALS 모델 학습
- ALS 또는 간단한 CF 스코어로 추천을 생성하는 모듈.

외부에서 주로 사용하는 메서드
-----------------------------
- load_data()
- build_interaction_matrix()
- compute_item_similarity()   # 현재 구현에서는 필수는 아님 (직접 점수 계산 방식 사용)
- train_als_model()
- recommend_for_user(user_id, top_k, filter_read_items)
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
np.random.seed(42)
logger = logging.getLogger(__name__)

try:
    from implicit.als import AlternatingLeastSquares
except ImportError:  # implicit 미설치 환경에서도 import 에러 안 나게
    AlternatingLeastSquares = None


# ============================================================
# 1. CFRecommender 본체
# ============================================================


class CFRecommender:
    """
    GoodBooks-10k ratings 기반 협업필터링 엔진.

    파이프라인 개요
    ---------------
    1) load_data()
       - ratings.csv 로딩
       - min_ratings_per_user / min_ratings_per_item 기준으로 필터링
       - valid_book_ids 가 주어지면, 그 book_id들만 남김

    2) build_interaction_matrix()
       - user_id / book_id를 내부 index로 매핑
       - CSR user-item 상호작용 행렬을 생성

    3) (선택) train_als_model()
       - implicit ALS로 잠재 요인 모델 학습

    4) recommend_for_user()
       - use_als=True 이고 ALS 모델이 있으면 ALS로 추천
       - 그 외에는 간단한 user-item 내적 기반 점수로 추천
    """

    def __init__(
        self,
        ratings_csv_path: Optional[str] = None,
        min_ratings_per_user: int = 5,
        min_ratings_per_item: int = 5,
        max_items_for_similarity: Optional[int] = None,
        use_als: bool = False,
        valid_book_ids: Optional[set[int]] = None,
        als_factors: int = 64,
        als_regularization: float = 0.01,
        als_iterations: int = 15,
    ) -> None:
        """
        Parameters
        ----------
        ratings_csv_path : str, optional
            평점 CSV 경로. None이면 프로젝트 루트 기준
            data/goodbooks-10k/ratings.csv 를 기본값으로 사용.
        min_ratings_per_user : int
            이 값보다 적게 평점을 남긴 유저는 제거.
        min_ratings_per_item : int
            이 값보다 적게 평점을 받은 책은 제거.
        max_items_for_similarity : int, optional
            (지금 구현에서는 직접 item_similarity를 쓰지 않으므로
             placeholder용. 향후 확장 시 사용 가능)
        use_als : bool
            True이면 ALS 모델 학습 및 ALS 기반 추천을 사용.
        valid_book_ids : set[int], optional
            유효한 book_id 집합. (BookRecommender의 df 기준)
            ratings에 존재하지만 books.csv에 없는 항목들을 제거하기 위해 사용.
        als_factors, als_regularization, als_iterations
            implicit ALS 하이퍼파라미터.
        """
        if ratings_csv_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            ratings_csv_path = os.path.join(
                base_dir, "data", "goodbooks-10k", "ratings.csv"
            )

        self.ratings_csv_path = ratings_csv_path
        self.min_ratings_per_user = min_ratings_per_user
        self.min_ratings_per_item = min_ratings_per_item
        self.max_items_for_similarity = max_items_for_similarity
        self.use_als = use_als
        self.valid_book_ids = valid_book_ids

        # ALS 파라미터
        self.als_factors = als_factors
        self.als_regularization = als_regularization
        self.als_iterations = als_iterations

        # 내부 상태
        self.ratings_df: Optional[pd.DataFrame] = None

        # user_id ↔ index 매핑
        self.user_to_index: Dict[int, int] = {}
        self.index_to_user: Dict[int, int] = {}

        # book_id ↔ index 매핑
        self.item_to_index: Dict[int, int] = {}
        self.index_to_item: Dict[int, int] = {}

        # 상호작용 행렬 (users x items)
        self.interaction_matrix: Optional[sparse.csr_matrix] = None

        # (선택) ALS 모델/요인
        self.als_model: Optional[AlternatingLeastSquares] = None
        self.item_similarity: Optional[sparse.csr_matrix] = None

    # --------------------------------------------------------
    # 1-1. 데이터 로딩
    # --------------------------------------------------------

    def load_data(self) -> None:
        """
        ratings_csv_path에서 평점 데이터를 로딩하고,
        최소 평점 수 기준 / valid_book_ids 기준으로 필터링한다.

        기대 컬럼
        ---------
        - user_id
        - book_id
        - rating
        """
        logger.info("[CF] Loading ratings from %s", self.ratings_csv_path)
        df = pd.read_csv(self.ratings_csv_path)

        required_cols = {"user_id", "book_id", "rating"}
        if not required_cols.issubset(df.columns):
            raise ValueError(
                f"ratings.csv 에 {required_cols} 컬럼이 모두 필요합니다. "
                f"현재 컬럼: {df.columns.tolist()}"
            )

        # valid_book_ids 가 주어졌다면 그에 맞게 필터링
        if self.valid_book_ids is not None:
            df = df[df["book_id"].isin(self.valid_book_ids)]

        # 최소 평점 수 기준 필터링 (유저)
        user_counts = df["user_id"].value_counts()
        valid_users = user_counts[user_counts >= self.min_ratings_per_user].index
        df = df[df["user_id"].isin(valid_users)]

        # 최소 평점 수 기준 필터링 (아이템)
        item_counts = df["book_id"].value_counts()
        valid_items = item_counts[item_counts >= self.min_ratings_per_item].index
        df = df[df["book_id"].isin(valid_items)]

        df = df.reset_index(drop=True)
        self.ratings_df = df

        logger.info(
            "[CF] Loaded ratings: %d rows, %d users, %d items",
            len(df),
            df["user_id"].nunique(),
            df["book_id"].nunique(),
        )

    # --------------------------------------------------------
    # 1-2. 상호작용 행렬 구성
    # --------------------------------------------------------

    def build_interaction_matrix(self) -> None:
        """
        ratings_df를 기반으로 user-item CSR 상호작용 행렬을 생성한다.
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

        # (row, col, data) 형태로 CSR 구성
        rows = df["user_id"].map(self.user_to_index).to_numpy()
        cols = df["book_id"].map(self.item_to_index).to_numpy()

        # 여기서는 explicit rating을 그대로 쓰거나, implicit으로 1.0만 쓰는 등 선택 가능
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
    # 1-3. (선택) item_similarity / ALS 학습
    # --------------------------------------------------------

    def compute_item_similarity(self) -> None:
        """
        item-based CF를 위한 item-item similarity 행렬을 계산한다.

        아이디어:
        - A = interaction_matrix (users x items)
        - item_co_counts = A.T @ A  (items x items)  → 공출현 정도
        - 대각 성분을 이용해 cosine 유사도 형태로 정규화:
          sim[i,j] = co[i,j] / sqrt(co[i,i] * co[j,j])
        - 결과를 sparse matrix로 self.item_similarity에 저장
        """
        if self.interaction_matrix is None:
            raise RuntimeError("먼저 build_interaction_matrix()를 호출해야 합니다.")

        # items x users
        item_user_matrix = self.interaction_matrix.T.tocsr()

        # 공출현 행렬 (items x items)
        item_co_counts = item_user_matrix @ item_user_matrix.T  # sparse (I x I)

        # 대각 성분 (각 아이템의 자기 공출현: co[i,i])
        diag = item_co_counts.diagonal().astype(np.float32)
        # 0으로 나누는 것 방지용
        diag[diag == 0] = 1e-8
        inv_sqrt_diag = 1.0 / np.sqrt(diag)

        # coo 포맷으로 바꿔서 각 원소에 대해 정규화 진행
        coo = item_co_counts.tocoo()
        # cos_sim[i,j] = co[i,j] * (1/sqrt(diag[i])) * (1/sqrt(diag[j]))
        data = coo.data * inv_sqrt_diag[coo.row] * inv_sqrt_diag[coo.col]

        # 자기 자신(sim[i,i])은 의미 없으니 0으로 두고 싶다면 여기서 필터링 가능(선택)
        # 예: i != j 인 것만 남기고 싶으면:
        # mask = coo.row != coo.col
        # row = coo.row[mask]
        # col = coo.col[mask]
        # data = data[mask]
        # sim_matrix = sparse.csr_matrix((data, (row, col)), shape=item_co_counts.shape)

        sim_matrix = sparse.csr_matrix((data, (coo.row, coo.col)), shape=item_co_counts.shape)

        self.item_similarity = sim_matrix

        logger.info(
            "[CF] Computed item-item similarity matrix: shape=%s, nnz=%d",
            sim_matrix.shape,
            sim_matrix.nnz,
        )


    def train_als_model(self) -> None:
        """
        implicit ALS 모델을 학습한다. use_als=True일 때 사용.

        - ALS는 implicit 피드백(시청, 클릭, 조회 등)에 잘 맞는 모델.
        - 여기서는 rating을 그대로 implicit strength로 사용하거나,
          필요하면 전처리 과정에서 가중치를 조정할 수 있음.
        """
        if not self.use_als:
            logger.info("[CF][ALS] use_als=False 이므로 ALS 학습을 건너뜁니다.")
            return

        if AlternatingLeastSquares is None:
            raise RuntimeError(
                "implicit 라이브러리가 설치되어 있지 않습니다. "
                "pip install implicit 로 설치 후 다시 시도해 주세요."
            )

        if self.interaction_matrix is None:
            raise RuntimeError("먼저 build_interaction_matrix()를 호출해야 합니다.")

        # implicit ALS는 (items x users) 형식의 행렬을 기대
        # transpose 해서 넘겨준다.
        item_user_matrix = self.interaction_matrix.T.tocsr()

        logger.info(
            "[CF][ALS] Training ALS model (factors=%d, reg=%.4f, iter=%d)...",
            self.als_factors,
            self.als_regularization,
            self.als_iterations,
        )

        model = AlternatingLeastSquares(
            factors=self.als_factors,
            regularization=self.als_regularization,
            iterations=self.als_iterations,
            random_state=42,
        )

        # implicit ALS는 data에 negative weight가 있으면 안되므로 abs 사용 가능
        model.fit(item_user_matrix)

        self.als_model = model

        logger.info(
            "[CF][ALS] Training complete. user_factors=%s, item_factors=%s",
            model.user_factors.shape,
            model.item_factors.shape,
        )

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
        특정 user_id에 대해 상위 top_k 추천을 반환.

        반환 형식
        --------
        [
            {
                "book_id": int,
                "score": float,
                "title": Optional[str],  # 여기서는 title 정보가 없으므로 빈값
                "authors": Optional[str],
            },
            ...
        ]
        """
        if self.interaction_matrix is None:
            raise RuntimeError("먼저 build_interaction_matrix()를 호출해야 합니다.")

        # 외부 user_id → 내부 인덱스 (0 ~ num_users-1)
        user_idx = self._get_user_index(user_id)
        if user_idx is None:
            logger.warning("[CF] Unknown user_id=%s (cold-start). 빈 추천을 반환합니다.", user_id)
            return []

        user_idx = int(user_idx)
        user_row = self.interaction_matrix.getrow(user_idx)  # (1, num_items)

                 # ----------------------------------------------------
        # 1) ALS 기반 추천 (선택)
        # ----------------------------------------------------
        if self.use_als and self.als_model is not None:
            # implicit ALS + recalculate_user 모드 사용
            # → userid는 더미 (0)로 두고,
            #   실제 유저 정보는 user_items로 전달해서 on-the-fly로 유저 벡터를 만든다.
            user_items = self.interaction_matrix[user_idx]

            rec_items, rec_scores = self.als_model.recommend(
                userid=0,                    # 더미 index (user_factors 범위 안)
                user_items=user_items,
                N=top_k,
                filter_already_liked_items=False,
                recalculate_user=True,       # user_items 기반으로 유저 벡터 재계산
            )

            results: List[Dict[str, float]] = []
            for raw_idx, score in zip(rec_items, rec_scores):
                idx_int = int(raw_idx)

                # 1) ALS가 내부 인덱스(0 ~ num_items-1)를 반환한 경우
                if idx_int in self.index_to_item:
                    book_id = int(self.index_to_item[idx_int])
                else:
                    # 2) 혹시 원래 book_id를 반환하는 경우 → 그대로 사용
                    book_id = idx_int
                    # 우리 상호작용 행렬에 없는 book_id라면 스킵
                    if book_id not in self.item_to_index:
                        continue

                results.append(
                    {
                        "book_id": book_id,
                        "score": float(score),
                        "title": None,
                        "authors": None,
                    }
                )

            # (안전용) 이미 본 아이템 제거
            if filter_read_items:
                seen_indices = set(self._get_seen_item_indices(user_idx))
            else:
                seen_indices = set()

            filtered_results: List[Dict[str, float]] = []
            for r in results:
                # book_id가 매핑에 없으면 그냥 통과 (혹은 필요하면 continue)
                idx = self.item_to_index.get(r["book_id"])
                if idx is None:
                    continue
                if idx in seen_indices:
                    continue
                filtered_results.append(r)

            return filtered_results[:top_k]

        # ----------------------------------------------------
        # 2) item-based CF (item_similarity가 있을 경우)
        # ----------------------------------------------------
        if self.item_similarity is not None:
            # user_row: (1, num_items)
            # item_similarity: (num_items, num_items)
            # → scores: (1, num_items) = user_row * item_similarity
            scores_matrix = user_row @ self.item_similarity  # (1, num_items)
            scores = np.asarray(scores_matrix.todense()).ravel()  # (num_items,)

            # 이미 본 아이템 제거 옵션
            if filter_read_items:
                seen_idx = self._get_seen_item_indices(user_idx)
                scores[seen_idx] = -np.inf

            # 상위 top_k 인덱스 선택
            top_indices = np.argsort(scores)[::-1][:top_k]

            results: List[Dict[str, float]] = []
            for idx in top_indices:
                if scores[idx] == -np.inf:
                    continue
                book_id = int(self.index_to_item[int(idx)])
                score = float(scores[idx])
                results.append(
                    {
                        "book_id": book_id,
                        "score": score,
                        "title": None,
                        "authors": None,
                    }
                )

            return results[:top_k]

        # ----------------------------------------------------
        # 3) (fallback) 단순 popularity 기반 추천
        # ----------------------------------------------------
        item_popularity = np.asarray(self.interaction_matrix.sum(axis=0)).ravel()

        if filter_read_items:
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

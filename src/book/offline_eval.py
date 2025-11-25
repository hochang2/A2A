# src/book/offline_eval.py

import os
from datetime import datetime
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from .recommender import BookRecommender
from .cf_recommender import CFRecommender
from .graph_book import merge_candidates
from .metrics import ndcg_at_k, recall_at_k
from src.config import HYBRID_ALPHA_CONTENT

# ============================================================
# 0. 경로 및 환경 변수 설정
# ============================================================

# offline_eval.py 위치: a2a/src/book/offline_eval.py
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 예: C:\Users\lhc\Desktop\a2a

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "goodbooks-10k")
RATINGS_CSV = os.path.join(DATA_DIR, "ratings.csv")

# 결과 저장 디렉토리 및 파일
RESULT_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULT_DIR, exist_ok=True)
RESULT_CSV = os.path.join(RESULT_DIR, "offline_eval_results.csv")


# ============================================================
# 1. 데이터 로딩 및 유저별 train/test 분리
# ============================================================

def load_ratings(min_ratings_per_user: int = 10) -> pd.DataFrame:
    """
    ratings.csv를 로드하고, 최소 평점 개수 이상 남긴 유저만 남긴다.
    """
    df = pd.read_csv(RATINGS_CSV)
    user_counts = df["user_id"].value_counts()
    valid_users = user_counts[user_counts >= min_ratings_per_user].index
    df = df[df["user_id"].isin(valid_users)].reset_index(drop=True)
    return df


def train_test_split_per_user(
    df: pd.DataFrame,
    test_ratio: float = 0.2,
) -> Dict[int, Tuple[List[int], List[int]]]:
    """
    각 user_id에 대해 (train_items, test_items) 리스트를 만든다.
    여기서는 rating >= 4 인 책들만 positive로 본다고 가정.
    """
    user_pos_items: Dict[int, Tuple[List[int], List[int]]] = {}

    for user_id, group in df.groupby("user_id"):
        pos = group[group["rating"] >= 4]["book_id"].tolist()
        if len(pos) < 2:
            continue

        np.random.shuffle(pos)
        n_test = max(1, int(len(pos) * test_ratio))
        test_items = pos[:n_test]
        train_items = pos[n_test:]

        user_pos_items[user_id] = (train_items, test_items)

    return user_pos_items


# ============================================================
# 2. 콘텐츠 기반: TF-IDF를 이용한 유저 프로필 & 후보 생성
# ============================================================

def build_user_profile_tfidf(
    train_items: List[int],
    content_rec: BookRecommender,
):
    """
    유저가 과거에 본 책(train_items)의 TF-IDF 벡터를 평균내서
    '유저 프로필 벡터'를 만드는 함수.

    반환값:
        - shape (n_features,)인 1D numpy 배열 또는 None
    """
    if not train_items:
        return None

    df = content_rec.df
    tfidf = content_rec.tfidf_matrix  # (n_books, n_features) sparse matrix

    # train_items에 해당하는 row 인덱스 찾기
    rows = df.index[df["book_id"].isin(train_items)].tolist()
    if not rows:
        return None

    # 해당 책들의 TF-IDF 벡터 평균 → (1, n_features)
    user_vec = tfidf[rows].mean(axis=0)

    # numpy 1D array로 변환
    user_vec = np.asarray(user_vec).ravel()  # (n_features,)

    # L2 정규화
    norm = np.linalg.norm(user_vec)
    if norm > 0:
        user_vec = user_vec / norm

    return user_vec


def get_content_candidates_from_profile_tfidf(
    user_vec,
    content_rec: BookRecommender,
    top_k: int = 50,
):
    """
    유저 프로필 TF-IDF 벡터(user_vec)와
    모든 책의 TF-IDF 벡터(content_rec.tfidf_matrix)의 내적(dot)으로
    유사도를 계산해 콘텐츠 후보 리스트를 만든다.
    """
    if user_vec is None:
        return []

    tfidf = content_rec.tfidf_matrix  # (n_books, n_features)
    df = content_rec.df

    # sims: (n_books,)
    sims = tfidf.dot(user_vec)
    sims = np.asarray(sims).ravel()

    # 상위 top_k 인덱스
    top_idx = np.argsort(sims)[::-1][:top_k]

    results = []
    for idx in top_idx:
        row = df.iloc[idx]
        results.append(
            {
                "book_id": int(row["book_id"]),
                "title": str(row["title"]),
                "authors": str(row.get("authors", "")),
                "score": float(sims[idx]),
            }
        )

    # 0~1 정규화
    if results:
        max_score = max(r["score"] for r in results)
        if max_score > 0:
            for r in results:
                r["score"] = r["score"] / max_score

    return results


def build_profile_preference_text(
    train_items: List[int],
    content_rec: BookRecommender,
    max_genres: int = 3,
    max_titles: int = 3,
) -> str:
    """
    유저가 과거에 좋아한 책(train_items)을 기반으로
    - 자주 등장한 장르(top-N)
    - 대표 타이틀 몇 개
    를 이용해 preference_text를 만든다.
    """
    if not train_items:
        return ""

    df = content_rec.df
    user_books = df[df["book_id"].isin(train_items)].copy()
    if user_books.empty:
        return ""

    # 1) 장르 기반 프로필
    genre_tokens: List[str] = []
    if "genres" in user_books.columns:
        genre_series = user_books["genres"].fillna("")
        all_genres: List[str] = []
        for g in genre_series:
            if isinstance(g, str) and g.strip():
                all_genres.extend([s.strip() for s in g.split("|") if s.strip()])

        if all_genres:
            genre_counts = pd.Series(all_genres).value_counts()
            top_genres = genre_counts.head(max_genres).index.tolist()
            genre_tokens = top_genres

    # 2) 대표 타이틀 일부
    title_tokens: List[str] = []
    if "title" in user_books.columns:
        title_tokens = (
            user_books["title"]
            .astype(str)
            .dropna()
            .head(max_titles)
            .tolist()
        )

    parts: List[str] = []
    if genre_tokens:
        parts.append(" ".join(genre_tokens))
    if title_tokens:
        parts.append(" ".join(title_tokens))

    preference_text = " ".join(parts).strip()
    return preference_text


# ============================================================
# 3. 추천 함수 (offline_eval용)
# ============================================================

def recommend_for_eval(
    user_id: int,
    content_rec: BookRecommender,
    cf_rec: CFRecommender,
    train_items: List[int] | None,
    top_k: int = 50,
    mode: str = "cf_only",
) -> List[int]:
    """
    offline_eval 용 추천 함수.

    mode 설명
    --------
    - "cf_only":
        → CFRecommender 기반 협업필터링만 사용
    - "hybrid_v2":
        → CF + TF-IDF 유저 프로필 기반 콘텐츠 추천을 merge
          (user_vec = train_items의 TF-IDF 평균)
    - "hybrid_best":
        → CF + 장르/타이틀 기반 preference_text로 만든 콘텐츠 추천을 merge
    """
    # 1) CF-only baseline
    if mode in ("cf_only", "cf_als"):
        cf_candidates = cf_rec.recommend_for_user(
            user_id=user_id,
            top_k=top_k,
            filter_read_items=False,  # 평가에서는 숨기지 않는다
        )
        return [int(c["book_id"]) for c in cf_candidates]

    # 2) hybrid_v2: TF-IDF 유저 프로필 벡터 기반
    elif mode == "hybrid_v2":
        cf_candidates = cf_rec.recommend_for_user(
            user_id=user_id,
            top_k=top_k,
            filter_read_items=False,
        )

        user_vec = build_user_profile_tfidf(train_items, content_rec)
        content_candidates = get_content_candidates_from_profile_tfidf(
            user_vec,
            content_rec,
            top_k=top_k,
        )

        if not cf_candidates and not content_candidates:
            return []

        if content_candidates and not cf_candidates:
            return [int(c["book_id"]) for c in content_candidates[:top_k]]
        if cf_candidates and not content_candidates:
            return [int(c["book_id"]) for c in cf_candidates[:top_k]]

        merged = merge_candidates(
            content_candidates=content_candidates,
            cf_candidates=cf_candidates,
            alpha=HYBRID_ALPHA_CONTENT,
        )
        return [int(c["book_id"]) for c in merged[:top_k]]

    # 3) hybrid_best: 장르/타이틀 기반 preference_text 활용
    elif mode == "hybrid_best":
        cf_candidates = cf_rec.recommend_for_user(
            user_id=user_id,
            top_k=top_k,
            filter_read_items=False,
        )

        content_candidates: List[Dict[str, any]] = []
        if train_items:
            preference_text = build_profile_preference_text(
                train_items=train_items,
                content_rec=content_rec,
                max_genres=3,
                max_titles=3,
            )
            if preference_text:
                content_candidates = content_rec.recommend_with_preferences(
                    preference_text=preference_text,
                    mood_keywords=None,
                    genres=None,
                    top_k=top_k,
                )

        if not cf_candidates and not content_candidates:
            return []

        if content_candidates and not cf_candidates:
            return [int(c["book_id"]) for c in content_candidates[:top_k]]
        if cf_candidates and not content_candidates:
            return [int(c["book_id"]) for c in cf_candidates[:top_k]]

        merged = merge_candidates(
            content_candidates=content_candidates,
            cf_candidates=cf_candidates,
            alpha=HYBRID_ALPHA_CONTENT,
        )
        return [int(c["book_id"]) for c in merged[:top_k]]

    # 4) 알 수 없는 mode면 CF-only로 fallback
    else:
        cf_candidates = cf_rec.recommend_for_user(
            user_id=user_id,
            top_k=top_k,
            filter_read_items=False,
        )
        return [int(c["book_id"]) for c in cf_candidates]


# ============================================================
# 4. 전체 오프라인 평가 실행
# ============================================================

def run_offline_eval(
    k_list=(5, 10, 20),
    max_users: int = 500,
    mode: str = "cf_only",
):
    """
    전체 오프라인 평가 루프.

    - ratings.csv → 유저별 train/test 분리
    - Recommender 초기화
    - 유저별 추천 → NDCG@k, Recall@k 계산
    - 결과를 CSV에 append
    """
    print("[INFO] Load ratings...")
    df = load_ratings(min_ratings_per_user=10)
    user_pos = train_test_split_per_user(df, test_ratio=0.2)

    print("[INFO] Init recommenders...")
    content_rec = BookRecommender()

    # ✅ mode에 따라 CF 설정
    if mode in ("cf_only", "hybrid_v2", "hybrid_best"):
        # 기본: item-based CF (ALS OFF)
        cf_rec = CFRecommender(use_als=False)
        cf_rec.load_data()
        cf_rec.build_interaction_matrix()
        cf_rec.compute_item_similarity()  # 이제 실제 item-based CF용 similarity 계산

    elif mode == "cf_als":
        # ALS 기반 CF
        cf_rec = CFRecommender(use_als=True)
        cf_rec.load_data()
        cf_rec.build_interaction_matrix()
        cf_rec.train_als_model()  # ALS 학습

    else:
        # 혹시 모르는 fallback
        cf_rec = CFRecommender(use_als=False)
        cf_rec.load_data()
        cf_rec.build_interaction_matrix()
        cf_rec.compute_item_similarity()


    results = {k: {"ndcg": [], "recall": []} for k in k_list}

    users = list(user_pos.keys())
    np.random.shuffle(users)
    users = users[:max_users]

    for idx, user_id in enumerate(users, start=1):
        train_items, test_items = user_pos[user_id]

        rec_ids = recommend_for_eval(
            user_id=user_id,
            content_rec=content_rec,
            cf_rec=cf_rec,
            train_items=train_items,
            top_k=max(k_list),
            mode=mode,
        )

        if not rec_ids:
            continue

        relevances = [1 if bid in test_items else 0 for bid in rec_ids]

        for k in k_list:
            nd = ndcg_at_k(relevances, k)
            rc = recall_at_k(test_items, rec_ids, k)
            results[k]["ndcg"].append(nd)
            results[k]["recall"].append(rc)

        if idx % 50 == 0:
            print(f"[PROGRESS] {idx}/{len(users)} users evaluated")

    print("\n=== Offline Evaluation Results ===")
    rows = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    n_users_eval = len(users)

    for k in k_list:
        ndcg_mean = float(np.mean(results[k]["ndcg"])) if results[k]["ndcg"] else 0.0
        recall_mean = float(np.mean(results[k]["recall"])) if results[k]["recall"] else 0.0
        print(f"mode={mode}, k={k}: NDCG@{k}={ndcg_mean:.4f}, Recall@{k}={recall_mean:.4f}")

        rows.append(
            {
                "timestamp": timestamp,
                "mode": mode,
                "k": k,
                "ndcg": ndcg_mean,
                "recall": recall_mean,
                "n_users": n_users_eval,
            }
        )

    # CSV로 저장 (append)
    df_result = pd.DataFrame(rows)
    if os.path.exists(RESULT_CSV):
        df_result.to_csv(
            RESULT_CSV,
            mode="a",
            header=False,
            index=False,
            encoding="utf-8",
        )
    else:
        df_result.to_csv(
            RESULT_CSV,
            mode="w",
            header=True,
            index=False,
            encoding="utf-8",
        )

    print(f"\n[INFO] Saved evaluation results to {RESULT_CSV}")


# ============================================================
# 5. CLI 실행 진입점
# ============================================================

if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "cf_only"
    run_offline_eval(mode=mode)

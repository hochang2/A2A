# src/book/user_history.py
from __future__ import annotations

import os
from typing import Dict, Any, Set, List

import pandas as pd
from collections import Counter


def load_user_profile(user_id: int, books_df: pd.DataFrame) -> Dict[str, Any]:
    """
    my_ratings.csv를 읽어서 해당 user_id의
    - seen_book_ids: 이미 선택/평가한 book_id 집합
    - top_genres: 자주 고른 장르 상위 N개
    를 계산해서 반환한다.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ratings_path = os.path.join(base_dir, "data", "my_ratings.csv")

    seen_book_ids: Set[int] = set()
    genre_counter: Counter[str] = Counter()

    if os.path.exists(ratings_path):
        df = pd.read_csv(ratings_path)

        # 해당 user만 필터
        df_user = df[df["user_id"] == int(user_id)]

        for _, row in df_user.iterrows():
            bid = int(row["book_id"])
            seen_book_ids.add(bid)

            # 이 책의 장르를 books_df에서 찾아서 카운트
            book_row = books_df[books_df["book_id"] == bid]
            if not book_row.empty:
                genres_str = str(book_row.iloc[0].get("genres", "")).lower()
                # "horror, thriller" / "horror;thriller" 등 처리
                genres = [
                    g.strip()
                    for g in genres_str.replace(";", ",").split(",")
                    if g.strip()
                ]
                for g in genres:
                    genre_counter[g] += 1

    # 상위 최대 5개 장르만 preference로 사용
    top_genres: List[str] = [g for g, _ in genre_counter.most_common(5)]

    return {
        "seen_book_ids": seen_book_ids,
        "top_genres": top_genres,
    }

# src/book/metrics.py
from typing import List
import numpy as np


def ndcg_at_k(relevances: List[int], k: int) -> float:
    """
    relevances: 추천 리스트에 대한 relevance (0/1 또는 0~3 같은 정수)
    k: 상위 k개까지만 평가
    """
    r = np.asarray(relevances)[:k]
    if r.size == 0:
        return 0.0

    # DCG
    dcg = np.sum(r / np.log2(np.arange(2, r.size + 2)))

    # IDCG (이 이상 잘 나올 수 없는 이상적인 DCG)
    ideal_r = np.sort(r)[::-1]
    idcg = np.sum(ideal_r / np.log2(np.arange(2, ideal_r.size + 2)))

    return float(dcg / idcg) if idcg > 0 else 0.0


def recall_at_k(relevant_items: List[int], recommended_items: List[int], k: int) -> float:
    """
    relevant_items: 이 유저에게 '정답'인 아이템 id들 (예: 테스트에 쓴 책들)
    recommended_items: 추천 결과 book_id 리스트
    """
    if not relevant_items:
        return 0.0

    recommended_at_k = recommended_items[:k]
    hits = sum(1 for x in recommended_at_k if x in relevant_items)
    return float(hits / len(relevant_items))

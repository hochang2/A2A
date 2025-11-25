# src/book/graph_book.py

"""
LangGraph 기반 Book 추천 파이프라인.

구조 개요
---------
1) LLM Decider (llm_decider.decide_strategy_with_llm)
   - user_input을 받아서 현재 감정, 원하는 감정, 장르, 전략(by_title / by_mood 등)을 JSON으로 파싱.
   - 결과는 state["decision"]에 저장.

2) Candidate Generation (BookRecommender + CFRecommender)
   - 콘텐츠 기반 후보:
       BookRecommender.recommend_from_llm_decision(llm_decision, top_k)
   - CF 기반 후보:
       CFRecommender.recommend_for_user(user_id, top_k, filter_read_items=True)
   - 두 후보를 merge_candidates()로 합쳐서
       - content_score, cf_score, hybrid_score를 계산.
   - 결과를 state["candidates"]에 저장.

3) LLM Reranker (llm_reranker.rerank_with_llm)
   - 입력: user_input, llm_decision, candidates
   - 출력: {
       "reranked": [ ... 책 dict ... ],
       "natural_output": "사용자에게 보여줄 자연어 추천 문장"
     }
   - 결과를 state["reranked"], state["natural_output"]에 저장.

4) run_book_recommendation()
   - 외부(예: CLI, API)에서 호출하는 헬퍼.
   - BookState 초기화 → graph.invoke → 최종 BookState 반환.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from langgraph.graph import StateGraph, END

from src.common.state_types import BaseRecState
from src.config import (
    HYBRID_ALPHA_CONTENT,
    MAX_CANDIDATES_FOR_LLM,
)
from .recommender import BookRecommender
from .cf_recommender import CFRecommender
from . import llm_decider
from . import llm_reranker

logger = logging.getLogger(__name__)


# ============================================================
# 1. State 정의
# ============================================================


class BookState(BaseRecState):
    """
    LangGraph에서 사용하는 Book 도메인 상태 타입.

    필드
    ----
    user_input : 사용자가 입력한 자연어 문장
    user_id    : GoodBooks-10k 기준 user_id (int)
    decision   : llm_decider가 반환한 JSON(dict)
    candidates : LLM에 보내기 전, 전통 추천 시스템이 뽑아 놓은 후보 리스트
    reranked   : LLM reranker가 최종적으로 점수를 매긴 후보 리스트
    natural_output : 사용자에게 바로 보여줄 한국어 추천 설명
    """

    user_input: str
    user_id: int
    decision: Dict[str, Any]
    candidates: List[Dict[str, Any]]
    reranked: List[Dict[str, Any]]
    natural_output: str


# ============================================================
# 2. 전역 싱글톤 Recommender (lazy init)
# ============================================================

_content_rec: Optional[BookRecommender] = None
_cf_rec: Optional[CFRecommender] = None


def get_recommenders() -> tuple[BookRecommender, CFRecommender]:
    """
    BookRecommender / CFRecommender를 lazy init 후 반환.

    - import 시점에 무거운 작업이 돌지 않도록 하고,
      실제로 추천을 처음 호출할 때 한 번만 초기화되도록 설계.
    """
    global _content_rec, _cf_rec

    if _content_rec is None:
        logger.info("[Graph] Initializing BookRecommender (content-based)")
        _content_rec = BookRecommender()

    if _cf_rec is None:
        logger.info("[Graph] Initializing CFRecommender (item-based CF)")
        # content_rec에서 book_id universe를 가져와 CF에 넘겨줌
        valid_book_ids = set(_content_rec.df["book_id"].unique())

        _cf_rec = CFRecommender(
            min_ratings_per_user=5,
            min_ratings_per_item=5,
            max_items_for_similarity=None,
            use_als=False,  # ⚠ 현재는 ALS 비활성화, item-based CF만 사용
            valid_book_ids=valid_book_ids,
        )
        _cf_rec.load_data()
        _cf_rec.build_interaction_matrix()
        # item-based similarity 계산
        _cf_rec.compute_item_similarity()

    return _content_rec, _cf_rec


# ============================================================
# 3. 후보 merge 유틸
# ============================================================


def _normalize_scores_by_rank(
    items: List[Dict[str, Any]],
    score_key: str,
) -> None:
    """
    주어진 score_key 기준으로 items를 정렬한 뒤,
    '랭크 기반' 0~1 점수로 다시 매긴다.

    예:
        N개 아이템이 있을 때,
        1등 → 1.0
        2등 → (N-2)/(N-1)
        ...
        꼴등 → 0.0
    """
    if not items:
        return

    # score 기준으로 내림차순 정렬
    items.sort(key=lambda x: x.get(score_key, 0.0), reverse=True)
    n = len(items)
    if n == 1:
        items[0][score_key] = 1.0
        return

    for rank, item in enumerate(items):
        # rank: 0이 1등
        item[score_key] = float(n - 1 - rank) / float(n - 1)


def merge_candidates(
    content_candidates: List[Dict[str, Any]] | None,
    cf_candidates: List[Dict[str, Any]] | None,
    alpha: float,
) -> List[Dict[str, Any]]:
    """
    콘텐츠 기반 후보 + CF 후보를 book_id 기준으로 merge하여
    content_score / cf_score / hybrid_score를 계산한다.

    - alpha: content와 cf의 비율 (0.0 ~ 1.0)
        hybrid_score = alpha * content_score + (1 - alpha) * cf_score
    - content_score / cf_score는 모두 0~1로 정규화된 값이라고 가정하되,
      필요시 여기서 rank 기반으로 한 번 더 normalize.

    반환 값: book_id 기준으로 unique한 후보 리스트 (hybrid_score 기준 내림차순)
    """
    content_candidates = content_candidates or []
    cf_candidates = cf_candidates or []

    # 1) book_id → 후보 dict 병합
    merged: Dict[int, Dict[str, Any]] = {}

    # 콘텐츠 후보 먼저
    for c in content_candidates:
        bid = int(c["book_id"])
        merged[bid] = {
            "book_id": bid,
            "title": c.get("title"),
            "authors": c.get("authors"),
            "content_score": float(c.get("score", c.get("content_score", 0.0))),
            "cf_score": 0.0,
        }

    # CF 후보 overlay
    for c in cf_candidates:
        bid = int(c["book_id"])
        if bid not in merged:
            merged[bid] = {
                "book_id": bid,
                "title": c.get("title"),
                "authors": c.get("authors"),
                "content_score": 0.0,
                "cf_score": float(c.get("score", c.get("cf_score", 0.0))),
            }
        else:
            merged[bid]["cf_score"] = float(
                c.get("score", c.get("cf_score", merged[bid]["cf_score"]))
            )
            # title/authors가 비어 있으면 CF 쪽 정보로 채우기
            if not merged[bid].get("title"):
                merged[bid]["title"] = c.get("title")
            if not merged[bid].get("authors"):
                merged[bid]["authors"] = c.get("authors")

    merged_list = list(merged.values())

    # 2) rank 기반 정규화 (content_score / cf_score 각각)
    _normalize_scores_by_rank(merged_list, "content_score")
    _normalize_scores_by_rank(merged_list, "cf_score")

    # 3) hybrid_score 계산
    for item in merged_list:
        c_score = float(item.get("content_score", 0.0))
        cf_score = float(item.get("cf_score", 0.0))
        item["hybrid_score"] = alpha * c_score + (1.0 - alpha) * cf_score

    # 4) hybrid_score 기준 정렬
    merged_list.sort(key=lambda x: x.get("hybrid_score", 0.0), reverse=True)
    return merged_list


# ============================================================
# 4. LangGraph 노드 정의
# ============================================================


def parse_intent_node(state: BookState) -> BookState:
    """
    1단계: LLM으로 사용자 입력을 분석하여
    - strategy (by_title / by_mood / 등)
    - mentioned_titles
    - mood_keywords, genres 등
    을 추출한다.
    """
    user_input = state.get("user_input", "")
    logger.debug("[Graph] parse_intent_node - user_input=%s", user_input)

    try:
        decision = llm_decider.decide_strategy_with_llm(user_input)
        state["decision"] = decision
    except Exception as e:
        logger.exception("[Graph] LLM decider error: %s", e)
        # 실패 시 최소 구조라도 유지
        state["decision"] = {
            "strategy": "by_mood",
            "mentioned_titles": [],
            "mood_keywords": [],
            "genres": [],
            "extra_constraints": [],
        }

    # 디버그용 요약
    d = state["decision"]
    logger.info(
        "[LLM 분석 결과 요약]\n"
        "- strategy: %s\n"
        "- mentioned_titles: %s\n"
        "- mood_keywords: %s\n"
        "- genres: %s\n"
        "- extra_constraints: %s",
        d.get("strategy"),
        d.get("mentioned_titles"),
        d.get("mood_keywords"),
        d.get("genres"),
        d.get("extra_constraints"),
    )

    return state


def generate_candidates_node(state: BookState) -> BookState:
    """
    2단계: 전통 추천 시스템으로 후보 책 리스트를 생성.

    - 콘텐츠 기반: BookRecommender.recommend_from_llm_decision
    - CF 기반: CFRecommender.recommend_for_user
    - 둘 다 있으면 merge_candidates()로 hybrid_score 계산
    - 하나만 있으면 그쪽 후보만 사용
    """
    user_id = state.get("user_id")
    decision = state.get("decision", {})
    user_input = state.get("user_input", "")

    content_rec, cf_rec = get_recommenders()

    # 1) 콘텐츠 기반 후보
    content_candidates: List[Dict[str, Any]] = []
    try:
        content_candidates = content_rec.recommend_from_llm_decision(
            llm_decision=decision,
            user_input=user_input,  # 🔹 추가
            top_k=MAX_CANDIDATES_FOR_LLM,
        )

    except Exception as e:
        logger.exception("[Graph] content recommend_from_llm_decision error: %s", e)
        content_candidates = []

    # 2) CF 기반 후보
    cf_candidates: List[Dict[str, Any]] = []
    try:
        if user_id is not None:
            cf_candidates = cf_rec.recommend_for_user(
                user_id=user_id,
                top_k=MAX_CANDIDATES_FOR_LLM,
                # 온라인 추천에서는 이미 본 책은 웬만하면 제외
                filter_read_items=True,
            )
    except Exception as e:
        logger.exception("[Graph] CF recommend_for_user error: %s", e)
        cf_candidates = []

    # 3) merge 로직
    if content_candidates and cf_candidates:
        candidates = merge_candidates(
            content_candidates=content_candidates,
            cf_candidates=cf_candidates,
            alpha=HYBRID_ALPHA_CONTENT,
        )
    elif cf_candidates:
        # CF만 있을 때도 후속 단계에서 hybrid_score에 맞춰 사용 가능하도록 필드 맞추기
        candidates = []
        for c in cf_candidates:
            candidates.append(
                {
                    "book_id": int(c["book_id"]),
                    "title": c.get("title"),
                    "authors": c.get("authors"),
                    "content_score": 0.0,
                    "cf_score": float(c.get("score", c.get("cf_score", 0.0))),
                    "hybrid_score": float(c.get("score", c.get("cf_score", 0.0))),
                }
            )
    elif content_candidates:
        candidates = []
        for c in content_candidates:
            candidates.append(
                {
                    "book_id": int(c["book_id"]),
                    "title": c.get("title"),
                    "authors": c.get("authors"),
                    "content_score": float(c.get("score", c.get("content_score", 0.0))),
                    "cf_score": 0.0,
                    "hybrid_score": float(c.get("score", c.get("content_score", 0.0))),
                }
            )
    else:
        logger.warning("[Graph] No candidates from either content or CF.")
        candidates = []

    state["candidates"] = candidates
    return state


def rerank_with_llm_node(state: BookState) -> BookState:
    """
    3단계: LLM으로 후보들을 감정/무드/조건에 맞게 재정렬하고,
    자연어 추천 문장(natural_output)을 생성한다.
    """
    user_input = state.get("user_input", "")
    decision = state.get("decision", {})
    candidates = state.get("candidates", [])

    logger.debug(
        "[Graph] rerank_with_llm_node - #candidates=%d",
        len(candidates),
    )

    if not candidates:
        # 후보가 하나도 없으면 LLM 호출 대신 기본 메시지
        state["reranked"] = []
        state[
            "natural_output"
        ] = "지금은 추천할 수 있는 책 후보가 없습니다. 나중에 다시 시도해 주세요."
        return state

    try:
        result = llm_reranker.rerank_with_llm(
            user_input=user_input,
            llm_decision=decision,
            candidates=candidates,
        )
        reranked = result.get("reranked", [])
        natural_output = result.get("natural_output", "").strip()

        state["reranked"] = reranked or candidates
        if natural_output:
            state["natural_output"] = natural_output
        else:
            # natural_output이 비어 있으면 간단한 기본 설명 생성
            titles = [c.get("title") for c in state["reranked"][:3] if c.get("title")]
            if titles:
                state[
                    "natural_output"
                ] = f"지금 상황에 어울리는 책으로는 {', '.join(titles)} 등을 추천드릴 수 있습니다."
            else:
                state[
                    "natural_output"
                ] = "사용자님의 취향에 맞는 책 몇 권을 추천해 두었습니다."

    except Exception as e:
        logger.exception("[Graph] LLM reranker error: %s", e)
        # 실패 시: 후보는 그대로 두고, 간단한 fallback 문장 사용
        state["reranked"] = candidates
        state[
            "natural_output"
        ] = "시스템 내부 오류로 인해 단순 추천 순서로 책을 보여드립니다. 양해 부탁드립니다."

    return state


# ============================================================
# 5. 그래프 구성 + 헬퍼
# ============================================================


from langgraph.graph import StateGraph, END
# 필요하면 타입용으로만: from langgraph.graph import CompiledGraph  (버전에 따라 다를 수 있음)

def build_book_graph():
    """
    BookState를 사용하는 LangGraph 파이프라인을 구성하여
    compile()까지 마친 객체를 반환한다.
    """
    graph = StateGraph(BookState)

    graph.add_node("parse_intent", parse_intent_node)
    graph.add_node("generate_candidates", generate_candidates_node)
    graph.add_node("rerank_with_llm", rerank_with_llm_node)

    graph.set_entry_point("parse_intent")

    graph.add_edge("parse_intent", "generate_candidates")
    graph.add_edge("generate_candidates", "rerank_with_llm")
    graph.add_edge("rerank_with_llm", END)

    # 🔥 핵심: 여기서 compile() 호출
    app = graph.compile()
    return app


# 전역 그래프 싱글톤
_book_graph = build_book_graph()


def run_book_recommendation(user_input: str, user_id: int) -> BookState:
    """
    CLI / API 등 외부에서 사용하는 진입점.

    예:
        state = run_book_recommendation("지금 우울한데 위로되는 판타지 소설 추천", user_id=123)
        print(state["natural_output"])
    """
    initial_state: BookState = {
        "user_input": user_input,
        "user_id": user_id,
        "decision": {},
        "candidates": [],
        "reranked": [],
        "natural_output": "",
    }

    final_state: BookState = _book_graph.invoke(initial_state)
    return final_state

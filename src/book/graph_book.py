# src/book/graph_book.py

"""
LangGraph 기반 Book 추천 파이프라인 (단순화 버전).

구조 개요
---------
1) LLM Decider (llm_decider.decide_strategy_with_llm)
   - user_input을 받아서 현재 감정, 원하는 감정, 장르, 전략 등을 JSON으로 파싱.
   - 결과는 state["decision"]에 저장.

2) Candidate Generation (콘텐츠 기반 SBERT만 사용)
   - BookRecommender.recommend_from_llm_decision(llm_decision, top_k, user_input, exclude_book_ids)
   - user_profile의 seen_books를 이용해 "이미 읽은 책"은 제외.
   - 결과를 state["candidates"]에 저장.
   - CFRecommender는 초기 추천(run_chat_llm_demo.get_initial_recommendations)에서만 사용.

3) LLM Reranker (llm_reranker.rerank_with_llm)
   - 입력: user_input, llm_decision, candidates (+ user_top_genres; 나중에 필요 시 Phase 2에서 정리 가능)
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
import os

from langgraph.graph import StateGraph, END

from src.common.state_types import BaseRecState
from src.config import (
    MAX_CANDIDATES_FOR_LLM,
)
from .recommender import BookRecommender
from .cf_recommender import CFRecommender
from . import llm_decider
from . import llm_reranker
from src.book.user_profile import get_user_profile

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
        base_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        my_ratings_path = os.path.join(
            base_dir, "data", "goodbooks-10k", "my_ratings.csv"
        )
        _cf_rec = CFRecommender(
            min_ratings_per_user=1,
            min_ratings_per_item=1,
            max_items_for_similarity=None,
            valid_book_ids=valid_book_ids,
        )
        _cf_rec.load_data()
        _cf_rec.build_interaction_matrix()
        # item-based similarity 계산 (초기 추천 등에서 사용 가능)
        _cf_rec.compute_item_similarity()

    return _content_rec, _cf_rec


# ============================================================
# 3. LangGraph 노드 정의
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

    🔹 현재 버전 목표:
      - 대화형 추천에서는 **콘텐츠 기반(SBERT)** 만 사용
      - CF는 초기 추천(user_id만 있을 때)에서만 사용
    """
    user_id = state.get("user_id")
    decision = state.get("decision", {})
    user_input = state.get("user_input", "")

    # BookRecommender만 사용
    content_rec, _ = get_recommenders()

    # (당장은 단순화를 위해 seen_books / user_top_genres 안 씀)
    # 필요해지면 나중에 다시 붙이면 됨
    try:
        content_candidates = content_rec.recommend_from_llm_decision(
            llm_decision=decision,
            top_k=MAX_CANDIDATES_FOR_LLM,
            user_input=user_input,
        )
    except Exception as e:
        logger.exception("[Graph] content recommend_from_llm_decision error: %s", e)
        state["candidates"] = []
        return state

    # SBERT 점수를 그대로 hybrid_score에 복사
    candidates: List[Dict[str, Any]] = []
    for c in content_candidates:
        score = float(c.get("score", c.get("content_score", 0.0)))
        candidates.append(
            {
                "book_id": int(c["book_id"]),
                "title": c.get("title"),
                "authors": c.get("authors"),
                "content_score": score,
                "hybrid_score": score,  # 지금은 content-only
            }
        )

    candidates.sort(key=lambda x: x.get("hybrid_score", 0.0), reverse=True)
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
    user_id = state.get("user_id")

    logger.debug(
        "[Graph] rerank_with_llm_node - #candidates=%d",
        len(candidates),
    )

    if not candidates:
        state["reranked"] = []
        state[
            "natural_output"
        ] = "지금은 추천할 수 있는 책 후보가 없습니다. 나중에 다시 시도해 주세요."
        return state

    # user_profile에서 top_genres 다시 한 번 가져와서 reranker에 넘김
    user_top_genres: List[str] = []
    if user_id is not None:
        try:
            profile = get_user_profile(int(user_id))
            user_top_genres = profile.get("top_genres", []) or []
        except Exception as e:
            logger.exception("[Graph] get_user_profile error in rerank node: %s", e)
            user_top_genres = []

    try:
        result = llm_reranker.rerank_with_llm(
            user_input=user_input,
            llm_decision=decision,
            candidates=candidates,
            user_top_genres=user_top_genres,
        )
        reranked = result.get("reranked", [])
        natural_output = result.get("natural_output", "").strip()

        state["reranked"] = reranked or candidates
        if natural_output:
            state["natural_output"] = natural_output
        else:
            titles = [
                c.get("title") for c in state["reranked"][:3] if c.get("title")
            ]
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
        state["reranked"] = candidates
        state[
            "natural_output"
        ] = "시스템 내부 오류로 인해 단순 추천 순서로 책을 보여드립니다. 양해 부탁드립니다."

    # 🔹 여기서 공통으로 인사말 prefix 붙이기
    try:
        uid = state.get("user_id")
        first_title = None
        if state.get("reranked"):
            first_title = state["reranked"][0].get("title")

        if uid is not None and first_title:
            prefix = f"안녕하세요 {uid}님, 오늘은 『{first_title}』를 포함해 몇 권의 책을 추천드려요.\n\n"
        elif uid is not None:
            prefix = f"안녕하세요 {uid}님, 지금의 기분과 취향에 맞는 책들을 추천드려요.\n\n"
        else:
            prefix = ""

        state["natural_output"] = prefix + state.get("natural_output", "")
    except Exception as e:
        logger.exception("[Graph] greeting prefix error: %s", e)

    return state


# ============================================================
# 4. 그래프 구성 + 헬퍼
# ============================================================


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

# src/book/run_chat_llm_demo.py
from .graph_book import run_book_recommendation
import os
import logging
from src.config import (
    BOOK_TFIDF_MAX_FEATURES,
    HYBRID_ALPHA_CONTENT,
    LLM_MODEL_DECIDER,
    LLM_MODEL_RERANKER,
    MAX_CANDIDATES_FOR_LLM,
)
def setup_logging():
    debug = os.getenv("A2A_DEBUG", "0") == "1"
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    )

TEST_USER_ID = 314

def main():
    while True:
        try:
            user_input = input("지금 읽고 싶은 책/기분/취향을 자유롭게 적어보세요:\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n종료합니다.")
            break

        if not user_input:
            continue
        if user_input.lower() in ["quit", "exit", "q"]:
            print("종료합니다.")
            break

        # LangGraph 기반 파이프라인 실행
        state = run_book_recommendation(user_input, user_id=TEST_USER_ID)

        decision = state.get("decision", {})
        reranked = state.get("reranked", [])
        candidates = state.get("candidates", [])
        natural_output = state.get("natural_output", "").strip()

        print("\n[LLM 분석 결과 요약]")
        print(f"- strategy: {decision.get('strategy')}")
        print(f"- mentioned_titles: {decision.get('mentioned_titles')}")
        print(f"- mood_keywords: {decision.get('mood_keywords')}")
        print(f"- genres: {decision.get('genres')}")
        print(f"- extra_constraints: {decision.get('extra_constraints')}")

        # 1) 자연어 추천 결과 출력
        if natural_output:
            print("\n[자연어 추천 결과]")
            print(natural_output)

        # 2) 필요하면 디버그용으로 상위 5개도 같이 표시
        final_list = reranked if reranked else candidates
        if final_list:
            print("\n[디버그용: 최종 추천 리스트 Top-5]")
            for i, book in enumerate(final_list[:5], start=1):
                title = book.get("title", "(제목 없음)")
                authors = book.get("authors", "")
                hybrid_score = book.get("hybrid_score", 0.0)
                llm_score = book.get("llm_score", None)

                if llm_score is not None:
                    print(f"{i}. {title} / {authors} (hybrid {hybrid_score:.3f}, LLM {llm_score:.3f})")
                else:
                    print(f"{i}. {title} / {authors} (hybrid {hybrid_score:.3f})")

if __name__ == "__main__":
    main()

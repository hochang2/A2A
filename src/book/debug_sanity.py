# src/book/debug_sanity.py

import json

from .graph_book import run_book_recommendation

# CF가 같이 섞이게 하고 싶으면, 실제로 존재하는 user_id로 바꿔도 됩니다.
TEST_USER_ID = 312


TEST_QUERIES = [
    "심심한데 설레고 싶다",
    "호러",
    "잔잔한 에세이",
    "우울한데 위로받고 싶다",
    "밤새 몰입해서 읽을 판타지",
    "가볍게 읽을 수 있는 로맨스 소설",
    "현실적이고 씁쓸한 현대 한국 소설",
]


def pretty_print_top5(state: dict, top_k: int = 5) -> None:
    reranked = state.get("reranked", [])[:top_k]

    print("[TOP-5 후보]")
    if not reranked:
        print("  (후보가 없습니다.)")
        return

    for i, c in enumerate(reranked, start=1):
        title = c.get("title")
        authors = c.get("authors")

        hybrid = float(c.get("hybrid_score", 0.0) or 0.0)
        llm_score = float(c.get("llm_score", 0.0) or 0.0)
        final_score = float(c.get("final_score", 0.0) or 0.0)

        print(
            f"{i}. {title} / {authors} "
            f"(hybrid={hybrid:.3f}, llm={llm_score:.3f}, final={final_score:.3f})"
        )


def run_sanity_check() -> None:
    for q in TEST_QUERIES:
        print("=" * 80)
        print(f"[INPUT]\n{q}\n")

        state = run_book_recommendation(user_input=q, user_id=TEST_USER_ID)

        decision = state.get("decision", {})
        print("[LLM_DECISION]")
        print(json.dumps(decision, ensure_ascii=False, indent=2))
        print()

        print("[NATURAL_OUTPUT]")
        natural = (state.get("natural_output") or "").strip()
        print(natural if natural else "(natural_output 없음)")
        print()

        pretty_print_top5(state, top_k=5)
        print()  # 빈 줄


if __name__ == "__main__":
    run_sanity_check()

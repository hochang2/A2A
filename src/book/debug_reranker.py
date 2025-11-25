# src/book/debug_reranker.py

import json

from src.book.llm_decider import decide_strategy_with_llm
from src.book import llm_reranker


def make_dummy_candidates():
    """
    LLM이 제목만 보고도 분위기를 대충 추론할 수 있게
    일부러 장르/무드가 드러나는 제목으로 구성한 샘플 후보들.
    hybrid_score는 기본 순위(전통 모델 결과)라고 생각하면 됨.
    """
    return [
        {
            "book_id": 1,
            "title": "Warm Romance 1",
            "authors": "Author A",
            "content_score": 0.9,
            "cf_score": 0.8,
            "hybrid_score": 0.95,
        },
        {
            "book_id": 2,
            "title": "Dark Thriller Night",
            "authors": "Author B",
            "content_score": 0.88,
            "cf_score": 0.82,
            "hybrid_score": 0.92,
        },
        {
            "book_id": 3,
            "title": "Calm Night Essay",
            "authors": "Author C",
            "content_score": 0.85,
            "cf_score": 0.78,
            "hybrid_score": 0.90,
        },
        {
            "book_id": 4,
            "title": "Epic Fantasy War",
            "authors": "Author D",
            "content_score": 0.87,
            "cf_score": 0.76,
            "hybrid_score": 0.88,
        },
        {
            "book_id": 5,
            "title": "Light Romance 2",
            "authors": "Author E",
            "content_score": 0.84,
            "cf_score": 0.79,
            "hybrid_score": 0.87,
        },
    ]


def print_candidates(title: str, cands):
    print(f"[{title}]")
    for c in cands:
        print(
            f"- {c.get('title')} "
            f"(id={c.get('book_id')}, "
            f"hybrid={c.get('hybrid_score', 0):.3f}, "
            f"llm={c.get('llm_score', 0):.3f}, "
            f"final={c.get('final_score', 0):.3f})"
        )
    print()


def run_one_version(label: str, rank_prompt: str, summary_prompt: str, user_input: str, llm_decision, base_candidates):
    """
    주어진 프롬프트 세트로 rerank_with_llm 한 번 돌리고 결과 출력.
    """
    # 프롬프트 교체
    llm_reranker.RANK_SYSTEM_PROMPT = rank_prompt
    llm_reranker.SUMMARY_SYSTEM_PROMPT = summary_prompt

    # 깊은 복사 대신, 간단히 새 리스트 만들어서 원본 안 건드리기
    candidates = [dict(c) for c in base_candidates]

    result = llm_reranker.rerank_with_llm(
        user_input=user_input,
        llm_decision=llm_decision,
        candidates=candidates,
        top_k=5,
    )

    reranked = result.get("reranked", [])
    natural_output = result.get("natural_output", "")

    print("=" * 80)
    print(f"[RESULT - RERANKER {label}]")
    print_candidates(f"Reranked Top-5 ({label})", reranked)
    print(f"[SUMMARY {label}]\n{natural_output}")
    print()


def main():
    # 1) 사용자 입력 받기 (테스트용)
    print("지금 읽고 싶은 책/기분/취향을 자유롭게 적어보세요:")
    user_input = input("> ").strip()

    # 2) decider 돌려서 llm_decision 얻기 (실제 파이프라인과 동일)
    llm_decision = decide_strategy_with_llm(user_input=user_input, debug=True)

    print("\n[LLM_DECISION]")
    print(json.dumps(llm_decision, ensure_ascii=False, indent=2))
    print()

    # 3) 샘플 후보 생성 (hybrid_score 기준으로 이미 정렬되어 있다고 가정)
    base_candidates = make_dummy_candidates()

    print("[BASE CANDIDATES (hybrid_score 기준)]")
    print_candidates("Base", base_candidates)

    # 4) V1 프롬프트로 테스트
    run_one_version(
        label="V1",
        rank_prompt=llm_reranker.RANK_SYSTEM_PROMPT_V1,
        summary_prompt=llm_reranker.SUMMARY_SYSTEM_PROMPT_V1,
        user_input=user_input,
        llm_decision=llm_decision,
        base_candidates=base_candidates,
    )

    # 5) V2 프롬프트로 테스트
    run_one_version(
        label="V2",
        rank_prompt=llm_reranker.RANK_SYSTEM_PROMPT_V2,
        summary_prompt=llm_reranker.SUMMARY_SYSTEM_PROMPT_V2,
        user_input=user_input,
        llm_decision=llm_decision,
        base_candidates=base_candidates,
    )

    print("=" * 80)
    print("V1과 V2의 순위/설명을 비교해서 더 마음에 드는 쪽을 선택하세요.")
    print("선택 후에는 llm_reranker.py에서 RANK_SYSTEM_PROMPT / SUMMARY_SYSTEM_PROMPT를")
    print("원하는 버전(V1 또는 V2)로 지정하면 됩니다.")
    print("=" * 80)


if __name__ == "__main__":
    main()

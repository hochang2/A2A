# src/book/run_chat_llm_demo.py

from .graph_book import run_book_recommendation, get_recommenders
import os
import logging
import csv
from typing import List, Dict, Any
from . import llm_reranker   # 👈 요거 추가


import numpy as np

from src.config import (
    BOOK_TFIDF_MAX_FEATURES,
    HYBRID_ALPHA_CONTENT,
    LLM_MODEL_DECIDER,
    LLM_MODEL_RERANKER,
    MAX_CANDIDATES_FOR_LLM,
)

# 프로젝트 루트 기준 경로 설정
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

# 🔹 my_ratings.csv 경로 (CF/개인화용 rating 로그)
MY_RATINGS_PATH = os.path.join(
    BASE_DIR, "data", "goodbooks-10k", "my_ratings.csv"
)


def setup_logging():
    debug = os.getenv("A2A_DEBUG", "0") == "1"
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    )


def append_rating(user_id: int, book_id: int, rating: float = 1.0) -> None:
    """
    my_ratings.csv에 (user_id, book_id, rating)을 한 줄 append.
    파일이 없으면 헤더를 포함해 새로 생성.
    """
    file_exists = os.path.exists(MY_RATINGS_PATH)
    os.makedirs(os.path.dirname(MY_RATINGS_PATH), exist_ok=True)

    with open(MY_RATINGS_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["user_id", "book_id", "rating"])
        writer.writerow([int(user_id), int(book_id), float(rating)])


# ============================================================
#  ⭐ 초기 추천 (user_input 없이, ratings.csv 기반 CF)
# ============================================================

def get_initial_recommendations(user_id: int, top_n: int = 5) -> List[Dict[str, Any]]:
    """
    초기 추천 로직:

    1) CFRecommender 기반 개인화 추천 시도
       - ratings.csv (+ my_ratings.csv)에 있는 user_id의 평점을 기반으로
         비슷한 책을 item-based CF로 추천.
       - 즉, "좋은 평점을 남긴 책 기준으로 비슷한 책 추천"에 해당.

    2) CF 추천 결과가 없으면 (완전 cold-start):
       - ratings_df에서 book_id별로
           * count(평점 수)
           * mean(평균 평점)
         을 구한 뒤,
           score = mean * log1p(count)
         순으로 상위 책을 골라서 추천.
       - 이 역시 ratings.csv 기반이므로,
         전역적으로 "평점 좋은 책"을 초기 추천으로 사용하게 됨.
    """
    logger = logging.getLogger(__name__)

    try:
        content_rec, cf_rec = get_recommenders()
    except Exception as e:
        logger.exception("[InitialRec] get_recommenders() 실패: %s", e)
        return []

    # --------------------------------------------------------
    # 1) CF 기반 개인화 추천 시도
    # --------------------------------------------------------
    cf_candidates: List[Dict[str, Any]] = []
    try:
        # top_n * 4 정도 넉넉하게 가져온 뒤 상위 top_n만 사용
        cf_candidates = cf_rec.recommend_for_user(
            user_id=user_id,
            top_k=top_n * 4,
            filter_read_items=True,
        )
    except Exception as e:
        logger.exception("[InitialRec] CF recommend_for_user 실패: %s", e)
        cf_candidates = []

    if cf_candidates:
        # book 메타데이터 조인
        df_books = content_rec.df.set_index("book_id")

        enriched: List[Dict[str, Any]] = []
        for c in cf_candidates:
            bid = int(c["book_id"])
            score = float(c.get("score", 0.0))

            title = None
            authors = None
            if bid in df_books.index:
                row = df_books.loc[bid]
                title = row.get("title")
                authors = row.get("authors")

            enriched.append(
                {
                    "book_id": bid,
                    "title": title,
                    "authors": authors,
                    "cf_score": score,
                }
            )

        enriched.sort(key=lambda x: x.get("cf_score", 0.0), reverse=True)
        return enriched[:top_n]

    # --------------------------------------------------------
    # 2) CF 결과가 없으면 → ratings.csv 기반 인기 + 평점 fallback
    # --------------------------------------------------------
    logger.info(
        "[InitialRec] CF 개인화 추천 결과 없음 (user_id=%s). ratings.csv 기반 기본 추천으로 대체합니다.",
        user_id,
    )

    ratings_df = getattr(cf_rec, "ratings_df", None)
    if ratings_df is None or ratings_df.empty:
        logger.warning("[InitialRec] ratings_df가 비어 있습니다. 초기 추천 불가.")
        return []

    # book_id별 평점 수 / 평균 평점
    stats = (
        ratings_df.groupby("book_id")["rating"]
        .agg(["count", "mean"])
        .reset_index()
    )

    # BookRecommender가 알고 있는 book universe로 제한
    valid_book_ids = set(content_rec.df["book_id"].unique())
    stats = stats[stats["book_id"].isin(valid_book_ids)]

    if stats.empty:
        logger.warning("[InitialRec] 유효한 book_id가 없습니다. 초기 추천 불가.")
        return []

    # score = mean_rating * log1p(count)  (많이 읽히면서 평점도 높은 책 우선)
    stats["score"] = stats["mean"] * np.log1p(stats["count"])
    stats = stats.sort_values("score", ascending=False).head(top_n)

    df_books = content_rec.df.set_index("book_id")

    results: List[Dict[str, Any]] = []
    for _, row in stats.iterrows():
        bid = int(row["book_id"])
        base_score = float(row["score"])

        title = None
        authors = None
        if bid in df_books.index:
            meta = df_books.loc[bid]
            title = meta.get("title")
            authors = meta.get("authors")

        results.append(
            {
                "book_id": bid,
                "title": title,
                "authors": authors,
                "cf_score": base_score,
            }
        )

    return results[:top_n]


# ============================================================
#  메인 대화 루프
# ============================================================

def main():
    setup_logging()

    # 🔹 유저 ID를 한 번 입력받아서 세션 동안 유지
    while True:
        try:
            user_id_str = input("당신의 user_id를 입력하세요 (예: 100001) 또는 'q'로 종료:\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n종료합니다.")
            return

        if not user_id_str:
            continue
        if user_id_str.lower() in ["q", "quit", "exit"]:
            print("종료합니다.")
            return

        try:
            user_id = int(user_id_str)
            break
        except ValueError:
            print("정수 user_id를 입력해주세요.")
            continue

    print(f"\nuser_id={user_id} 로 추천을 시작합니다.\n")

    # 🔹 여기서 한 번만 쓸 "초기 자연어 입력" 버퍼
    pending_input: Optional[str] = None

    # =======================================================
    #  🔸 STEP 0: 초기 추천 (user_input 없이, ratings 기반)
    # =======================================================
    initial_list = get_initial_recommendations(user_id=user_id, top_n=5)

    if initial_list:
        print(f"[{user_id}님을 위한 기본 추천]\n")
        for i, book in enumerate(initial_list, start=1):
            title = book.get("title") or "(제목 없음)"
            authors = book.get("authors") or ""
            book_id = book.get("book_id")
            print(f"{i}. (book_id={book_id}) {title} / {authors}")

        # 🔹 초기 추천에 대한 자연어 요약 설명
        try:
            initial_summary = llm_reranker.generate_summary_for_candidates(
                user_input="[INITIAL_RECOMMENDATION]",
                llm_decision={},          # 초기에는 별도 decider 없음
                candidates=initial_list,  # CF 기반 Top-5
            )
            if initial_summary:
                print("\n[추천 요약]")
                print(f"안녕하세요 {user_id}님, 오늘은 이런 책들을 추천드려요.\n")
                print(initial_summary)
        except Exception as e:
            logging.getLogger(__name__).exception(
                "[InitialRec] 요약 생성 중 오류: %s", e
            )

        print(
            "\n이 중 마음에 드는 책 번호를 입력하시거나,\n"
            "다른 종류의 책을 원하시면 지금 기분/상황/장르를 자유롭게 적어주세요.\n"
            '- 예) "1,3"  또는  "요즘 우울한데 위로되는 판타지 소설"\n'
        )

        feedback = input("> ").strip()
        chosen_indices: List[int] = []

        if feedback:
            # 🔹 1) 먼저 '정수 리스트'인지 판단
            tokens = [t.strip() for t in feedback.split(",") if t.strip()]
            all_int = True
            int_indices: List[int] = []
            for t in tokens:
                if t.isdigit():
                    int_indices.append(int(t))
                else:
                    all_int = False
                    break

            if all_int and int_indices:
                # ✅ 숫자 입력 → rating 로그만 남김 (기존 로직 유지)
                for idx in int_indices:
                    if 1 <= idx <= len(initial_list):
                        book = initial_list[idx - 1]
                        book_id = int(book["book_id"])
                        append_rating(user_id=user_id, book_id=book_id, rating=1.0)
                        chosen_indices.append(idx - 1)
                        print(f"  → [초기추천] user_id={user_id}, book_id={book_id} 로그 저장 완료.")
                    else:
                        print(f"  - 무시: {idx} (유효 범위 밖)")
                # (지금은 user_events.csv를 안 쓰므로 별도 로그 없음)

            else:
                # ✅ 숫자가 아니면 → 자연어 질의로 간주해서
                #    바로 다음 단계에서 첫 질문으로 사용
                pending_input = feedback
                print(f'\n[안내] "{feedback}" 내용을 바탕으로 감정/취향 기반 추천을 이어서 진행합니다.\n')

    else:
        print(
            "[알림] 아직 user profile 기반으로 추천할 책을 찾지 못했습니다.\n"
            "지금 기분/상황/원하는 장르를 한 줄로 적어주시면 거기 맞춰 추천해 드릴게요.\n"
        )
        # 이 경우에는 pending_input 없음 (그냥 아래 루프에서 입력 받음)


    # =======================================================
    #  🔁 STEP 1: 감정/기분 기반 대화형 추천 루프
    # =======================================================
    while True:
        # 🔹 pending_input 이 있으면, 그걸 첫 질문으로 사용
        if pending_input is not None:
            user_input = pending_input.strip()
            pending_input = None
            print("지금 읽고 싶은 책/기분/취향을 자유롭게 적어보세요 (q: 종료):")
            print(f"> {user_input}")
        else:
            try:
                user_input = input("지금 읽고 싶은 책/기분/취향을 자유롭게 적어보세요 (q: 종료):\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n종료합니다.")
                break

            if not user_input:
                continue
            if user_input.lower() in ["quit", "exit", "q"]:
                print("종료합니다.")
                break

        # LangGraph 기반 파이프라인 실행
        state = run_book_recommendation(user_input, user_id=user_id)

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

        # 2) 디버그용 리스트 (Top-N)
        final_list = reranked if reranked else candidates

        if final_list:
            TOP_N = 5
            top_list = final_list[:TOP_N]

            print("\n[최종 추천 리스트 Top-5]")
            for i, book in enumerate(top_list, start=1):
                title = book.get("title", "(제목 없음)")
                authors = book.get("authors", "")
                book_id = book.get("book_id")
                hybrid_score = book.get("hybrid_score", 0.0)
                llm_score = book.get("llm_score", None)

                if llm_score is not None:
                    print(
                        f"{i}. (book_id={book_id}) {title} / {authors} "
                        f"(hybrid {hybrid_score:.3f}, LLM {llm_score:.3f})"
                    )
                else:
                    print(
                        f"{i}. (book_id={book_id}) {title} / {authors} "
                        f"(hybrid {hybrid_score:.3f})"
                    )

            # 🔹 유저 피드백 입력
            feedback = input(
                "\n마음에 드는 책 번호를 입력하세요 (여러 개면 쉼표로 구분, 없으면 엔터):\n> "
            ).strip()

            if feedback:
                try:
                    raw_indices = [
                        int(x.strip())
                        for x in feedback.split(",")
                        if x.strip()
                    ]
                    for idx in raw_indices:
                        # 유저 입력은 1-based, 내부 인덱스는 0-based
                        if 1 <= idx <= len(top_list):
                            book = top_list[idx - 1]
                            book_id = int(book["book_id"])
                            append_rating(user_id=user_id, book_id=book_id, rating=1.0)
                            print(f"  → user_id={user_id}, book_id={book_id} 로그 저장 완료.")
                        else:
                            print(f"  - 무시: {idx} (유효 범위 밖)")
                except ValueError:
                    print("번호를 정수로 입력해주세요. rating 로그는 저장되지 않았습니다.")

        # final_list가 비어 있으면 (후보 없음)
        else:
            print("\n[알림] 이번에는 추천 후보를 찾지 못했습니다.")


if __name__ == "__main__":
    main()

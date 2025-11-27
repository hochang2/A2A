# src/book/llm_reranker.py
import os
import json
import logging
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from src.common.llm_utils import _extract_text_from_response, _strip_markdown_fence
from src.config import (
    LLM_MODEL_RERANKER,
    MAX_CANDIDATES_FOR_LLM,
)

# .env 로드
load_dotenv()

logger = logging.getLogger(__name__)

ENV_DEBUG = os.getenv("A2A_DEBUG", "0") == "1"

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# 전통 모델 vs LLM 재랭킹 비중 (hybrid_score vs llm_score)
# 현재 hybrid_score는 사실상 "콘텐츠 기반 점수(content_score)"라고 보면 됨.
RERANKER_ALPHA_HYBRID = 0.5


# =========================
#  프롬프트 (단일 버전)
# =========================

RANK_SYSTEM_PROMPT = """
당신은 감정 기반 책 추천 리스트를 정리하고,
'기본 순위(hybrid_score)'를 사용자의 감정/상황에 맞게 살짝 조정해 주는 랭킹 전용 어시스턴트입니다.

입력으로는 다음 정보가 주어집니다:
- user_input: 사용자의 자연어 입력
- llm_decision: LLM이 분석한 전략/무드/장르/감정 정보
  - current_emotion: 사용자의 현재 감정 (영어 키워드 리스트)
  - desired_feeling: 사용자가 책을 통해 도달하고 싶은 감정 (영어 키워드 리스트)
  - content_mood: 책의 분위기 (영어 키워드 리스트)
  - genres / genres_en: 사용자가 선호/요청한 장르 (한국어 / 영어)
- user_top_genres: 사용자의 과거 읽기/담아두기 기록을 기반으로 계산한 "장기 취향 상위 장르" 리스트 (영어 소문자)
- candidates: 이미 전통 추천 시스템이 뽑아 놓은 상위 후보 리스트
  - 각 책에 대해 다음 정보가 주어집니다:
    - book_id, title, authors
    - hybrid_score: 전통 추천 모델(콘텐츠 기반 등)이 미리 계산한 기본 점수 (0~1)
    - genres_text: 이 책의 장르/서브장르를 나타내는 텍스트 (예: "fantasy young-adult adventure romance")
      (books.csv + book_genres.json을 합친 정보)
    - tags_text: Goodreads tag 기반 상위 태그 텍스트 (예: "magic dragons epic-fantasy high-fantasy")

이 리스트는 이미 hybrid_score 기준으로 내림차순 정렬되어 있습니다.
(위에 있을수록 기본적으로 더 추천에 적합한 책입니다.)

당신의 역할은 다음과 같습니다:
1) candidates 중에서 사용자 상황/감정/제약에 잘 맞는 책들을 골라,
   상위 5~10권 정도를 선택하고 적당한 순서로 나열합니다.
2) hybrid_score는 "기본 순위"이므로, 이를 완전히 무시하지 말고
   상위권 안에서 순서를 약간 조정하는 용도로 사용하세요.
3) score는 단순히 "최종 추천 강도"를 0.0~1.0 사이로 표현하는 값입니다.
   정확한 수학적 계산이 아니라, 상대적인 선호도(상위권일수록 1에 가깝게)를
   부드럽게 매긴다고 생각하면 됩니다.

장르/태그 활용 정책:
- llm_decision.genres_en이 비어 있지 않다면,
  → candidates의 genres_text, tags_text 안에 이 장르/키워드가 포함된 책을 우선적으로 상위에 올립니다.
- user_top_genres가 주어졌다면,
  → 사용자의 장기 취향(user_top_genres)과 현재 요청 장르(genres_en)가 둘 다 잘 맞는 책에 가산점을 줍니다.
  → 예: user_top_genres=["fantasy","young-adult"], genres_en=["romance"]
       이고, 어떤 책의 genres_text가 "fantasy young-adult romance"라면 상위로 올릴 가치가 큽니다.
- tags_text에는 분위기/주제(예: "dark", "cozy", "humor", "horror", "sad", "uplifting" 등)를 나타내는 단어가 포함될 수 있습니다.
  → current_emotion, desired_feeling, content_mood와 잘 맞는 태그가 있으면 점수를 올리고,
    완전히 반대되는 태그만 가득하면 점수를 약간 낮춥니다.

감정/무드 관련 정책(예시):

- current_emotion=["sad"], desired_feeling=["comforted", "cheered_up"]
  → 전쟁, 학살, 극단적인 비극, 과도하게 잔혹한 공포/스릴러는 점수를 낮추고,
    따뜻한 인간관계, 성장, 치유, 희망적인 결말을 가진 책에 점수를 더 줍니다.

- current_emotion=["afraid", "anxious"], desired_feeling=["comforted"]
  → 공포/호러, 강한 스릴러는 상위권에서 피하고,
    안정감·안전·회복을 주는 내용에 점수를 더 줍니다.

- desired_feeling=["deep_dive"]
  → 사용자가 일부러 어두운 감정을 파고들고 싶어하는 경우이므로,
    어두운 분위기/우울한 내용도 허용되지만,
    지나치게 자극적이거나 트라우마를 유발할 수 있는 내용은 점수를 조절합니다.

순위 조정 정책:

- desired_feeling 이나 content_mood 가 빈 리스트([])인 경우,
  → 특별한 감정/분위기 제약이 거의 없는 요청입니다.
  → 이때는 hybrid_score의 순서를 크게 바꾸지 말고,
     상위 10권을 거의 그대로 사용하되, 약간만 순서를 조정할 수 있습니다.

- desired_feeling 이나 content_mood 에 구체적인 값이 있는 경우,
  → 이 정보를 활용해 상위 후보들 중에서 더 잘 맞는 책을 위쪽으로 올리되,
     hybrid_score 기준 Top-3 중에서 최소 1권은 반드시 포함하세요.
  → hybrid_score 순위를 절대 완전히 뒤집지 말고,
     hybrid_score 기준 상위 15권 안에서만 재정렬하세요.

주의:
- 제목(title)에 포함된 단어(예: "Romance", "Thriller", "Horror", "Essay" 등)를
  분위기/장르를 추론하는 단서로 활용해도 됩니다.
- genres_text, tags_text 안의 단어들도 분위기/장르/주제를 추론하는 핵심 단서입니다.
- 너무 많은 책을 새로 추가하거나 빼려고 하지 말고,
  이미 주어진 후보 안에서 "선택+순서 조정"에 집중하세요.

출력 형식은 반드시 다음 JSON 하나만 반환하세요:

{
  "reranked": [
    {"book_id": 123, "score": 0.93},
    {"book_id": 456, "score": 0.87}
  ]
}

조건:
- reranked 리스트는 book_id 기준으로 중복 없이 상위 10개까지만 포함.
- score는 0.0 이상 1.0 이하이며, 상대적인 선호 강도만 표현하면 됩니다.
- hybrid_score 기준 상위 15권 중에서만 최종 후보를 선택하세요.
- hybrid_score 기준 Top-3 중에서 최소 1권은 반드시 포함하세요.
- JSON 이외의 텍스트는 출력하지 마세요.
"""

SUMMARY_SYSTEM_PROMPT = """
당신은 감정 기반 책 추천 시스템의 '설명 전용 어시스턴트'입니다.

입력:
- user_input: 사용자의 자연어 입력
- llm_decision: LLM이 분석한 전략/무드/장르/감정 정보
- selected_candidates: 최종적으로 선택된 책 리스트
  (book_id, title, authors, genres_text, tags_text, content_score, hybrid_score, llm_score, final_score 포함 가능)

역할:
- selected_candidates를 바탕으로, 사용자에게 보여줄 한국어 추천 문장을 작성합니다.
- 점수(숫자)는 언급하지 말고, 각 책의 분위기·내용·장르·태그·읽기 적합한 상황을 중심으로 설명합니다.
- 사용자의 현재 감정(current_emotion)과 desired_feeling, content_mood를 자연스럽게 반영해,
  "왜 이 책이 지금의 당신에게 어울리는지"를 중심으로 서술합니다.
- 가능한 경우, genres_text와 tags_text에 담긴 정보(예: romance, fantasy, horror, humor, cozy, dark 등)를 활용해
  책의 분위기와 특징을 설명하세요.
- 문장은 2~5문단 정도의 자연스러운 한국어로 작성하세요.

출력 형식:
- JSON이 아니라, 순수한 한국어 문장만 출력하세요.
- 코드 블록, 따옴표, 불필요한 메타 설명은 넣지 마세요.
- 없는 정보를 만들어내지 말고, 입력에 주어진 정보만 활용하세요.
"""


def _call_llm_for_ranking(
    user_input: str,
    llm_decision: Dict[str, Any],
    candidates_block: str,
    user_top_genres: Optional[List[str]] = None,
) -> Dict[int, float]:
    """
    랭킹 전용 LLM 호출: JSON만 받아서 {book_id: score} 맵으로 반환.
    """
    user_content = f"""
[사용자 입력]
{user_input}

[LLM 의사결정 (전략/무드/장르/감정)]
{json.dumps(llm_decision, ensure_ascii=False, indent=2)}

[사용자의 장기 취향 상위 장르 (user_top_genres)]
{json.dumps(user_top_genres or [], ensure_ascii=False, indent=2)}

[후보 리스트]
{candidates_block}

위 정보를 바탕으로, 아래 JSON 형식만 출력하세요:

{{
  "reranked": [
    {{"book_id": 123, "score": 0.93}}
  ]
}}
"""

    resp = client.responses.create(
        model=LLM_MODEL_RERANKER,
        max_output_tokens=400,
        temperature=0.2,
        input=[
            {"role": "system", "content": RANK_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )
    raw = _extract_text_from_response(resp)
    cleaned = _strip_markdown_fence(raw)

    if ENV_DEBUG:
        logger.debug("[LLM Reranker Rank Raw]\n%s", raw)
        logger.debug("[LLM Reranker Rank Cleaned]\n%s", cleaned)

    data = json.loads(cleaned)
    reranked = data.get("reranked", [])

    score_map: Dict[int, float] = {}
    for item in reranked:
        try:
            bid = int(item["book_id"])
            score = float(item["score"])
            score_map[bid] = score
        except Exception:
            continue

    return score_map


def generate_summary_for_candidates(
    user_input: str,
    llm_decision: Dict[str, Any],
    candidates: List[Dict[str, Any]],
) -> str:
    """
    외부에서 임의의 후보 리스트에 대해 한국어 자연어 설명을 만들 때 쓰는 helper.
    - run_book_recommendation 외부(예: 초기 CF 추천)에서도 재사용 가능.
    """
    return _call_llm_for_summary(
        user_input=user_input,
        llm_decision=llm_decision,
        final_candidates=candidates,
    )


def _call_llm_for_summary(
    user_input: str,
    llm_decision: Dict[str, Any],
    final_candidates: List[Dict[str, Any]],
) -> str:
    """
    설명 전용 LLM 호출: 자연어 한국어 문장만 반환.
    """
    lines = []
    for c in final_candidates:
        lines.append(
            f"- book_id={c.get('book_id')}, "
            f"title={c.get('title')}, "
            f"authors={c.get('authors')}, "
            f"genres_text={c.get('genres_text', '')}, "
            f"tags_text={c.get('tags_text', '')}"
        )
    candidates_block = "\n".join(lines)

    user_content = f"""
[사용자 입력]
{user_input}

[LLM 의사결정 (전략/무드/장르/감정)]
{json.dumps(llm_decision, ensure_ascii=False, indent=2)}

[최종 선택된 후보 리스트]
{candidates_block}

위 정보를 바탕으로, 사용자에게 보여줄 자연스러운 한국어 추천 문장을 작성하세요.
JSON이 아니라, 순수한 한국어 문장만 출력하세요.
"""

    resp = client.responses.create(
        model=LLM_MODEL_RERANKER,
        max_output_tokens=700,
        temperature=0.5,
        input=[
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )
    raw = _extract_text_from_response(resp)
    cleaned = _strip_markdown_fence(raw)

    if ENV_DEBUG:
        logger.debug("[LLM Reranker Summary Raw]\n%s", raw)
        logger.debug("[LLM Reranker Summary Cleaned]\n%s", cleaned)

    return cleaned.strip()


def rerank_with_llm(
    user_input: str,
    llm_decision: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    user_top_genres: Optional[List[str]] = None,
    top_k: int = 10,
) -> Dict[str, Any]:
    """
    1) 랭킹 전용 LLM 호출 → llm_score 부여
    2) hybrid_score와 llm_score를 합쳐 final_score 계산
    3) 최종 후보 top_k를 선택
    4) 설명 전용 LLM 호출 → natural_output 생성

    반환 형식
    --------
    {
      "reranked": [ ... 최종 후보 리스트 ... ],
      "natural_output": "사용자에게 보여줄 한국어 추천 문장"
    }
    """
    if not candidates:
        return {
            "reranked": [],
            "natural_output": "지금은 추천할 수 있는 책 후보가 없습니다. 나중에 다시 시도해 주세요.",
        }

    # 🔹 LLM에 넘길 후보는 상위 N개만 사용 (이미 hybrid_score로 정렬되어 있다고 가정)
    limited_candidates = candidates[:MAX_CANDIDATES_FOR_LLM]

    # 후보 리스트 직렬화 (LLM 랭킹용: hybrid_score + genres_text + tags_text)
    lines = []
    for c in limited_candidates:
        genres_text = c.get("genres_text", "")
        tags_text = c.get("tags_text", "")
        lines.append(
            f"- book_id={c.get('book_id')}, "
            f"title={c.get('title')}, "
            f"authors={c.get('authors')}, "
            f"hybrid_score={c.get('hybrid_score', 0.0):.3f}, "
            f"genres_text={genres_text}, "
            f"tags_text={tags_text}"
        )
    candidates_block = "\n".join(lines)

    # 내부 정규화 함수
    def _norm(scores: List[float]) -> List[float]:
        if not scores:
            return []
        s_min = min(scores)
        s_max = max(scores)
        if s_max == s_min:
            return [1.0 for _ in scores]
        return [(s - s_min) / (s_max - s_min) for s in scores]

    try:
        # 1) 랭킹 전용 LLM 호출 → book_id별 score_map
        score_map = _call_llm_for_ranking(
            user_input=user_input,
            llm_decision=llm_decision,
            candidates_block=candidates_block,
            user_top_genres=user_top_genres,
        )

        # 2) 각 후보에 llm_score 부여
        for c in limited_candidates:
            bid = int(c.get("book_id"))
            c["llm_score"] = float(score_map.get(bid, 0.0))

        # 3) hybrid_score 정규화 후 final_score 계산
        hybrid_list = [float(c.get("hybrid_score", 0.0)) for c in limited_candidates]
        hybrid_norm = _norm(hybrid_list)

        for c, h_norm in zip(limited_candidates, hybrid_norm):
            llm_score = float(c.get("llm_score", 0.0))
            c["final_score"] = (
                RERANKER_ALPHA_HYBRID * h_norm + (1.0 - RERANKER_ALPHA_HYBRID) * llm_score
            )

        # 4) final_score 기준 정렬 후 top_k 선택
        limited_candidates.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
        final_candidates = limited_candidates[:top_k]

        # 5) 설명 전용 LLM 호출
        natural_output = _call_llm_for_summary(
            user_input=user_input,
            llm_decision=llm_decision,
            final_candidates=final_candidates,
        )

        return {
            "reranked": final_candidates,
            "natural_output": natural_output,
        }

    except Exception as e:
        logger.exception("[LLM Reranker 오류] fallback으로 대체합니다: %s", e)
        # fallback: hybrid_score 기준 정렬 + 간단한 문장 (전체 candidates에서 top_k)
        candidates.sort(key=lambda x: x.get("hybrid_score", 0.0), reverse=True)
        final_candidates = candidates[:top_k]
        fallback_text = (
            "추천 시스템 내부 오류로, 기본 점수(hybrid_score)를 기준으로 책을 추천드립니다."
        )
        return {
            "reranked": final_candidates,
            "natural_output": fallback_text,
        }

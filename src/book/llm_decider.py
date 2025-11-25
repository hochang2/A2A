# src/book/llm_decider.py

import os
import json
import logging
from typing import Dict, Any
from src.common.llm_utils import _extract_text_from_response, _strip_markdown_fence


from openai import OpenAI
from dotenv import load_dotenv
from src.config import (
    BOOK_TFIDF_MAX_FEATURES,
    HYBRID_ALPHA_CONTENT,
    LLM_MODEL_DECIDER,
    LLM_MODEL_RERANKER,
    MAX_CANDIDATES_FOR_LLM,
)


# .env 로드 (프로젝트 루트의 .env)
load_dotenv()

# ---------------------------
# 로깅/디버그 설정
# ---------------------------
# 환경변수 A2A_DEBUG=1 이면 디버그 모드
ENV_DEBUG = os.getenv("A2A_DEBUG", "0") == "1"

logger = logging.getLogger(__name__)


import json
import logging

logger = logging.getLogger(__name__)

def parse_decision(raw_text: str) -> dict:
    """
    LLM이 반환한 JSON 문자열을 dict로 파싱하고,
    빠진 필드는 기본값으로 채운다.
    """
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        logger.exception("LLM decider JSON 파싱 실패, fallback 사용")
        # 완전 실패했을 때 최소 구조
        return {
            "domain": "book",
            "mentioned_titles": [],
            "mood_keywords": [],
            "current_emotion": [],
            "desired_feeling": [],
            "content_mood": [],
            "genres": [],
            "genres_en": [],
            "strategy": "by_mood",
            "extra_constraints": [],
            "_debug": {
                "raw_response_text": raw_text,
            },
        }

    # 빠질 수 있는 필드들 기본값 채우기
    data.setdefault("domain", "book")
    data.setdefault("mentioned_titles", [])
    data.setdefault("mood_keywords", [])
    data.setdefault("current_emotion", [])
    data.setdefault("desired_feeling", [])
    data.setdefault("content_mood", [])
    data.setdefault("genres", [])
    data.setdefault("genres_en", [])
    data.setdefault("strategy", "by_mood")
    data.setdefault("extra_constraints", [])

    # 디버그용 필드
    data.setdefault("_debug", {})
    data["_debug"].setdefault("raw_response_text", raw_text)

    logger.debug("[Parsed JSON]\n%s", json.dumps(data, ensure_ascii=False, indent=2))
    return data

# ---------------------------
# OpenAI 클라이언트
# ---------------------------
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

SYSTEM_PROMPT = """
당신은 책 추천 시스템의 “전략 결정 + 감정/취향 파싱” 에이전트입니다.
사용자 입력을 분석하여 오직 JSON 객체만 반환하세요.

목표:
- 사용자의 감정/상황을 이해하고 추천 전략을 결정합니다.
- 감정 / 분위기 / 장르 / 제약 조건을 구조화합니다.
- 과도한 추론은 하지 않습니다. 명확한 단서가 있을 때만 추론합니다.

출력 스키마(JSON):

{
  "domain": "book",
  "mentioned_titles": [],
  "mood_keywords": [],
  "current_emotion": [],
  "desired_feeling": [],
  "content_mood": [],
  "genres": [],
  "genres_en": [],
  "strategy": "by_title | by_mood | hybrid",
  "extra_constraints": []
}

필드별 지침:

1) domain
- 항상 "book"

2) mentioned_titles
- 사용자가 직접 언급한 책 제목 리스트
- 없으면 []

3) mood_keywords
- 사용자 문장 속 감정/분위기 표현(한국어)을 그대로 추출합니다.
- 예: "우울해", "잔잔한", "따뜻한", "무서운", "몰입되는"

4) current_emotion
- 사용자의 현재 감정을 나타내는 영어 단어 리스트입니다.
- 아래 목록 중에서만 선택합니다:
  sad, depressed, tired, stressed, angry, anxious, lonely, scared, bored, calm

- 예:
  - "나 우울해" → ["sad"]
  - "요즘 너무 힘들고 지쳤어" → ["tired", "stressed"]
- 확신이 없다면 빈 리스트 [] 로 둡니다.

5) desired_feeling
- 사용자가 책을 통해 도달하고 싶은 감정(영어) 리스트입니다.
- 사용자의 발화에 명확한 단서가 있을 때만 추론합니다.
- 가능한 값 예시:
  comforted, cheered_up, calm, relaxed, immersed, thrilled, deep_dive, inspired

- 예:
  - "위로되는 책" → ["comforted"]
  - "아무 생각 안 나게 몰입되는" → ["immersed"]
- 애매하면 [] 로 둡니다.

6) content_mood
- 책 자체가 갖기를 원하는 분위기(영어) 리스트입니다.
- 예시:
  warm, healing, dark, scary, funny, hopeful, romantic, calm, thrilling, suspenseful, light

- 예:
  - "잔잔한 로맨스" → ["calm", "romantic"]
  - "무서운 소설" → ["scary", "dark"]
  - "몰입되는 스릴러" → ["thrilling", "suspenseful"]
- 사용자가 분위기를 언급하지 않으면 [] 로 둡니다.

7) genres / genres_en
- 사용자가 명시한 장르만 기록합니다.
- genres: 한국어 장르
- genres_en: 간단한 영어 매핑

  - "로맨스" → "romance"
  - "에세이" → "essay"
  - "판타지" → "fantasy"
  - "스릴러" → "thriller"
  - "공포" → "horror"

8) strategy
- 다음 기준으로 선택합니다.

  - 사용자가 특정 책을 언급한 경우:
    "by_title"
  - 감정/분위기/상황을 중심으로 요청한 경우:
    "by_mood"
  - 특정 책 + 감정/상황을 함께 언급한 경우:
    "hybrid"

9) extra_constraints
- 분량, 시간, 상황 등 추가 제약 조건을 한국어 한 문장씩 요약합니다.
- 예:
  - "지하철에서 가볍게 읽을" → "지하철에서 읽기 좋음"
  - "너무 두꺼운 건 싫어" → "두꺼운 책 제외"

중요 규칙:
- 설명 문장을 쓰지 말고, JSON 객체만 출력하세요.
- 모든 필드는 반드시 존재해야 합니다.
- 값이 없으면 [] (빈 리스트)를 사용하고, null이나 "" 는 쓰지 마세요.
- 과도한 추론을 하지 말고, 사용자 발화에 근거가 있는 정보만 채우세요.
"""


def decide_strategy_with_llm(user_input: str, debug: bool = False) -> Dict[str, Any]:
    """
    LLM에게 user_input을 넘겨 전략/취향 정보를 JSON으로 받는다.
    디버그 모드일 때는:
      - 사용자 입력
      - LLM raw 응답 텍스트
      - 파싱된 JSON
    을 로그/리턴값에 함께 포함한다.

    debug 인자를 True로 주거나, 환경변수 A2A_DEBUG=1 이면 디버그 ON.
    """
    # 함수 인자 또는 환경변수 중 하나라도 True이면 디버그
    debug_mode = debug or ENV_DEBUG

    if debug_mode:
        logger.debug("=== [LLM Decider] 시작 ===")
        logger.debug("사용자 입력: %s", user_input)

    try:
        response = client.responses.create(
            model=LLM_MODEL_DECIDER,  # 필요에 따라 다른 모델로 교체 가능
            # reasoning={"effort": "low"},
            input=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": user_input,
                },
            ],
            max_output_tokens=300,
        )

        raw_text = _extract_text_from_response(response)
        cleaned_text = _strip_markdown_fence(raw_text)

        if debug_mode:
            logger.debug("[LLM Raw Text]\n%s", raw_text)
            logger.debug("[LLM Cleaned JSON Text]\n%s", cleaned_text)

        # JSON 파싱
        data = json.loads(cleaned_text)

        # 최소 안전장치
        if "strategy" not in data:
            data["strategy"] = "by_mood"

        # 디버그 정보 JSON 안에도 포함
        if debug_mode:
            data["_debug"] = {
                "user_input": user_input,
                "raw_response_text": raw_text,
                "cleaned_json_text": cleaned_text,
            }

            # 핵심 필드들도 보기 좋게 로그에 찍기
            logger.debug("[Parsed JSON]\n%s", json.dumps(data, ensure_ascii=False, indent=2))
            logger.debug("→ domain: %s", data.get("domain"))
            logger.debug("→ strategy: %s", data.get("strategy"))
            logger.debug("→ mentioned_titles: %s", data.get("mentioned_titles"))
            logger.debug("→ mood_keywords: %s", data.get("mood_keywords"))
            logger.debug("→ genres: %s", data.get("genres"))
            logger.debug("→ extra_constraints: %s", data.get("extra_constraints"))
            logger.debug("=== [LLM Decider] 종료 ===")

        return data

    except Exception as e:
        logger.exception("[LLM 오류] 전략 결정을 기본값으로 대체합니다.")

        # 아주 단순한 fallback
        fallback = {
            "domain": "book",
            "mentioned_titles": [],
            "mood_keywords": [],
            "genres": [],
            "strategy": "by_mood",
            "extra_constraints": [],
        }

        if debug_mode:
            fallback["_debug"] = {
                "user_input": user_input,
                "error": str(e),
            }

        return fallback

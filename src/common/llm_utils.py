import logging
from typing import Any

logger = logging.getLogger(__name__)
def _extract_text_from_response(response: Any) -> str:
    """
    OpenAI Responses API 응답 객체에서 텍스트만 안전하게 꺼내는 헬퍼.
    (모델/버전이 바뀌어도 최소한 디버깅에 쓸 문자열은 뽑힐 수 있게)
    """
    try:
        # 최신 Responses API 포맷 기준 (output[0].content[0].text)
        first_output = response.output[0]
        first_content = first_output.content[0]
        text = getattr(first_content, "text", str(first_content))
        return str(text)
    except Exception as e:
        logger.warning("응답 텍스트 추출 실패, 전체 응답을 문자열로 반환합니다: %s", e)
        return str(response)


def _strip_markdown_fence(text: str) -> str:
    """
    LLM이 ```json ... ``` 처럼 코드블록으로 감싸서 줄 때를 대비한 처리.
    """
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # 첫 줄은 ``` 또는 ```json 이라고 가정
        lines = lines[1:]
        # 마지막 줄에 ``` 가 있으면 제거
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text

# src/book/debug_decider.py
from src.book.llm_decider import decide_strategy_with_llm

def run_one(text: str):
    print("=" * 80)
    print("[USER]", text)
    decision = decide_strategy_with_llm(text, debug=True)
    print("[DECISION]")
    import json
    print(json.dumps(decision, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    cases = [
        "나 요즘 너무 우울해.. 위로되는 책 추천해줘",
        "해리 포터 같은 판타지 소설 없을까?",
        "야밤에 잔잔한 로맨스 읽고 싶어",
        "아무 생각 안 나게 몰입되는 스릴러 소설 없을까",
        "출퇴근 지하철에서 가볍게 읽을 에세이 추천해줘",
    ]
    for c in cases:
        run_one(c)

# src/common/state_types.py
from typing import TypedDict, List, Dict, Any

class BaseRecState(TypedDict, total=False):
    user_input: str
    user_id: int
    decision: Dict[str, Any]
    candidates: List[Dict[str, Any]]
    reranked: List[Dict[str, Any]]
    natural_output: str
from collections import Counter, defaultdict
from typing import List
import re
import uuid

def _extract_answer(text: str) -> str:
    """Extract answer from boxed notation or return full text if not found"""
    match = re.findall(r'\\boxed{((?:[^{}]|{[^{}]*})*)}', text)[-1]
    if match=='':
        match=str(uuid.uuid1())
    return match

def _agg_prm_min_max(x_list: List[str], ans_list: List[str], v_list: List[List[float]]) -> str:
    """Aggregate multiple candidates using min-max voting strategy"""
    v_list = [min(v) if v else -1.0 for v in v_list]

    text_max = x_list[v_list.index(max(v_list))]
    return text_max

def _agg_prm_last_max(x_list: List[str], ans_list: List[str], v_list: List[List[float]]) -> str:
    """Aggregate multiple candidates using last step score"""
    v_list = [v[-1] if v else -1.0 for v in v_list]

    text_max = x_list[v_list.index(max(v_list))]
    return text_max

def _agg_majority_vote(x_list: List[str], ans_list: List[str], v_list: List[List[float]]) -> str:
    """Aggregate multiple candidates using majority voting on extracted answers"""
    valid_answers = [ans for ans in ans_list if ans]
    
    if not valid_answers:
        v_list = [v[-1] if v else -1.0 for v in v_list]
        return x_list[v_list.index(max(v_list))]

    counts = Counter(valid_answers)
    most_common = max(counts, key=counts.get)
    
    for ans, text in zip(ans_list, x_list):
        if ans == most_common:
            return text

    return x_list[0]

def _agg_orm_vote(x_list: List[str], ans_list: List[str], v_list: List[float]) -> str:
    """Aggregate by summing scores for identical answers"""
    assert len(ans_list) == len(v_list)
    ans_dict = defaultdict(lambda: 0.0)
    for ans, v in zip(ans_list, v_list):
        ans_dict[ans] += v

    highest_ans = max(ans_dict, key=ans_dict.get)
    
    for ans, text in zip(ans_list, x_list):
        if ans == highest_ans:
            return text
            
    return x_list[0]

def _agg_prm_min_vote(x_list: List[str], ans_list: List[str], v_list: List[List[float]]) -> str:
    """Aggregate using minimum scores and vote by summing scores"""
    v_list = [min(v) if v else -1.0 for v in v_list]
    return _agg_orm_vote(x_list, ans_list, v_list)

def _agg_prm_last_vote(x_list: List[str], ans_list: List[str], v_list: List[List[float]]) -> str:
    """Aggregate using last scores and vote by summing scores"""
    v_list = [v[-1] if v else -1.0 for v in v_list]
    return _agg_orm_vote(x_list, ans_list, v_list)
"""
LLM Reranking using /v1/completions API.
Takes top candidates and returns ordered list based on LLM relevance judgment.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import urllib.request
from old.utils import g, standard_key

LM_BASE_URL = "http://127.0.0.1:1234"
LM_API_KEY = "lmstudio"
DEFAULT_MODEL = "google/gemma-4-e2b"
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 32
RERANK_K = 3
OUTPUT_K = 3


def parse_numbers(response, candidate_ids):
    """
    Parse numbered candidates from LLM response.
    Returns list of candidate IDs in ranked order.
    """
    if not response:
        return None
    
    response_upper = response.upper()
    
    found = []
    for cid in candidate_ids:
        if cid.upper() in response_upper:
            found.append(cid)
    
    for cid in candidate_ids:
        if len(found) >= RERANK_K:
            break
        if cid not in found:
            found.append(cid)
    
    return found[:RERANK_K]


def llm_rerank(query, candidates):
    """
    Rerank candidates using LLM via /v1/completions API.
    Returns reranked list of candidate IDs.
    """
    standards = g("standards")
    id_to_std = {standard_key(s): s for s in standards}
    
    candidate_display = []
    for cid in candidates:
        if cid not in id_to_std:
            continue
        s = id_to_std[cid]
        title = s.get("title", "")[:80]
        candidate_display.append((cid, title))
    
    if len(candidate_display) < 2:
        return candidates[:OUTPUT_K]
    
    candidates_list = "\n".join([f"{i+1}. {title}" for i, (cid, title) in enumerate(candidate_display)])
    
    prompt = f"""Query: {query}

Candidates:
{candidates_list}

Most relevant candidates (numbers):"""

    payload = {
        "model": DEFAULT_MODEL,
        "prompt": prompt,
        "max_tokens": LLM_MAX_TOKENS,
        "temperature": LLM_TEMPERATURE,
        "stream": False
    }
    
    req = urllib.request.Request(
        f"{LM_BASE_URL}/v1/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-api-key": LM_API_KEY},
        method="POST",
    )
    
    llm_output = ""
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            response = json.loads(resp.read().decode("utf-8"))
            llm_output = response.get("choices", [{}])[0].get("text", "").strip()
    except Exception:
        pass
    
    candidate_ids = [cid for cid, _ in candidate_display]
    reranked = parse_numbers(llm_output, candidate_ids)
    
    if reranked is None:
        reranked = candidates[:RERANK_K]
    
    return reranked

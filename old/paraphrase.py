"""
LLM Paraphrase generation using /v1/messages API.
Always enabled in the baseline pipeline.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import urllib.request
from old.retrieval import retrieve_dense

LM_BASE_URL = "http://127.0.0.1:1234"
LM_API_KEY = "lmstudio"
DEFAULT_MODEL = "google/gemma-4-e2b"
PARAPHRASE_TOP_DENSE = 50


def generate_paraphrase(query):
    """
    Generate a paraphrased version of the query using precise BIS technical terminology.
    Returns the paraphrased query string.
    """
    payload = {
        "model": DEFAULT_MODEL,
        "max_tokens": 48,
        "temperature": 0.4,
        "system": "You are a technical assistant for BIS standards lookup.",
        "messages": [{"role": "user", "content": f"Rewrite the query using precise BIS technical terminology. Return ONE short sentence only.\n\nQuery: {query}\n\nRewritten:"}]
    }
    
    req = urllib.request.Request(
        f"{LM_BASE_URL}/v1/messages",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-api-key": LM_API_KEY},
        method="POST",
    )
    
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            response = json.loads(resp.read().decode("utf-8"))
            text = response.get("content", [{}])[0].get("text", "").strip()
            return text if text else ""
    except Exception:
        return ""


def paraphrase_retrieval(query, k=PARAPHRASE_TOP_DENSE):
    """
    Generate paraphrase and retrieve top-k dense candidates.
    Returns list of (doc_id, score).
    """
    para_query = generate_paraphrase(query)
    if not para_query:
        return []
    return retrieve_dense(para_query, k=k)

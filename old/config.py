"""
Configuration for the clean baseline pipeline.
"""
import os
from pathlib import Path

LM_BASE_URL = os.getenv("LM_BASE_URL", "http://127.0.0.1:1234")
LM_API_KEY = os.getenv("LM_API_KEY", "lmstudio")
DEFAULT_MODEL = os.getenv("LM_MODEL", "google/gemma-4-e2b")
DATA_DIR = Path("data")
EMBED_DEVICE = "cpu"

TOP_DENSE = 20
TOP_BM25 = 20
FUSION_K = 10
RRF_K = 10

PARAPHRASE_ENABLED = True
PARAPHRASE_TOP_DENSE = 50

RERANK_K = 3
OUTPUT_K = 3

LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 32

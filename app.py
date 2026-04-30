from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
import pydantic
from typing import List, Optional
import time
import os
import sys
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import existing inference pipeline
try:
    from inference import run_pipeline
except ImportError:
    def run_pipeline(query):
        time.sleep(1.0)
        return {
            "retrieved": ["IS 455: 1989", "IS 269: 2015"],
            "rationale": "Demo rationale: This standard covers Portland slag cement requirements.",
            "latency_seconds": 1.0
        }

app = FastAPI(title="BIS Standards Discovery API")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

class SearchQuery(pydantic.BaseModel):
    query: str
    category_id: Optional[str] = None

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={})

@app.get("/api/guided_data")
async def get_guided_data():
    try:
        # Resolve paths relative to project root
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        profiles_path = os.path.join(data_dir, "section_profiles.json")
        metadata_path = os.path.join(data_dir, "metadata_store.json")
        
        with open(profiles_path, "r", encoding="utf-8") as f:
            profiles = json.load(f)
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
            
        return JSONResponse(content={"profiles": profiles, "metadata": metadata})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/api/search")
async def search(data: SearchQuery):
    if not data.query or len(data.query.strip()) < 2:
        raise HTTPException(status_code=400, detail="Query too short")
    
    try:
        result = run_pipeline(data.query)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    # Use CPU to avoid CUDA hangs if needed
    os.environ["BIS_FORCE_CPU"] = "1"
    uvicorn.run(app, host="127.0.0.1", port=8000)

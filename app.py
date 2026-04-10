import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from config import MODEL_PATH, HOST, PORT
from models.patho_model import PathoVLModel
from services.tenx_service import process_window

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Pathology Inference Service")

# 全局模型实例
model = None

@app.on_event("startup")
async def load_model():
    global model
    model = PathoVLModel(MODEL_PATH)
    logger.info("Model loaded")

# ========== 请求/响应模型 ==========
class TileCoord(BaseModel):
    x: int   # 瓦片左上角x坐标（像素）
    y: int   # 瓦片左上角y坐标（像素）

class TenXWindowRequest(BaseModel):
    tile_root: str
    level: int
    windows: List[List[TileCoord]]   # 每个窗口包含16个坐标（4x4顺序）

class TenXWindowResponse(BaseModel):
    results: List[dict]

@app.post("/infer/10x_window", response_model=TenXWindowResponse)
async def infer_10x_window(req: TenXWindowRequest):
    results = []
    for window_coords in req.windows:
        if len(window_coords) != 16:
            raise HTTPException(status_code=400, detail="Each window must contain exactly 16 coordinates")
        # 转换为 (x,y) 元组列表
        coords = [(c.x, c.y) for c in window_coords]
        result = process_window(model, req.tile_root, req.level, coords)
        results.append(result)
    return TenXWindowResponse(results=results)

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
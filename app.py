import logging
import base64
import io
from PIL import Image
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

class ChatRequest(BaseModel):
    images: List[str]   # Base64 编码的图片列表（不包含 data:image/... 前缀，仅纯 base64）
    question: str

class ChatResponse(BaseModel):
    think: str
    answer: str
    raw_output: str

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

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.images:
        raise HTTPException(status_code=400, detail="At least one image is required")
    # 解码 Base64 图像为 PIL Image
    pil_images = []
    for b64_str in req.images:
        try:
            img_data = base64.b64decode(b64_str)
            img = Image.open(io.BytesIO(img_data)).convert("RGB")
            pil_images.append(img)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image base64: {str(e)}")
    # 调用模型推理
    result = model.infer_multiple_images(pil_images, req.question)
    return ChatResponse(
        think=result.get("think", ""),
        answer=result.get("answer", ""),
        raw_output=result.get("raw_output", "")
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
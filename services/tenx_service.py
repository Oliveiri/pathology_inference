import os
from typing import List, Tuple
from utils.image_utils import stitch_4x4_tiles, is_blank
from utils.prompt_templates import TENX_WINDOW_PROMPT
import config
import logging

logger = logging.getLogger(__name__)

def process_window(model, tile_root: str, level: int, coords: List[Tuple[int, int]]) -> dict:
    """
    处理单个10x窗口
    tile_root: 相对路径（如 "1"），实际路径为 BASE_TILE_PATH/tile_root/level/x_y.png
    coords: 16个 (x, y) 坐标（左上角像素坐标），顺序行主序
    """
    tile_paths = []
    for (x, y) in coords:
        # 正确文件名格式：f"{x}_{y}.png"
        path = os.path.join(config.BASE_TILE_PATH, tile_root, str(level), f"{x}_{y}.png")
        tile_paths.append(path)

    # 拼接
    try:
        stitched = stitch_4x4_tiles(tile_paths)
    except Exception as e:
        logger.error(f"Stitching failed: {e}")
        return {"error": "stitch_failed", "empty": True}

    # 空白判断
    if is_blank(stitched, threshold=config.BLANK_THRESHOLD):
        return {
            "empty": True,
            "output": "",
            "tumor_polygon": [],
            "subtype_scores": [0.0]*7,
            "boundary_type": "",
            "grade4_prob": 0.0
        }

    # 模型推理
    result = model.infer_single(stitched, TENX_WINDOW_PROMPT)
    raw_output = result.get("raw_output", "")
    tumor_polygon = result.get("tumor_polygon", [])
    subtype_scores = result.get("subtype_scores", [0.0]*7)
    boundary_type = result.get("boundary_type", "pushing")
    grade4_prob = result.get("grade4_prob", 0.0)

    return {
        "tumor_polygon": tumor_polygon,
        "subtype_scores": subtype_scores,
        "boundary_type": boundary_type,
        "grade4_prob": grade4_prob,
        "output": raw_output,
        "empty": False
    }
import os
import cv2
import numpy as np
from PIL import Image
from typing import List
import logging

logger = logging.getLogger(__name__)

def read_tile(tile_path: str, default_color=255) -> np.ndarray:
    """读取瓦片，若文件不存在则返回空白图像（白色）并记录日志"""
    if os.path.exists(tile_path):
        img = cv2.imread(tile_path)
        if img is not None:
            return img
        else:
            logger.warning(f"无法解码图像文件: {tile_path}")
    else:
        logger.warning(f"瓦片文件不存在: {tile_path}")
    # 生成空白图像
    blank = np.full((256, 256, 3), default_color, dtype=np.uint8)
    return blank

def stitch_4x4_tiles(tile_paths: List[str]) -> Image.Image:
    """
    将16张256x256瓦片拼接成1024x1024图像
    tile_paths 顺序：从左到右，从上到下（行主序）
    """
    if len(tile_paths) != 16:
        raise ValueError(f"Need exactly 16 tile paths, got {len(tile_paths)}")
    tiles = [read_tile(p) for p in tile_paths]
    rows = []
    for i in range(4):
        row = np.hstack(tiles[i*4:(i+1)*4])
        rows.append(row)
    full = np.vstack(rows)
    full_rgb = cv2.cvtColor(full, cv2.COLOR_BGR2RGB)
    return Image.fromarray(full_rgb)

def is_blank(image: Image.Image, threshold=250) -> bool:
    """判断图像是否为空白（灰度均值 > threshold）"""
    gray = image.convert('L')
    mean = np.mean(gray)
    return mean > threshold
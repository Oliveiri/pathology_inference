import os

MODEL_PATH = os.getenv("MODEL_PATH", "/data/models")
DEVICE = "cuda"
TILE_SIZE = 256
WINDOW_TILE_SIZE = 4   # 4x4 窗口
STRIDE_TILES = 4       # 步长 = 窗口大小（无重叠）
BLANK_THRESHOLD = 250
MAX_NEW_TOKENS = 512

# 瓦片存储根目录（所有 WSI 的父目录）
BASE_TILE_PATH = "/data/tiles"

HOST = "0.0.0.0"
PORT = 8000
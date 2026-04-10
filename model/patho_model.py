import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
from typing import Dict, Any
import json
import logging

logger = logging.getLogger(__name__)

class PathoVLModel:
    def __init__(self, model_path: str):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
        self.device = self.model.device
        logger.info(f"Model loaded on {self.device}")

    def infer_single(self, image: Image.Image, prompt: str) -> Dict[str, Any]:
        messages = [
            {
                "role": "system",
                "content": "You are a pathology expert, your task is to answer question step by step. Use the following format:<think> Your step-by-step reasoning </think><answer> Your final answer </answer>"
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=2048,  # 适当增加
                do_sample=False,
                temperature=0.0
            )
        output = self.processor.decode(generated_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        # 提取 <answer> 标签内的内容
        answer_start = output.find('<answer>')
        answer_end = output.find('</answer>')
        if answer_start != -1 and answer_end != -1:
            answer_content = output[answer_start + 8:answer_end].strip()
        else:
            answer_content = output  # 降级处理

        # 尝试从 answer_content 中提取 JSON
        try:
            start = answer_content.find('{')
            end = answer_content.rfind('}') + 1
            if start != -1 and end > start:
                json_str = answer_content[start:end]
                parsed = json.loads(json_str)
                parsed["raw_output"] = output  # 保留原始完整输出
                return parsed
            else:
                return {"error": "No JSON found in answer", "raw_output": output}
        except json.JSONDecodeError as e:
            return {"error": f"JSON parse error: {e}", "raw_output": output}
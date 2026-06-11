from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from .base import BaseModel


class Qwen3VLLocalChat(BaseModel):
    INTERLEAVE = True

    def __init__(self, model_path: str, max_new_tokens: int = 128, device_map: str = 'auto', **kwargs):
        super().__init__()
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            trust_remote_code=True,
        ).eval()

    def use_custom_prompt(self, dataset):
        return False

    def build_prompt(self, line, dataset):
        raise NotImplementedError

    def _messages(self, message):
        content = []
        for item in message:
            if item['type'] == 'image':
                content.append({'type': 'image', 'image': str(Path(item['value']).resolve())})
            elif item['type'] == 'text':
                content.append({'type': 'text', 'text': item['value']})
        return [{'role': 'user', 'content': content}]

    def generate_inner(self, message, dataset=None):
        messages = self._messages(message)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_paths = [item['value'] for item in message if item['type'] == 'image']
        images = [Image.open(path).convert('RGB') for path in image_paths]
        inputs = self.processor(text=[text], images=images, return_tensors='pt')
        inputs = inputs.to(self.model.device)
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        generated_ids = generated_ids[:, inputs.input_ids.shape[1]:]
        response = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return response.strip()

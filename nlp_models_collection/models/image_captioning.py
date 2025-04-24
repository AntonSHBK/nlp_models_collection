from typing import Optional

import torch
from transformers import (
    VisionEncoderDecoderModel,
    AutoTokenizer,
    AutoProcessor,
    PreTrainedModel,
    ProcessorMixin,
    pipeline,
    Pipeline
)

from .base_model import BaseModelLoader


class ImageCaptioningModelLoader(BaseModelLoader):
    """
    Класс для генерации подписей к изображениям (image captioning) с помощью моделей типа BLIP, GIT, и т.д.
    """

    def _load_model(self) -> PreTrainedModel:
        model = VisionEncoderDecoderModel.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            use_auth_token=self.hf_token,
            torch_dtype=self.params.get("torch_dtype", "auto"),
        )
        model.to(self.device)
        model.eval()
        return model

    def _load_tokenizer(self):
        return AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            use_auth_token=self.hf_token,
        )

    def _load_processor(self) -> ProcessorMixin:
        return AutoProcessor.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            use_auth_token=self.hf_token,
        )
        
    def _load_pipeline(self) -> Optional[Pipeline]:
        task = self.params.get("task", "image-to-text")
        return pipeline(
            task=task,
            model=self.model,
            tokenizer=self.tokenizer,
            processor=self.processor,
            device=self.device,
        )

    def generate(self, image, **kwargs):
        """
        Генерация подписи к изображению. Возвращает текст.
        """
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **kwargs)

        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

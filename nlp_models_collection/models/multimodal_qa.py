from typing import Optional

import torch
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    pipeline,
    Pipeline
)

from .base_multimodal_model import BaseMultimodalModelLoader


class MultimodalQAModelLoader(BaseMultimodalModelLoader):
    """
    Класс для мультимодальных моделей вопрос-ответ (например, BLIP2, OFA),
    которые принимают изображение и текстовый вопрос, и выдают текстовый ответ.
    """

    def _load_model(self):
        model = AutoModelForVision2Seq.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            use_auth_token=self.hf_token,
            torch_dtype="auto",
        )
        model.to(self.device)
        model.eval()
        return model

    def _load_processor(self):
        return AutoProcessor.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            use_auth_token=self.hf_token,
        )

    def _load_pipeline(self) -> Optional[Pipeline]:
        task = self.params.pop("task", "image-text-to-text")
        return pipeline(
            task=task,
            model=self.model,
            processor=self.processor,
            device=self.device,
        )

    def generate(self, image, question: str, **kwargs):
        """
        Генерирует текстовый ответ на вопрос по изображению.
        """
        inputs = self.processor(images=image, text=question, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **kwargs)

        return self.processor.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

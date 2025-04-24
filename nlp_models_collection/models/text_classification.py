from typing import Optional

import torch
from transformers import (
    AutoModelForSequenceClassification,
    pipeline,
    Pipeline
)

from .base_text_model import BaseTextModelLoader


class TextClassificationModelLoader(BaseTextModelLoader):
    """
    Класс для загрузки моделей классификации текста (например, BERT, RoBERTa, DistilBERT).
    """

    def _load_model(self):
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            use_auth_token=self.hf_token,
        )
        model.to(self.device)
        return model

    def _load_pipeline(self) -> Optional[Pipeline]:
        task = self.params.get("task", "text-classification")
        return pipeline(
            task=task,
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
        )    

    def _generate(self, input_text, **kwargs):
        """
        Предсказывает метку класса для входного текста. Возвращает индекс класса и логиты.
        """
        inputs = self._tokenize(input_text, **kwargs.get("tokenize_kwargs", {}))
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            predicted_class = logits.argmax(dim=-1)
        return {
            "class_id": predicted_class.cpu().tolist(),
            "logits": logits.cpu().tolist()
        }
    
    
    def predict_label(self, input_text, **kwargs):
        """
        Возвращает предсказанную метку класса (str) вместо ID.
        """
        result = self.generate(input_text, **kwargs)
        class_ids = result["class_id"]

        id2label = self.model.config.id2label
        labels = [id2label[str(cls_id)] if str(cls_id) in id2label else str(cls_id) for cls_id in class_ids]

        return labels

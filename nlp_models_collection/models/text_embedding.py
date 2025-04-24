from typing import Union, List, Optional, Literal

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from .base_model import BaseModelLoader


class TextEmbeddingModelLoader(BaseModelLoader):
    """
    Класс для генерации эмбеддингов текста с возможностью выбора стратегии агрегирования.
    """

    def _load_model(self) -> PreTrainedModel:
        model = AutoModel.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            use_auth_token=self.hf_token,
            torch_dtype="auto",
        )
        model.to(self.device)
        model.eval()
        return model

    def _load_tokenizer(self) -> PreTrainedTokenizer:
        return AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            use_auth_token=self.hf_token,
        )

    def _load_pipeline(self) -> Optional[None]:
        return None

    def _load_processor(self) -> Optional[None]:
        return None

    def embed(
        self,
        text: Union[str, List[str]],
        pooling: Literal["cls", "mean", "max", "attention"] = "cls",
        normalize: bool = False,
        **kwargs
    ) -> List[List[float]]:
        """
        Получает эмбеддинги текста с выбранной стратегией pooling-а.
        """
        if isinstance(text, str):
            text = [text]

        encoded_input = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            **kwargs
        ).to(self.device)

        with torch.no_grad():
            model_output = self.model(**encoded_input)

        embeddings = self._apply_pooling(model_output, encoded_input, pooling)

        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings.cpu().tolist()

    def _apply_pooling(self, model_output, encoded_input, pooling: str) -> torch.Tensor:
        last_hidden = model_output.last_hidden_state  # (batch, seq_len, hidden_size)

        if pooling == "cls":
            return last_hidden[:, 0]
        elif pooling == "mean":
            attention_mask = encoded_input["attention_mask"]
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size())
            sum_embeddings = torch.sum(last_hidden * input_mask_expanded, dim=1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            return sum_embeddings / sum_mask
        elif pooling == "max":
            return torch.max(last_hidden, dim=1).values
        elif pooling == "attention":
            # Просто пример: softmax по длине последовательности
            attn_weights = torch.softmax(encoded_input["attention_mask"].float(), dim=1).unsqueeze(-1)
            return torch.sum(last_hidden * attn_weights, dim=1)
        else:
            raise ValueError(f"Unknown pooling strategy: {pooling}")

    def similarity(
        self,
        a: Union[str, List[str]],
        b: Union[str, List[str]],
        pooling: str = "cls"
    ) -> Union[float, List[float]]:
        """
        Косинусное сходство между строками или списками строк.
        """
        a_emb = torch.tensor(self.embed(a, pooling=pooling, normalize=True)).to(self.device)
        b_emb = torch.tensor(self.embed(b, pooling=pooling, normalize=True)).to(self.device)

        # Расчёт косинусного сходства
        sim = F.cosine_similarity(a_emb, b_emb)
        return sim.item() if sim.numel() == 1 else sim.tolist()

    def batch_embed(
        self,
        texts: List[str],
        pooling: str = "cls"
    ) -> torch.Tensor:
        """
        Батчевая генерация эмбеддингов с нормализацией.
        """
        embeddings = torch.tensor(self.embed(texts, pooling=pooling, normalize=True)).to(self.device)
        return embeddings

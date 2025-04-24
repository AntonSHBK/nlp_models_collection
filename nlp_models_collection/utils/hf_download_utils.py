import os
from pathlib import Path
from typing import Optional
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    AutoModelForImageClassification,
    AutoModelForVision2Seq,
    AutoTokenizer,
    AutoProcessor
)


def get_hf_token(explicit_token: Optional[str] = None) -> Optional[str]:
    """
    Возвращает Hugging Face токен: переданный явно или из переменной окружения HF_TOKEN.
    """
    return explicit_token or os.getenv("HF_TOKEN")


def download_model(
    model_name: str,
    model_type: str = "AutoModel",
    cache_dir: Path = Path("cache_dir"),
    hf_token: Optional[str] = None,
    trust_remote_code: bool = False,
    **kwargs
):
    """
    Загружает модель нужного типа с Hugging Face.
    :param model_type: тип модели ('AutoModel', 'AutoModelForCausalLM', ...)
    """
    hf_token = get_hf_token(hf_token)

    model_classes = {
        "AutoModel": AutoModel,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoModelForSequenceClassification": AutoModelForSequenceClassification,
        "AutoModelForSeq2SeqLM": AutoModelForSeq2SeqLM,
        "AutoModelForImageClassification": AutoModelForImageClassification,
        "AutoModelForVision2Seq": AutoModelForVision2Seq,
    }

    if model_type not in model_classes:
        raise ValueError(f"Unsupported model type: {model_type}")

    return model_classes[model_type].from_pretrained(
        model_name,
        cache_dir=cache_dir,
        use_auth_token=hf_token,
        trust_remote_code=trust_remote_code,
        **kwargs
    )


def download_tokenizer(
    model_name: str,
    cache_dir: Path = Path("cache_dir"),
    hf_token: Optional[str] = None,
    trust_remote_code: bool = False,
    **kwargs
):
    hf_token = get_hf_token(hf_token)
    return AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        use_auth_token=hf_token,
        trust_remote_code=trust_remote_code,
        **kwargs
    )


def download_processor(
    model_name: str,
    cache_dir: Path = Path("cache_dir"),
    hf_token: Optional[str] = None,
    **kwargs
):
    hf_token = get_hf_token(hf_token)
    return AutoProcessor.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        use_auth_token=hf_token,
        **kwargs
    )

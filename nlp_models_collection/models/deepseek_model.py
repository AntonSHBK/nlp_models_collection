from pathlib import Path

from .base_causal_text_model import BaseCausalTextModelLoader


class DeepSeekModelLoader(BaseCausalTextModelLoader):
    """
    Класс для загрузки и использования модели DeepSeek (например, deepseek-ai/deepseek-llm-r1-b70).
    """

    def __init__(
        self, 
        model_name: str,
        cache_dir: Path = Path("cache_dir"),
        task: str ="text-generation",
        **kwargs
    ):
        kwargs.setdefault("task", task)
        kwargs.setdefault("torch_dtype", "auto")
        super().__init__(model_name, cache_dir, **kwargs)

    def format_chat_prompt(self, messages: list) -> str:
        """
        Форматирует сообщения для модели DeepSeek-R1-Distill-Qwen-1.5B.
        Использует простой человекоподобный стиль диалога:
        "User: ... \nAssistant: ..."
        """
        prompt = ""
        for msg in messages:
            role = msg["role"].capitalize()
            content = msg["content"].strip()
            prompt += f"{role}: {content}\n"
        prompt += "Assistant: "
        return prompt

from .base_text_model import BaseTextModelLoader


class TextToTextModelLoader(BaseTextModelLoader):
    """
    Класс загрузки моделей типа text-to-text (T5, FLAN-T5, BART и др.)
    """

    def __init__(self, model_name: str, task: str ="text2text-generation", **kwargs):
        kwargs.setdefault("task", task)
        super().__init__(model_name, **kwargs)

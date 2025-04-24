# Описание всех классов:

- Названия класса
- Базового класса (наследования)
- Типа задачи
- Примеры применений
- Особенности реализации

---

### 📋 Таблица: реализованные классы моделей

| Класс                              | Базовый класс                    | Тип задачи                        | Примеры моделей                                    | Особенности/назначение                                                 |
|-----------------------------------|----------------------------------|----------------------------------|---------------------------------------------------|------------------------------------------------------------------------|
| `BaseModelLoader`                 | `ABC` (абстрактный класс)        | Базовый загрузчик                | –                                                 | Общий интерфейс: модель, токенизатор, pipeline, processor             |
| `BaseTextModelLoader`            | `BaseModelLoader`                | Text-to-text (Seq2Seq)           | T5, FLAN, BART                                     | Использует `AutoModelForSeq2SeqLM`, `text2text-generation` pipeline   |
| `BaseCausalTextModelLoader`      | `BaseModelLoader`                | Causal LM                        | GPT, LLaMA, DeepSeek                              | Использует `AutoModelForCausalLM`, ручной `.generate()`               |
| `BaseVisionModelLoader`          | `BaseModelLoader`                | Image input                      | ViT, ResNet                                        | Использует `AutoProcessor`, `AutoModelForImageClassification`         |
| `BaseMultimodalModelLoader`      | `BaseModelLoader`                | Text + Image                     | BLIP, Flamingo                                     | Обрабатывает несколько входов: текст + изображение                    |
| `TextToTextModelLoader`          | `BaseTextModelLoader`            | Текст → Текст                    | `google/flan-t5-small`                            | Seq2Seq генерация текста, удобно для перевода, перефразирования       |
| `TextClassificationModelLoader`  | `BaseTextModelLoader`            | Классификация текста             | `distilbert-base-uncased-finetuned-sst-2-english` | Возвращает логиты + метки; метод `predict_label()`                    |
| `TextEmbeddingModelLoader`       | `BaseModelLoader`                | Генерация эмбеддингов            | `sentence-transformers`, `e5`, `MiniLM`           | Поддерживает pooling (`cls`, `mean`, `max`, `attention`) и `similarity()` |
| `DeepSeekModelLoader`            | `BaseCausalTextModelLoader`      | Текстовая генерация              | `deepseek-llm-r1-distill-qwen-1.5b`               | Компактная causal LM; генерация с параметрами                         |
| `ImageClassificationModelLoader`| `BaseVisionModelLoader`          | Классификация изображений        | `google/vit-base-patch16-224`                     | Возвращает top-k классы; метод `predict_label()`                      |
| `ImageCaptioningModelLoader`     | `BaseModelLoader`                | Генерация подписи к изображению  | `vit-gpt2`, `BLIP`, `GIT`                         | Использует `VisionEncoderDecoderModel`, `.generate()`                 |
| `MultimodalQAModelLoader`        | `BaseMultimodalModelLoader`      | Вопрос-ответ по изображению      | `Salesforce/blip2-flan-t5-xl`                     | Принимает изображение + текст и генерирует текст                      |

---

### 📌 Ключевые фичи архитектуры:

- У всех классов реализованы **ленивые загрузчики компонентов** (`model`, `tokenizer`, `pipeline`, `processor`)
- Поддерживаются **ручные методы**: `generate()`, `tokenize()`, `decode()` с `**kwargs`
- Классы легко расширяемы: любой новый тип модели можно реализовать в 5-10 строках, переопределяя `_load_*` методы


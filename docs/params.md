## В `DeepSeekModelLoader` (и других `*ModelLoader`)

### Метод `.generate(...)` поддерживает 3 типа параметров:

```python
model.generate(
    input_text="your prompt here",
    tokenize_kwargs={...},
    generation_kwargs={...},
    decode_kwargs={...}
)
```

---

## 1. `tokenize_kwargs`: параметры токенизации

Передаются в `tokenizer(...)`, управляют тем, **как входной текст преобразуется в токены**.

| Параметр        | Описание |
|------------------|----------|
| `padding`        | `"max_length"` или `True` — выравнивание по длине |
| `truncation`     | `True` — обрезает слишком длинный ввод |
| `max_length`     | Максимальная длина входа (в токенах) |
| `return_tensors` | `pt` — возвращает PyTorch тензоры (автоматически устанавливается) |

### Пример:
```python
tokenize_kwargs={
    "padding": "max_length",
    "max_length": 128,
    "truncation": True
}
```

---

## 2. `generation_kwargs`: параметры генерации

Передаются в `model.generate(...)`, управляют **тем, как модель генерирует токены**.

| Параметр           | Описание |
|---------------------|----------|
| `temperature`       | Насколько «творчески» генерировать (0.7–1.0 — типично) |
| `top_k`             | Обрезка по k наиболее вероятным токенам |
| `top_p`             | Нуклеарная сэмплировка (например, 0.9) |
| `num_beams`         | Количество лучей при beam search (если используется) |
| `do_sample`         | Включить сэмплирование (важно при использовании `top_k`, `top_p`) |
| `max_new_tokens`    | Сколько новых токенов генерировать |
| `repetition_penalty`| Штраф за повторения |
| `eos_token_id`      | Токен окончания генерации |

### Пример:
```python
generation_kwargs={
    "temperature": 0.8,
    "top_p": 0.95,
    "do_sample": True,
    "max_new_tokens": 64
}
```

---

## 3. `decode_kwargs`: параметры декодирования токенов в текст

Передаются в `tokenizer.batch_decode(...)`.

| Параметр                | Описание |
|--------------------------|----------|
| `skip_special_tokens`    | Удалить токены вроде `<pad>`, `<eos>` |
| `clean_up_tokenization_spaces` | Чистка пробелов и пунктуации |

### Пример:
```python
decode_kwargs={
    "skip_special_tokens": True,
    "clean_up_tokenization_spaces": True
}
```

---

## Полный пример

```python
output = model.generate(
    input_text="Explain quantum computing in simple terms.",
    tokenize_kwargs={"padding": "max_length", "max_length": 128, "truncation": True},
    generation_kwargs={"temperature": 0.9, "top_p": 0.95, "max_new_tokens": 50, "do_sample": True},
    decode_kwargs={"skip_special_tokens": True}
)
print(output[0])
```
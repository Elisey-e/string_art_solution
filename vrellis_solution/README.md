# Vrellis-style string art (grvlbit / Petros Vrellis)

Популярный **жадный** алгоритм ниткографии, как в open-source проекте [grvlbit/stringart](https://github.com/grvlbit/stringart) (29★, вдохновлён [Petros Vrellis — A New Way to Knit](https://artof01.com/vrellis/works/knit.html)).

Такой подход используют для **продаваемых портретов**: гвозди по окружности, на каждом шаге выбирается хорда с максимальной «темнотой» по целевому изображению, остаток уменьшается.

## Запуск

```bash
python3 build_schema.py input.png --nails 240 --lines 5000 --weight 0.08 --out out
```

или `./run.sh`

## Параметры

| Параметр | По умолчанию | Описание |
|----------|--------------|----------|
| `--nails` | 240 | Гвозди по окружности |
| `--lines` | 5000 | Макс. число нитей (итераций) |
| `--weight` | 0.08 | Сила одной нити при вычитании |
| `--min-gap` | 12 | Мин. расстояние между гвоздями по дуге |

## Выход

- `schema.csv` — пары гвоздей `nail_a`, `nail_b`
- `preview.png` — растр ×3 (1311×1311 для входа 437px)
- `preview.svg` — вектор, бесконечный зум
- `metadata.json`

## Препроцессинг (как в grvlbit)

1. Grayscale → invert  
2. EDGE_ENHANCE_MORE  
3. Контраст ×1.4  

## Зависимости

```
numpy Pillow scikit-image
```

## Сравнение с FBP (task.md)

| | FBP (`fbp_solution`) | Vrellis (`vrellis_solution`) |
|--|---------------------|------------------------------|
| Метод | Radon + ramp + dither | Greedy по гвоздям |
| Задание | ✅ по task.md | Улучшение / SOTA-подход |
| Качество портрета | Среднее | Высокое |

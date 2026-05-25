# Вычислительная ниткография

Проект с двумя реализациями алгоритма вычислительной ниткографии.

## Решения

### 1. FBP-решение (по заданию) `fbp_solution/`

Алгоритм на основе **Filtered Back-Projection** — соответствует заданию (task.md).

**Особенности:**
- Инверсия изображения ДО Radon
- Чёрный фон, белые нити
- Random или error-diffusion дизеринг
- Непрерывные линии `(angle, ρ)`

**Результаты:**
- Лучшие параметры: 90 углов × 10 нитей = 325 нитей
- RMSE: 0.459, MAE: 0.364

**Запуск:**
```bash
cd fbp_solution
python3 build_schema.py input.png --angles 90 --threads 20 --seed 1 --out out
# → out/preview.png (×3) и out/preview.svg
```

### 2. Greedy-решение (высокое качество) `greedy_solution/`

Алгоритм с **жадным выбором линий по остатку** — значительно превосходит FBP.

**Особенности:**
- Хорды между гвоздями на окружности
- Жадный выбор по максимальному вкладу в ошибку
- Ранняя остановка по насыщению
- Комплексные метрики качества (SSIM, PSNR, зоновые)

**Результаты:**
- 300 гвоздей × 8000 линий
- RMSE: 0.191 (в 2.4× лучше FBP)
- SSIM: 0.997 (почти идеально)

**Запуск:**
```bash
cd greedy_solution
python3 build_schema.py input.png --nails 300 --min-nail-gap 10 --lines 8000 --darkness 0.02 --out out
# → out/preview.png (×3) и out/preview.svg
```

## Сравнение

| Метрика | FBP | Greedy |
|---------|-----|--------|
| RMSE | 0.459 | **0.191** |
| MAE | 0.364 | **0.154** |
| SSIM | N/A | **0.997** |
| Нити | 325 | 8000 |
| Время | ~4 сек | ~2 мин |
| По заданию | ✅ | ❌ |
| Качество | Удовлетворительно | **Отлично** |

## Структура проекта

```
string_art_solution/
├── fbp_solution/         # FBP-алгоритм (по заданию)
│   ├── build_schema.py   # основной скрипт
│   ├── string_art.py     # функции Radon, фильтрации
│   ├── visualize_schema.py
│   ├── run.sh
│   ├── README.md
│   └── out_example/      # пример результата
├── greedy_solution/      # Greedy-алгоритм (высокое качество)
│   ├── build_schema.py   # основной скрипт
│   ├── string_art.py     # nail catalogue, greedy selection
│   ├── metrics.py        # комплексные метрики
│   ├── visualize_schema.py
│   ├── run.sh
│   ├── README.md
│   └── out_example/      # пример результата
├── input.png             # входное изображение
├── task.md               # текст задания
├── DEBUG_REPORT.md       # подробный отчёт отладки
└── README.md             # этот файл
```

## Дополнительные файлы

- `DEBUG_REPORT.md` — подробный отчёт об отладке и сравнении алгоритмов
- `task.md` — оригинальный текст задания

## Зависимости

```bash
pip install numpy pillow scipy scikit-image
```

Для использования виртуального окружения:
```bash
# Создание
python3 -m venv .venv
source .venv/bin/activate

# Установка
pip install numpy pillow scipy scikit-image
```

## Формат входа

Входное изображение должно быть:
- квадратным
- одноканальным, режим Pillow `L` (оттенки серого)

## Литература

1. Birsak et al. "String Art: Towards Computational Fabrication of String Images" (TU Wien, 2018)
2. Petros Vrellis — оригинальный подход к вычислительной ниткографии
3. Bridges 2022 — "Computational String Art"
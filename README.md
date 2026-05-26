# Вычислительная ниткография

Три реализации алгоритма ниткографии для портретов.

## Решения

### 1. `fbp_solution/` — FBP по заданию (task.md)

Томографический алгоритм: Radon → ramp-фильтр → прореживание → random dither → обратная проекция.

**Улучшения для качества:**
- Инверсия и подъём яркости перед Radon (`--brightness-lift`)
- Нормировка вероятностей **по каждому углу**
- Взвешенная обратная проекция (яркость нити ∝ вероятность)
- 360 углов × 40 нитей на угол

```bash
cd fbp_solution && ./run.sh
```

### 2. `greedy_solution/` — жадный по остатку (эксперимент)

Хорды между гвоздями, sparse greedy, SSIM ≈ 0.997.

```bash
cd greedy_solution && ./run.sh
```

### 3. `vrellis_solution/` — Vrellis / grvlbit (SOTA для портретов)

Жадный алгоритм как в [grvlbit/stringart](https://github.com/grvlbit/stringart) и работах **Petros Vrellis** — стандарт для коммерческих портретов из нитей.

```bash
cd vrellis_solution && ./run.sh
```

## Сравнение (input.png 437×437)

| Решение | Нитей | RMSE | Назначение |
|---------|-------|------|------------|
| **FBP** | ~9000 | 0.39 | Сдача по task.md |
| **Greedy** | 8000 | 0.19 | Макс. метрики |
| **Vrellis** | 5000 | 0.02 | Качество портрета |

Каждая папка содержит `out_example/` с `preview.png` (×3), `preview.svg` и `schema.csv`.

## Зависимости

```bash
pip install numpy pillow scipy scikit-image
```

## Файлы

- `input.png` — тестовый портрет
- `task.md` — формулировка задания

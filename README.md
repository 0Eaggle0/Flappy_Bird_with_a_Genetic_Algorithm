# 🐦 Flappy Bird с Генетическим Алгоритмом

> Демонстрация нейроэволюции: популяция птиц учится играть в Flappy Bird через естественный отбор — без обратного распространения ошибки, без обучающих данных, только эволюция.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![Pygame](https://img.shields.io/badge/Pygame-2.6.1-green)
![NumPy](https://img.shields.io/badge/NumPy-2.4.4-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📖 О проекте

Этот проект объединяет классическую игру Flappy Bird с **генетическим алгоритмом (ГА)** для эволюции простых нейронных сетей, которые учатся играть самостоятельно. Каждая птица управляется небольшой нейронной сетью прямого распространения. Из поколения в поколение лучшие птицы передают свои «гены» (веса сети) следующему поколению через отбор, скрещивание и мутацию — как в биологической эволюции.

Все алгоритмы реализованы **с нуля** с использованием только NumPy и Pygame — без TensorFlow, PyTorch и других ML-фреймворков.

---

## ✨ Возможности

- 🧬 **Генетический алгоритм** с настраиваемым отбором, скрещиванием, мутацией и элитизмом
- 🧠 **Нейронные сети** (архитектура 5 → 6 → 1), эволюционирующие без обратного распространения
- 🎮 **Интерактивный GUI** — наблюдай за эволюцией в реальном времени
- 📊 **График приспособленности** — лучший / средний / худший результат по поколениям
- 🎛️ **Слайдеры** — настройка параметров ГА прямо во время работы
- ⚡ **Режим ускорения** — пропускай рендеринг и эволюционируй на максимальной скорости
- 👁️ **Визуализация сенсоров** — смотри, что «видит» каждая птица
- 📈 **11 уровней сложности** — трубы ускоряются, сужаются и начинают двигаться вертикально
- 🖥️ **Безголовый режим** — запуск эволюции без графики (для CI/серверов)

---

## 🖼️ Интерфейс

```
┌──────────────────────┬──────────────────────────┐
│                      │  Поколение:   42          │
│   ИГРОВОЕ ПОЛЕ       │  Живых:       12 / 50     │
│   600 × 600 пкс      │  Лучший счёт: 138 труб   │
│                      │                           │
│   [птицы летят]      │  ┌─ График приспособ. ──┐ │
│                      │  │  ████ макс            │ │
│                      │  │  ░░░░ среднее         │ │
│                      │  │  ···· мин             │ │
│                      │  └───────────────────────┘ │
│                      │                           │
│                      │  [Слайдеры]               │
│                      │  Популяция     ──●──      │
│                      │  Мутация       ─●───      │
│                      │  Элитизм       ──●──      │
│                      │  ...                      │
└──────────────────────┴──────────────────────────┘
```

---

## 🧠 Архитектура нейронной сети

Каждая птица управляется нейронной сетью **5 → 6 → 1**:

```
Входы (5)              Скрытый (6)        Выход (1)
─────────              ───────────        ─────────
y птицы (норм.)  ─┐
скорость птицы   ─┤─── tanh ───┐
дист. до трубы   ─┤─── tanh ───┤─── sigmoid ─── ПРЫЖОК?
центр зазора Y   ─┤─── tanh ───┘
скорость зазора  ─┘
```

| Слой    | Размер | Активация |
|---------|--------|-----------|
| Входной | 5      | —         |
| Скрытый | 6      | tanh      |
| Выходной| 1      | sigmoid   |

**Геном** — плоский массив из **43 чисел** (все веса и смещения):

```
W1 (5×6 = 30)  +  b1 (6)  +  W2 (6×1 = 6)  +  b2 (1)  =  43 гена
```

Выходной нейрон даёт команду `прыжок`, если его значение ≥ 0.5.

---

## 🧬 Генетический алгоритм

### Функция приспособленности

| Событие                              | Очки                  |
|--------------------------------------|-----------------------|
| Каждый прожитый кадр                 | +1                    |
| Каждый кадр вблизи центра зазора     | +0.5                  |
| Каждая пройденная труба              | +25                   |
| Прохождение 1000 труб (спидран)      | +500 × бонус скорости |

### Цикл эволюции

```
1. Запустить поколение → оценить всех птиц в игре
2. Отсортировать по приспособленности
3. Скопировать топ-N птиц без изменений (элитизм)
4. Заполнить остаток популяции:
     выбрать 2 родителя → скрещивание → мутация → добавить потомка
5. Повторить с шага 1
```

### Операторы

| Оператор     | Метод                       | Описание                                                    |
|--------------|-----------------------------|-------------------------------------------------------------|
| Отбор        | Рулетка / Турнир            | Рулетка: вероятность ∝ приспособленности. Турнир: лучший из k=3 случайных |
| Скрещивание  | Одноточечное                | Случайная точка разреза; обмен хвостами между 2 родителями  |
| Мутация      | Гауссовский шум             | Каждый ген мутирует с вероятностью p; шум ~ N(0, σ)         |
| Элитизм      | Копирование топ-N           | Лучшие N особей переходят без изменений                     |

---

## 🎮 Управление

| Клавиша  | Действие                              |
|----------|---------------------------------------|
| `SPACE`  | Пауза / Продолжить                    |
| `F`      | Включить/выключить ускоренный режим   |
| `S`      | Показать/скрыть визуализацию сенсоров |
| `+` / `-`| Изменить скорость симуляции (1×–50×)  |
| `R`      | Сбросить эволюцию с начала            |

---

## 📈 Уровни сложности

Игра усложняется каждые 100 пройденных труб. Всего **11 стадий (0–10)**:

| Стадия | Скорость | Зазор (пкс) | Движение труб | Двойные трубы |
|--------|----------|-------------|---------------|---------------|
| 0      | 3.0      | 170         | Нет           | Нет           |
| 1      | 3.5      | 170         | Нет           | Нет           |
| 2      | 3.5      | 170         | Нет           | Нет           |
| 3      | 3.5      | 150         | Да            | Нет           |
| 4      | 4.0      | 140         | Да            | 25%           |
| 5      | 4.0      | 140         | Да            | 25%           |
| 6      | 4.5      | 130         | Да            | 25%           |
| 7      | 4.5      | 130         | Да            | 25%           |
| 8      | 5.0      | 120         | Да            | 40%           |
| 9      | 5.5      | 110         | Да            | 40%           |
| 10     | 5.5      | 110         | Да            | 40%           |

С 3-й стадии трубы движутся вертикально по **синусоидальной траектории**.

---

## ⚙️ Настраиваемые параметры

Все параметры регулируются **слайдерами в GUI** во время работы:

| Параметр            | По умолчанию | Диапазон | Описание                                       |
|---------------------|--------------|----------|------------------------------------------------|
| `population_size`   | 50           | 10–200   | Количество птиц в поколении                    |
| `mutation_rate`     | 0.05         | 0.0–0.5  | Вероятность мутации каждого гена               |
| `mutation_strength` | 0.3          | 0.0–1.0  | Стандартное отклонение гауссовского шума        |
| `crossover_rate`    | 0.7          | 0.0–1.0  | Вероятность скрещивания (иначе — клонирование) |
| `elite_count`       | 2            | 0–10     | Количество лучших особей без изменений          |
| `selection_method`  | roulette     | переключ.| `roulette` (рулетка) или `tournament` (турнир) |

---

## 🚀 Запуск

### Требования

- Python **3.11+**
- pip

### Установка

```bash
# 1. Клонировать репозиторий
git clone https://github.com/0Eaggle0/Flappy_Bird_with_a_Genetic_Algorithm.git
cd Flappy_Bird_with_a_Genetic_Algorithm

# 2. Создать и активировать виртуальное окружение
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# 3. Установить зависимости
pip install pygame numpy
```

### Запуск

```bash
# Интерактивный режим (полный GUI)
python main.py

# Безголовый режим (без окна, статистика в консоли)
python test_headless.py
```

---

## 📁 Структура проекта

```
RGR/
├── main.py               # Интерактивное GUI-приложение (точка входа)
├── game.py               # Движок Flappy Bird + стадии сложности
├── genetic_algorithm.py  # ГА: отбор, скрещивание, мутация, элитизм
├── neural_network.py     # Нейронная сеть 5→6→1
├── test_headless.py      # Запуск без отображения (тесты)
└── README.md
```

---

## 📦 Зависимости

| Библиотека | Версия | Назначение                              |
|------------|--------|-----------------------------------------|
| pygame     | 2.6.1  | Рендеринг игры, цикл событий, UI        |
| numpy      | 2.4.4  | Матричные операции для нейронной сети   |

---

## 📚 Как это работает — шаг за шагом

1. **Инициализация** — создаётся 50 птиц, каждая со случайно инициализированной нейронной сетью.
2. **Оценка** — все птицы одновременно играют в Flappy Bird. Птица погибает при столкновении с трубой или границей экрана.
3. **Подсчёт приспособленности** — каждая птица получает очки за время выживания и количество пройденных труб.
4. **Отбор** — более приспособленные птицы с большей вероятностью становятся родителями.
5. **Скрещивание** — геномы двух родителей разрезаются в случайной точке и рекомбинируются в двух потомков.
6. **Мутация** — к небольшому проценту генов каждого потомка добавляется случайный гауссовский шум.
7. **Элитизм** — топ-2 птицы копируются в следующее поколение без изменений.
8. **Повторение** — новая популяция снова играет, и цикл продолжается. Со временем птицы становятся заметно умнее.

---

## 🎓 Учебный контекст

Проект разработан в рамках курсовой работы:

> **«Проектирование и тестирование программного обеспечения»** — Семестр 4  
> Группа ИСТ-241

Цель работы — практически реализовать и проанализировать генетический алгоритм, применённый к реальной задаче (игровой ИИ), наблюдать за эволюционной динамикой и изучить влияние гиперпараметров ГА на скорость сходимости и итоговую производительность.

---

## 📄 Лицензия

Проект распространяется под лицензией **MIT** — используйте, изменяйте и распространяйте свободно.

---

---

# 🐦 Flappy Bird with Genetic Algorithm

> Neuroevolution demo: a population of birds learns to play Flappy Bird through natural selection — no backpropagation, no training data, pure evolution.

---

## 📖 About

This project combines a classic Flappy Bird game with a **Genetic Algorithm (GA)** to evolve simple neural networks that learn to play the game on their own. Each bird is controlled by a small feedforward neural network. Over generations, the best-performing birds pass their "genes" (network weights) to the next generation through selection, crossover, and mutation — just like biological evolution.

All algorithms are implemented **from scratch** using only NumPy and Pygame — no TensorFlow, PyTorch, or other ML frameworks.

---

## ✨ Features

- 🧬 **Genetic Algorithm** with configurable selection, crossover, mutation, and elitism
- 🧠 **Neural Networks** (5 → 6 → 1 architecture) evolved without backpropagation
- 🎮 **Interactive GUI** — watch evolution happen in real time
- 📊 **Live fitness chart** — tracks best / average / worst score per generation
- 🎛️ **Runtime sliders** — adjust GA parameters without restarting
- ⚡ **Fast-forward mode** — skip rendering and evolve at maximum speed
- 👁️ **Sensor visualization** — see what each bird "sees"
- 📈 **11 difficulty stages** — pipes get faster, narrower, and start moving vertically
- 🖥️ **Headless test mode** — run evolution without any graphics (CI/server friendly)

---

## 🧠 Neural Network Architecture

Each bird is controlled by a **5 → 6 → 1** feedforward neural network:

```
Inputs (5)          Hidden (6)         Output (1)
─────────           ──────────         ──────────
bird_y_rel    ─┐
bird_vy       ─┤─── tanh ───┐
pipe_dist_x   ─┤─── tanh ───┤─── sigmoid ─── JUMP?
gap_center_y  ─┤─── tanh ───┘
gap_vy        ─┘
```

| Layer  | Size | Activation |
|--------|------|------------|
| Input  | 5    | —          |
| Hidden | 6    | tanh       |
| Output | 1    | sigmoid    |

**Genome** — a flat array of **43 floats** (all weights and biases):

```
W1 (5×6 = 30)  +  b1 (6)  +  W2 (6×1 = 6)  +  b2 (1)  =  43 genes
```

---

## 🧬 Genetic Algorithm

### Fitness Function

| Event                            | Points              |
|----------------------------------|---------------------|
| Each frame survived              | +1                  |
| Each frame near gap center       | +0.5                |
| Each pipe passed                 | +25                 |
| Finishing 1000 pipes (speedrun)  | +500 × speed bonus  |

### Evolution Cycle

```
1. Run generation → evaluate all birds in game
2. Sort by fitness
3. Copy top N birds unchanged (elitism)
4. Fill rest: select 2 parents → crossover → mutate → add child
5. Repeat from step 1
```

### Operators

| Operator   | Method                | Details                                               |
|------------|-----------------------|-------------------------------------------------------|
| Selection  | Roulette / Tournament | Roulette: proportional to fitness. Tournament: best of k=3 |
| Crossover  | Single-point          | Random cut; swap tails between 2 parents              |
| Mutation   | Gaussian noise        | Each gene mutates with probability p; noise ~ N(0, σ) |
| Elitism    | Top-N copy            | Best N individuals survive unchanged                  |

---

## 🎮 Keyboard Controls

| Key      | Action                        |
|----------|-------------------------------|
| `SPACE`  | Pause / Resume                |
| `F`      | Toggle fast-forward mode      |
| `S`      | Toggle sensor visualization   |
| `+` / `-`| Increase / decrease sim speed (1×–50×) |
| `R`      | Reset evolution from scratch  |

---

## 📈 Difficulty Stages

| Stage | Speed | Gap (px) | Moving Pipes | Double Pipes |
|-------|-------|----------|--------------|--------------|
| 0     | 3.0   | 170      | No           | No           |
| 1     | 3.5   | 170      | No           | No           |
| 2     | 3.5   | 170      | No           | No           |
| 3     | 3.5   | 150      | Yes          | No           |
| 4     | 4.0   | 140      | Yes          | 25% chance   |
| 5     | 4.0   | 140      | Yes          | 25% chance   |
| 6     | 4.5   | 130      | Yes          | 25% chance   |
| 7     | 4.5   | 130      | Yes          | 25% chance   |
| 8     | 5.0   | 120      | Yes          | 40% chance   |
| 9     | 5.5   | 110      | Yes          | 40% chance   |
| 10    | 5.5   | 110      | Yes          | 40% chance   |

---

## ⚙️ Configurable Parameters

| Parameter          | Default | Range    | Description                                      |
|--------------------|---------|----------|--------------------------------------------------|
| `population_size`  | 50      | 10–200   | Number of birds per generation                   |
| `mutation_rate`    | 0.05    | 0.0–0.5  | Probability that each gene mutates               |
| `mutation_strength`| 0.3     | 0.0–1.0  | Standard deviation of Gaussian mutation noise    |
| `crossover_rate`   | 0.7     | 0.0–1.0  | Probability of crossover vs. cloning             |
| `elite_count`      | 2       | 0–10     | Number of best birds carried to next generation  |
| `selection_method` | roulette| toggle   | `roulette` or `tournament` selection             |

---

## 🚀 Getting Started

```bash
# Clone
git clone https://github.com/0Eaggle0/Flappy_Bird_with_a_Genetic_Algorithm.git
cd Flappy_Bird_with_a_Genetic_Algorithm

# Setup
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS / Linux
pip install pygame numpy

# Run
python main.py             # Interactive GUI
python test_headless.py    # Headless (no window)
```

---

## 📁 Project Structure

```
RGR/
├── main.py               # Interactive GUI application (entry point)
├── game.py               # Flappy Bird engine + difficulty stages
├── genetic_algorithm.py  # GA: selection, crossover, mutation, elitism
├── neural_network.py     # 5→6→1 feedforward neural network
├── test_headless.py      # Headless test runner
└── README.md
```

---

## 📄 License

MIT License — free to use, modify, and distribute.

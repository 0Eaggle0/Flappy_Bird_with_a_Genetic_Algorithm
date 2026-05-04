import numpy as np
from neural_network import NeuralNetwork


class GeneticAlgorithm:
    """
    Управляет популяцией нейросетей и их эволюцией.
    """

    def __init__(
        self,
        population_size: int = 50,
        mutation_rate: float = 0.05,
        mutation_strength: float = 0.3,
        crossover_rate: float = 0.7,
        elite_count: int = 2,
        selection_method: str = "roulette",
    ):
        """
        :param population_size: размер популяции (количество птичек)
        :param mutation_rate: вероятность мутации каждого гена (0..1)
        :param mutation_strength: сила мутации (амплитуда случайного шума)
        :param crossover_rate: вероятность скрещивания (иначе клонирование)
        :param elite_count: сколько лучших переходит в новое поколение без изменений
        :param selection_method: "roulette" (рулетка) или "tournament" (турнир)
        """
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.crossover_rate = crossover_rate
        self.elite_count = elite_count
        self.selection_method = selection_method

        self.generation = 0
        self.population = []          # список NeuralNetwork
        self.fitness_scores = []      # параллельный список значений фитнеса

        # История для построения графиков (как Рис. 6.11 в учебнике)
        self.history_max = []     # лучший фитнес каждого поколения
        self.history_avg = []     # средний фитнес каждого поколения
        self.history_min = []     # худший фитнес

        self.best_ever_genome = None
        self.best_ever_fitness = -float("inf")

        self._initialize_population()

    # Этап 1. Инициализация
    def _initialize_population(self):
        self.population = [NeuralNetwork() for _ in range(self.population_size)]
        self.fitness_scores = [0.0] * self.population_size

    # Этап 2. Оценка - выполняется снаружи (игрой), здесь только запись
    def set_fitness(self, index: int, fitness: float):
        self.fitness_scores[index] = fitness

    # Этап 3. Отбор
    def _roulette_select(self) -> NeuralNetwork:
        min_fit = min(self.fitness_scores)
        shifted = [f - min_fit + 1e-6 for f in self.fitness_scores]
        total = sum(shifted)

        spin = np.random.uniform(0, total)
        running_sum = 0.0
        for i, fit in enumerate(shifted):
            running_sum += fit
            if running_sum >= spin:
                return self.population[i]

        return self.population[-1]

    def _tournament_select(self, k: int = 3) -> NeuralNetwork:
        idxs = np.random.choice(self.population_size, k, replace=False)
        best_idx = max(idxs, key=lambda i: self.fitness_scores[i])
        return self.population[best_idx]

    def _select_parent(self) -> NeuralNetwork:
        if self.selection_method == "tournament":
            return self._tournament_select()
        return self._roulette_select()

    # Этап 3. Рекомбинация (скрещивание + мутация)
    @staticmethod
    def _crossover(parent_a: NeuralNetwork, parent_b: NeuralNetwork):
        genome_a = parent_a.genome
        genome_b = parent_b.genome
        length = len(genome_a)

        # Выбираем точку разделения
        point = np.random.randint(1, length)

        # Меняем местами "хвосты"
        child_a_genome = np.concatenate([genome_a[:point], genome_b[point:]])
        child_b_genome = np.concatenate([genome_b[:point], genome_a[point:]])

        return NeuralNetwork(child_a_genome), NeuralNetwork(child_b_genome)

    def _mutate(self, network: NeuralNetwork):
        """
        Мутация генома (Рис. 6.8 из учебника).
        Каждый ген с вероятностью mutation_rate получает гауссов шум.
        """
        genome = network.genome
        for i in range(len(genome)):
            if np.random.random() < self.mutation_rate:
                genome[i] += np.random.normal(0, self.mutation_strength)
                # Ограничиваем диапазон, чтобы веса не уходили в бесконечность
                genome[i] = np.clip(genome[i], -3.0, 3.0)
        network._unpack_weights()

    # Главный шаг - смена поколения
    def evolve(self):
        # Сохраняем статистику поколения
        max_fit = max(self.fitness_scores)
        avg_fit = sum(self.fitness_scores) / len(self.fitness_scores)
        min_fit = min(self.fitness_scores)

        self.history_max.append(max_fit)
        self.history_avg.append(avg_fit)
        self.history_min.append(min_fit)

        best_idx = self.fitness_scores.index(max_fit)
        if max_fit > self.best_ever_fitness:
            self.best_ever_fitness = max_fit
            self.best_ever_genome = self.population[best_idx].genome.copy()

        # Сортируем популяцию по фитнесу (для элитизма)
        sorted_indices = sorted(
            range(self.population_size),
            key=lambda i: self.fitness_scores[i],
            reverse=True,
        )

        new_population = []

        # Элитизм: топ-N лучших проходят без изменений
        for i in range(self.elite_count):
            elite_idx = sorted_indices[i]
            new_population.append(self.population[elite_idx].copy())

        # Остальные создаются через отбор + скрещивание + мутацию
        while len(new_population) < self.population_size:
            parent_a = self._select_parent()
            parent_b = self._select_parent()

            if np.random.random() < self.crossover_rate:
                child_a, child_b = self._crossover(parent_a, parent_b)
            else:
                child_a = parent_a.copy()
                child_b = parent_b.copy()

            self._mutate(child_a)
            self._mutate(child_b)

            new_population.append(child_a)
            if len(new_population) < self.population_size:
                new_population.append(child_b)

        self.population = new_population
        self.fitness_scores = [0.0] * self.population_size
        self.generation += 1

    def get_stats(self) -> dict:
        """Возвращает статистику текущего поколения."""
        if not self.history_max:
            return {
                "generation": self.generation,
                "max": 0.0,
                "avg": 0.0,
                "min": 0.0,
                "best_ever": self.best_ever_fitness,
            }
        return {
            "generation": self.generation,
            "max": self.history_max[-1],
            "avg": self.history_avg[-1],
            "min": self.history_min[-1],
            "best_ever": self.best_ever_fitness,
        }

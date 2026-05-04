import numpy as np


class NeuralNetwork:
    INPUT_SIZE = 5
    HIDDEN_SIZE = 6
    OUTPUT_SIZE = 1

    # Общее количество весов (это и есть длина хромосомы)
    GENOME_LENGTH = (
        INPUT_SIZE * HIDDEN_SIZE      # веса вход -> скрытый слой
        + HIDDEN_SIZE                 # смещения скрытого слоя
        + HIDDEN_SIZE * OUTPUT_SIZE   # веса скрытый -> выход
        + OUTPUT_SIZE                 # смещение выхода
    )

    def __init__(self, genome=None):
        if genome is None:
            # Случайная инициализация - начальная популяция
            genome = np.random.uniform(-1.0, 1.0, self.GENOME_LENGTH)

        self.genome = np.array(genome, dtype=np.float64)
        self._unpack_weights()

    def _unpack_weights(self):
        """Разворачивает плоский геном в матрицы весов."""
        idx = 0

        # Веса первого слоя
        size = self.INPUT_SIZE * self.HIDDEN_SIZE
        self.W1 = self.genome[idx:idx + size].reshape(self.INPUT_SIZE, self.HIDDEN_SIZE)
        idx += size

        # Смещения первого слоя
        self.b1 = self.genome[idx:idx + self.HIDDEN_SIZE]
        idx += self.HIDDEN_SIZE

        # Веса второго слоя
        size = self.HIDDEN_SIZE * self.OUTPUT_SIZE
        self.W2 = self.genome[idx:idx + size].reshape(self.HIDDEN_SIZE, self.OUTPUT_SIZE)
        idx += size

        # Смещения второго слоя
        self.b2 = self.genome[idx:idx + self.OUTPUT_SIZE]

    @staticmethod
    def _sigmoid(x):
        """Сигмоидная функция активации."""
        # Защита от переполнения
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def _tanh(x):
        """Гиперболический тангенс - для скрытого слоя."""
        return np.tanh(x)

    def forward(self, inputs):
        """
        Прямое распространение.
        :param inputs: список или массив из 5 нормализованных значений
        :return: True, если нужно прыгнуть
        """
        x = np.array(inputs, dtype=np.float64)

        # Скрытый слой с tanh-активацией
        hidden = self._tanh(x @ self.W1 + self.b1)

        # Выходной слой с сигмоидой
        output = self._sigmoid(hidden @ self.W2 + self.b2)

        return output[0] >= 0.5

    def copy(self):
        """Возвращает копию сети."""
        return NeuralNetwork(self.genome.copy())

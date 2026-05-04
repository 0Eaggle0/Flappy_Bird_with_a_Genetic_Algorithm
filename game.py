import math
import random
import pygame

from neural_network import NeuralNetwork


# ---------------------------------------------------------------------
# Игровые константы
# ---------------------------------------------------------------------
GAME_WIDTH = 600
GAME_HEIGHT = 600

GRAVITY = 0.5
FLAP_STRENGTH = -8.5
MAX_FALL_SPEED = 12

BIRD_X = 100              # X-координата всех птичек (фиксирована)
BIRD_RADIUS = 12

PIPE_WIDTH = 70
PIPE_GAP = 170            # базовый размер прохода (переопределяется стадией)
PIPE_SPEED = 3            # базовая скорость (переопределяется стадией)
PIPE_SPACING = 280        # базовое расстояние между парами труб
PIPE_MIN_TOP = 80         # минимальная высота верхней трубы

FINISH_SCORE = 1000       # финиш — пройти столько труб
DOUBLE_PIPE_SPACING = 170 # фиксированный отступ между трубами в двойной паре

# Параметры сложности по стадиям (каждые 100 труб — следующая стадия)
DIFFICULTY_STAGES = {
    0:  {"speed": 3.0, "gap": 170, "spacing": 280, "move_amp": 0,  "move_speed": 0.00, "double_chance": 0.0},
    1:  {"speed": 3.5, "gap": 170, "spacing": 280, "move_amp": 0,  "move_speed": 0.00, "double_chance": 0.0},
    2:  {"speed": 3.5, "gap": 150, "spacing": 260, "move_amp": 0,  "move_speed": 0.00, "double_chance": 0.0},
    3:  {"speed": 3.5, "gap": 150, "spacing": 260, "move_amp": 30, "move_speed": 0.03, "double_chance": 0.0},
    4:  {"speed": 4.0, "gap": 140, "spacing": 260, "move_amp": 30, "move_speed": 0.03, "double_chance": 0.0},
    5:  {"speed": 4.0, "gap": 140, "spacing": 260, "move_amp": 30, "move_speed": 0.05, "double_chance": 0.25},
    6:  {"speed": 4.5, "gap": 130, "spacing": 240, "move_amp": 30, "move_speed": 0.05, "double_chance": 0.25},
    7:  {"speed": 4.5, "gap": 130, "spacing": 240, "move_amp": 50, "move_speed": 0.05, "double_chance": 0.25},
    8:  {"speed": 5.0, "gap": 120, "spacing": 240, "move_amp": 50, "move_speed": 0.07, "double_chance": 0.40},
    9:  {"speed": 5.5, "gap": 110, "spacing": 240, "move_amp": 50, "move_speed": 0.07, "double_chance": 0.40},
    10: {"speed": 5.5, "gap": 110, "spacing": 240, "move_amp": 50, "move_speed": 0.07, "double_chance": 0.40},
}


def get_stage(score: int) -> int:
    """Возвращает номер стадии сложности по количеству пройденных труб."""
    if score >= FINISH_SCORE:
        return 10
    return min(score // 100, 10)


class Bird:
    """Птичка - игровая особь с нейросетью."""

    def __init__(self, network: NeuralNetwork, color: tuple):
        self.x = BIRD_X
        self.y = GAME_HEIGHT / 2
        self.velocity = 0.0
        self.network = network
        self.color = color
        self.alive = True
        self.fitness = 0.0     # "здоровье" хромосомы
        self.score = 0         # сколько труб прошёл
        self.frames_alive = 0
        self.finish_frame = None  # кадр, когда птица финишировала (1000 труб)

    def think(self, next_pipe):
        """Решает, прыгать или нет, основываясь на состоянии мира."""
        if next_pipe is None:
            return

        # Нормализуем входы в диапазон примерно [-1, 1]
        inputs = [
            (self.y - GAME_HEIGHT / 2) / (GAME_HEIGHT / 2),
            self.velocity / 10.0,
            (next_pipe.x - self.x) / GAME_WIDTH,
            (next_pipe.gap_center() - GAME_HEIGHT / 2) / (GAME_HEIGHT / 2),
            next_pipe.pipe_vy() / max(next_pipe.amplitude, 1),  # 0 для статичных труб
        ]

        if self.network.forward(inputs):
            self.flap()

    def flap(self):
        self.velocity = FLAP_STRENGTH

    def update(self):
        if not self.alive:
            return
        self.velocity += GRAVITY
        if self.velocity > MAX_FALL_SPEED:
            self.velocity = MAX_FALL_SPEED
        self.y += self.velocity
        self.frames_alive += 1
        self.fitness += 1   # фитнес растёт за каждый кадр выживания

    def collides_with_bounds(self) -> bool:
        return self.y - BIRD_RADIUS < 0 or self.y + BIRD_RADIUS > GAME_HEIGHT

    def collides_with_pipe(self, pipe) -> bool:
        # Простая прямоугольная проверка (с небольшим запасом)
        if self.x + BIRD_RADIUS < pipe.x:
            return False
        if self.x - BIRD_RADIUS > pipe.x + PIPE_WIDTH:
            return False
        if self.y - BIRD_RADIUS < pipe.top_height:
            return True
        if self.y + BIRD_RADIUS > pipe.top_height + pipe.gap:
            return True
        return False


class Pipe:
    """Пара труб (верхняя + нижняя) с проходом между ними.
    Поддерживает вертикальное движение (синусоидальное) для сложных стадий."""

    def __init__(self, x: float, params: dict):
        self.x = x
        self.gap = params["gap"]
        self.speed = params["speed"]
        self.amplitude = params["move_amp"]
        self.move_speed_factor = params["move_speed"]
        self.move_phase = random.uniform(0, 2 * math.pi)
        self.passed = False

        min_top = PIPE_MIN_TOP
        max_top = max(min_top, GAME_HEIGHT - self.gap - 80)
        self.base_top = random.randint(min_top, max_top)
        self.top_height = float(self.base_top)

    def update(self):
        self.x -= self.speed
        if self.amplitude > 0:
            self.move_phase += self.move_speed_factor
            new_top = self.base_top + self.amplitude * math.sin(self.move_phase)
            min_top = PIPE_MIN_TOP
            max_top = GAME_HEIGHT - self.gap - 80
            self.top_height = max(min_top, min(max_top, new_top))

    def is_off_screen(self) -> bool:
        return self.x + PIPE_WIDTH < 0

    def gap_center(self) -> float:
        return self.top_height + self.gap / 2

    def pipe_vy(self) -> float:
        """Вертикальная скорость центра прохода в текущем кадре."""
        if self.amplitude == 0:
            return 0.0
        return self.amplitude * self.move_speed_factor * math.cos(self.move_phase)


class FlappyGame:
    """
    Игровой мир со всей популяцией птичек.
    Каждое поколение - один прогон этого мира.
    """

    BIRD_COLORS = [
        (255, 200, 50), (50, 200, 255), (255, 100, 150),
        (150, 255, 100), (200, 150, 255), (255, 150, 50),
    ]

    def __init__(self, networks: list):
        """
        :param networks: список NeuralNetwork - вся популяция
        """
        self.birds = [
            Bird(net, self.BIRD_COLORS[i % len(self.BIRD_COLORS)])
            for i, net in enumerate(networks)
        ]
        self.pipes = []
        self.frame = 0
        self.score = 0          # количество пройденных труб (общий счёт раунда)
        self.finished_count = 0 # сколько птиц финишировало (дошли до 1000 труб)

        # Создаём первые трубы с параметрами стадии 0
        stage0 = DIFFICULTY_STAGES[0]
        first_pipe_x = GAME_WIDTH + 100
        self.pipes.append(Pipe(first_pipe_x, stage0))
        self.pipes.append(Pipe(first_pipe_x + stage0["spacing"], stage0))
        self.pipes.append(Pipe(first_pipe_x + stage0["spacing"] * 2, stage0))

    def _spawn_pipe(self, x: float):
        """Спавнит одну (или две для двойных труб) пары труб."""
        params = DIFFICULTY_STAGES[get_stage(self.score)]
        self.pipes.append(Pipe(x, params))
        if random.random() < params["double_chance"]:
            self.pipes.append(Pipe(x + DOUBLE_PIPE_SPACING, params))

    def get_next_pipe(self, bird: Bird):
        """Находит ближайшую трубу впереди птички."""
        for pipe in self.pipes:
            if pipe.x + PIPE_WIDTH > bird.x - BIRD_RADIUS:
                return pipe
        return None

    def step(self):
        """Один игровой шаг (кадр)."""
        # Обновляем трубы
        for pipe in self.pipes:
            pipe.update()

        # Убираем ушедшие, добавляем новые
        self.pipes = [p for p in self.pipes if not p.is_off_screen()]
        last_x = max(p.x for p in self.pipes) if self.pipes else GAME_WIDTH
        current_spacing = DIFFICULTY_STAGES[get_stage(self.score)]["spacing"]
        if last_x < GAME_WIDTH + current_spacing:
            self._spawn_pipe(last_x + current_spacing)

        # Обновляем птичек
        any_alive = False
        for bird in self.birds:
            if not bird.alive:
                continue
            any_alive = True

            next_pipe = self.get_next_pipe(bird)
            bird.think(next_pipe)
            bird.update()

            # Проверка столкновений
            if bird.collides_with_bounds():
                bird.alive = False
                continue
            for pipe in self.pipes:
                if bird.collides_with_pipe(pipe):
                    bird.alive = False
                    break

            # Бонус фитнеса за близость к центру прохода
            if bird.alive:
                next_pipe = self.get_next_pipe(bird)
                if next_pipe is not None:
                    dist_to_center = abs(bird.y - next_pipe.gap_center())
                    bird.fitness += max(0, 1.0 - dist_to_center / (GAME_HEIGHT / 2)) * 0.5

        # Засчитываем пройденные трубы
        for pipe in self.pipes:
            if not pipe.passed and pipe.x + PIPE_WIDTH < BIRD_X - BIRD_RADIUS:
                pipe.passed = True
                self.score += 1
                for bird in self.birds:
                    if bird.alive:
                        bird.score += 1
                        bird.fitness += 25.0   # большой бонус за пройденную трубу

                        # Детекция финиша
                        if bird.score >= FINISH_SCORE and bird.finish_frame is None:
                            bird.finish_frame = self.frame
                            self.finished_count += 1
                            bird.alive = False  # останавливаем птицу, фиксируем результат

        self.frame += 1
        return any_alive

    def get_alive_count(self) -> int:
        return sum(1 for b in self.birds if b.alive)

    def get_best_alive(self):
        """Возвращает живую птичку с наибольшим фитнесом (для подсветки)."""
        alive = [b for b in self.birds if b.alive]
        if not alive:
            return None
        return max(alive, key=lambda b: b.fitness)

    # -----------------------------------------------------------------
    # Отрисовка
    # -----------------------------------------------------------------
    def draw(self, surface: pygame.Surface, show_sensors: bool = True):
        # Фон - градиент неба
        surface.fill((135, 206, 235))

        # Дальние облака для атмосферы
        for cx, cy, r in [(80, 100, 30), (300, 60, 25), (480, 130, 35), (550, 80, 20)]:
            pygame.draw.circle(surface, (255, 255, 255), (cx, cy), r)

        # Трубы
        for pipe in self.pipes:
            # Цвет труб меняется для движущихся (чуть другой оттенок)
            pipe_color = (60, 160, 60) if pipe.amplitude == 0 else (60, 140, 180)
            hat_color = (40, 130, 40) if pipe.amplitude == 0 else (40, 110, 150)

            # Верхняя труба
            pygame.draw.rect(
                surface, pipe_color,
                (pipe.x, 0, PIPE_WIDTH, pipe.top_height),
            )
            # Нижняя труба
            pygame.draw.rect(
                surface, pipe_color,
                (pipe.x, pipe.top_height + pipe.gap,
                 PIPE_WIDTH, GAME_HEIGHT - pipe.top_height - pipe.gap),
            )
            # "Шляпки" труб
            pygame.draw.rect(
                surface, hat_color,
                (pipe.x - 4, pipe.top_height - 20, PIPE_WIDTH + 8, 20),
            )
            pygame.draw.rect(
                surface, hat_color,
                (pipe.x - 4, pipe.top_height + pipe.gap, PIPE_WIDTH + 8, 20),
            )

        # Подсвечиваем сенсоры лучшей живой птички
        best = self.get_best_alive()
        if show_sensors and best is not None:
            next_pipe = self.get_next_pipe(best)
            if next_pipe is not None:
                # Линия до центра прохода
                pygame.draw.line(
                    surface, (255, 255, 255, 100),
                    (int(best.x), int(best.y)),
                    (int(next_pipe.x + PIPE_WIDTH / 2), int(next_pipe.gap_center())),
                    1,
                )
                # Маркер центра прохода
                pygame.draw.circle(
                    surface, (255, 255, 255),
                    (int(next_pipe.x + PIPE_WIDTH / 2), int(next_pipe.gap_center())),
                    4, 1,
                )

        # Птички (живые поверх мёртвых)
        for bird in self.birds:
            if bird.alive:
                continue
            self._draw_bird(surface, bird, alpha=40)

        for bird in self.birds:
            if not bird.alive:
                continue
            self._draw_bird(surface, bird, alpha=255, is_best=(bird is best))

    @staticmethod
    def _draw_bird(surface, bird: Bird, alpha: int = 255, is_best: bool = False):
        if alpha < 255:
            s = pygame.Surface((BIRD_RADIUS * 2 + 4, BIRD_RADIUS * 2 + 4), pygame.SRCALPHA)
            color = (*bird.color, alpha)
            pygame.draw.circle(s, color, (BIRD_RADIUS + 2, BIRD_RADIUS + 2), BIRD_RADIUS)
            surface.blit(s, (int(bird.x - BIRD_RADIUS - 2), int(bird.y - BIRD_RADIUS - 2)))
        else:
            if is_best:
                pygame.draw.circle(
                    surface, (255, 255, 0),
                    (int(bird.x), int(bird.y)), BIRD_RADIUS + 4, 2,
                )
            pygame.draw.circle(
                surface, bird.color,
                (int(bird.x), int(bird.y)), BIRD_RADIUS,
            )
            # Глаз
            pygame.draw.circle(
                surface, (255, 255, 255),
                (int(bird.x + 4), int(bird.y - 3)), 4,
            )
            pygame.draw.circle(
                surface, (0, 0, 0),
                (int(bird.x + 5), int(bird.y - 3)), 2,
            )

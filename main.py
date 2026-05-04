import sys
import pygame

from neural_network import NeuralNetwork
from genetic_algorithm import GeneticAlgorithm
from game import FlappyGame, GAME_WIDTH, GAME_HEIGHT, get_stage, FINISH_SCORE


# Параметры окна
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 720

PANEL_X = GAME_WIDTH + 20      # X-координата правой панели
PANEL_WIDTH = WINDOW_WIDTH - PANEL_X - 20

CHART_X = PANEL_X
CHART_Y = 420
CHART_WIDTH = PANEL_WIDTH
CHART_HEIGHT = 148

# Цветовая палитра (тёмная тема)
COL_BG = (24, 26, 38)
COL_PANEL = (36, 40, 56)
COL_TEXT = (230, 230, 240)
COL_TEXT_DIM = (140, 145, 165)
COL_ACCENT = (255, 200, 50)
COL_ACCENT2 = (100, 200, 255)
COL_GOOD = (100, 220, 130)
COL_BAD = (240, 100, 110)
COL_BUTTON = (60, 70, 100)
COL_BUTTON_HOVER = (80, 95, 130)


# Простые UI-элементы
class Slider:
    """Горизонтальный слайдер для настройки числовых параметров ГА."""

    def __init__(self, x, y, width, label, min_val, max_val, value, fmt="{:.2f}"):
        self.x = x
        self.y = y
        self.width = width
        self.height = 20
        self.label = label
        self.min_val = min_val
        self.max_val = max_val
        self.value = value
        self.fmt = fmt
        self.dragging = False

    def draw(self, surface, font_label, font_value):
        # Подпись
        label_surf = font_label.render(self.label, True, COL_TEXT_DIM)
        surface.blit(label_surf, (self.x, self.y - 18))

        # Значение
        value_surf = font_value.render(self.fmt.format(self.value), True, COL_ACCENT)
        surface.blit(value_surf, (self.x + self.width - value_surf.get_width(), self.y - 18))

        # Дорожка
        pygame.draw.rect(surface, COL_PANEL,
                         (self.x, self.y + 8, self.width, 4), border_radius=2)

        # Заполненная часть
        ratio = (self.value - self.min_val) / (self.max_val - self.min_val)
        ratio = max(0, min(1, ratio))
        pygame.draw.rect(surface, COL_ACCENT2,
                         (self.x, self.y + 8, int(self.width * ratio), 4),
                         border_radius=2)

        # Ручка
        knob_x = self.x + int(self.width * ratio)
        pygame.draw.circle(surface, COL_TEXT, (knob_x, self.y + 10), 7)
        pygame.draw.circle(surface, COL_ACCENT2, (knob_x, self.y + 10), 5)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = event.pos
            if (self.x <= mx <= self.x + self.width
                    and self.y - 5 <= my <= self.y + 25):
                self.dragging = True
                self._update_from_mouse(mx)
                return True
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self._update_from_mouse(event.pos[0])
            return True
        return False

    def _update_from_mouse(self, mx):
        ratio = (mx - self.x) / self.width
        ratio = max(0, min(1, ratio))
        self.value = self.min_val + ratio * (self.max_val - self.min_val)


class Button:
    """Кнопка."""
    def __init__(self, x, y, width, height, label, action_id):
        self.rect = pygame.Rect(x, y, width, height)
        self.label = label
        self.action_id = action_id
        self.hover = False

    def draw(self, surface, font):
        color = COL_BUTTON_HOVER if self.hover else COL_BUTTON
        pygame.draw.rect(surface, color, self.rect, border_radius=6)
        text_surf = font.render(self.label, True, COL_TEXT)
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.hover = self.rect.collidepoint(event.pos)
        if event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos):
            return self.action_id
        return None


# Главное приложение
class App:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Нейроэволюция Flappy Bird (Генетический алгоритм)")
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()

        # Шрифты
        self.font_huge = pygame.font.SysFont("arial", 28, bold=True)
        self.font_large = pygame.font.SysFont("arial", 20, bold=True)
        self.font_med = pygame.font.SysFont("arial", 16)
        self.font_small = pygame.font.SysFont("arial", 13)
        self.font_tiny = pygame.font.SysFont("arial", 11)

        self.population_size = 50
        self.mutation_rate = 0.05
        self.mutation_strength = 0.3
        self.crossover_rate = 0.7
        self.elite_count = 2

        # UI-элементы
        self._build_ui()

        # Состояние
        self.paused = False
        self.fast_forward = False
        self.show_sensors = True
        self.sim_speed = 1                # сколько шагов симуляции за один кадр
        self.selection_method = "roulette"

        # Создаём ГА и первую игру
        self._reset_evolution()

    def _build_ui(self):
        """Создаёт слайдеры и кнопки."""
        x = PANEL_X + 15
        w = PANEL_WIDTH - 30
        y = 130

        self.sliders = {
            "pop": Slider(x, y, w, "Размер популяции", 10, 200, self.population_size, "{:.0f}"),
            "mut_rate": Slider(x, y + 50, w, "Вероятность мутации",
                               0.0, 0.5, self.mutation_rate, "{:.3f}"),
            "mut_str": Slider(x, y + 100, w, "Сила мутации",
                              0.0, 1.0, self.mutation_strength, "{:.2f}"),
            "cross": Slider(x, y + 150, w, "Вероятность скрещивания",
                            0.0, 1.0, self.crossover_rate, "{:.2f}"),
            "elite": Slider(x, y + 200, w, "Элитизм (топ-N)",
                            0, 10, self.elite_count, "{:.0f}"),
        }

        # Кнопки под слайдерами
        btn_y = 355
        btn_w = (PANEL_WIDTH - 60) // 3
        self.buttons = [
            Button(PANEL_X + 15, btn_y, btn_w, 36, "Reset", "reset"),
            Button(PANEL_X + 15 + btn_w + 15, btn_y, btn_w, 36,
                   "Метод: Рулетка", "selection"),
            Button(PANEL_X + 15 + (btn_w + 15) * 2, btn_y, btn_w, 36,
                   "FF: OFF", "fastforward"),
        ]

    def _reset_evolution(self):
        """Применяет настройки слайдеров и стартует эволюцию заново."""
        self.population_size = int(self.sliders["pop"].value)
        self.mutation_rate = self.sliders["mut_rate"].value
        self.mutation_strength = self.sliders["mut_str"].value
        self.crossover_rate = self.sliders["cross"].value
        self.elite_count = int(self.sliders["elite"].value)

        self.ga = GeneticAlgorithm(
            population_size=self.population_size,
            mutation_rate=self.mutation_rate,
            mutation_strength=self.mutation_strength,
            crossover_rate=self.crossover_rate,
            elite_count=self.elite_count,
            selection_method=self.selection_method,
        )
        self.game = FlappyGame(self.ga.population)

    # Обновление состояния
    def _apply_live_settings(self):
        """Параметры ГА (кроме размера популяции) можно менять на лету."""
        self.ga.mutation_rate = self.sliders["mut_rate"].value
        self.ga.mutation_strength = self.sliders["mut_str"].value
        self.ga.crossover_rate = self.sliders["cross"].value
        self.ga.elite_count = int(self.sliders["elite"].value)

    def _step_simulation(self):
        """Один шаг симуляции; при гибели всех - смена поколения."""
        any_alive = self.game.step()
        if not any_alive:
            self._next_generation()

    def _next_generation(self):
        """Записывает фитнес и запускает эволюцию."""
        # Временной бонус для финишистов: быстрее = больше
        finished = [b for b in self.game.birds if b.finish_frame is not None]
        if finished:
            max_frames = max(b.finish_frame for b in finished)
            for b in finished:
                b.fitness += (max_frames / b.finish_frame) * 500

        for i, bird in enumerate(self.game.birds):
            self.ga.set_fitness(i, bird.fitness)

        self._apply_live_settings()
        self.ga.evolve()

        # Если размер популяции изменили - надо пересоздать ГА
        target_size = int(self.sliders["pop"].value)
        if target_size != self.ga.population_size:
            # Сохраняем историю и лучшего
            history = (self.ga.history_max[:], self.ga.history_avg[:],
                       self.ga.history_min[:])
            best_genome = self.ga.best_ever_genome
            best_fitness = self.ga.best_ever_fitness
            generation = self.ga.generation

            self.ga = GeneticAlgorithm(
                population_size=target_size,
                mutation_rate=self.sliders["mut_rate"].value,
                mutation_strength=self.sliders["mut_str"].value,
                crossover_rate=self.sliders["cross"].value,
                elite_count=int(self.sliders["elite"].value),
                selection_method=self.selection_method,
            )
            self.ga.history_max, self.ga.history_avg, self.ga.history_min = history
            self.ga.best_ever_genome = best_genome
            self.ga.best_ever_fitness = best_fitness
            self.ga.generation = generation

            # Заменяем первую особь лучшей за всю историю (если есть)
            if best_genome is not None:
                self.ga.population[0] = NeuralNetwork(best_genome.copy())

        self.game = FlappyGame(self.ga.population)

    # Отрисовка
    def _draw_game_area(self):
        # Игровая область вписана в окно
        game_surface = pygame.Surface((GAME_WIDTH, GAME_HEIGHT))
        self.game.draw(game_surface, show_sensors=self.show_sensors)

        # Если включена быстрая перемотка - не отрисовываем мир (экономим время)
        if not self.fast_forward:
            self.screen.blit(game_surface, (0, 60))
        else:
            placeholder = pygame.Surface((GAME_WIDTH, GAME_HEIGHT))
            placeholder.fill((20, 25, 40))
            text = self.font_huge.render("FAST FORWARD", True, COL_ACCENT)
            placeholder.blit(text, text.get_rect(center=(GAME_WIDTH / 2, GAME_HEIGHT / 2 - 20)))
            sub = self.font_med.render("отрисовка отключена", True, COL_TEXT_DIM)
            placeholder.blit(sub, sub.get_rect(center=(GAME_WIDTH / 2, GAME_HEIGHT / 2 + 20)))
            self.screen.blit(placeholder, (0, 60))

    def _draw_top_bar(self):
        """Полоса сверху с заголовком и счётом текущего раунда."""
        pygame.draw.rect(self.screen, COL_PANEL, (0, 0, WINDOW_WIDTH, 60))
        title = self.font_large.render(
            "Нейроэволюция Flappy Bird", True, COL_TEXT,
        )
        self.screen.blit(title, (15, 18))

        sub = self.font_small.render(
            "Учебный проект по генетическим алгоритмам · ИСТ-241",
            True, COL_TEXT_DIM,
        )
        self.screen.blit(sub, (15, 40))

        # Счёт текущего раунда (живые/всего, очки, стадия)
        alive = self.game.get_alive_count()
        total = len(self.game.birds)
        stage = get_stage(self.game.score)
        score_text = (f"Живы: {alive}/{total}    "
                      f"Счёт: {self.game.score}    "
                      f"Стадия: {stage}/10")
        score_surf = self.font_med.render(score_text, True, COL_ACCENT)
        self.screen.blit(score_surf, (GAME_WIDTH - score_surf.get_width() - 15, 22))

    def _draw_panel(self):
        # Фон панели
        pygame.draw.rect(self.screen, COL_PANEL,
                         (PANEL_X, 60, PANEL_WIDTH, WINDOW_HEIGHT - 60))

        # Заголовок панели
        header = self.font_large.render("Параметры ГА", True, COL_TEXT)
        self.screen.blit(header, (PANEL_X + 15, 75))

        # Слайдеры
        for slider in self.sliders.values():
            slider.draw(self.screen, self.font_small, self.font_med)

        # График статистики
        self._draw_chart()

        # Статистика числами
        self._draw_stats()

        # Кнопки
        for btn in self.buttons:
            btn.draw(self.screen, self.font_med)

    def _draw_chart(self):
        # Заголовок
        title = self.font_med.render("Здоровье популяции по поколениям",
                                     True, COL_TEXT)
        self.screen.blit(title, (CHART_X, CHART_Y - 22))

        # Рамка графика
        pygame.draw.rect(self.screen, COL_BG,
                         (CHART_X, CHART_Y, CHART_WIDTH, CHART_HEIGHT))
        pygame.draw.rect(self.screen, COL_TEXT_DIM,
                         (CHART_X, CHART_Y, CHART_WIDTH, CHART_HEIGHT), 1)

        history_max = self.ga.history_max
        history_avg = self.ga.history_avg
        history_min = self.ga.history_min

        if not history_max:
            msg = self.font_small.render("Ожидание первого поколения...",
                                         True, COL_TEXT_DIM)
            rect = msg.get_rect(center=(CHART_X + CHART_WIDTH / 2,
                                        CHART_Y + CHART_HEIGHT / 2))
            self.screen.blit(msg, rect)
            return

        n = len(history_max)
        max_val = max(history_max) if max(history_max) > 0 else 1.0
        min_val = min(history_min)
        span = max_val - min_val
        if span < 1e-6:
            span = 1.0

        def to_screen(i, v):
            if n > 1:
                x = CHART_X + 5 + (CHART_WIDTH - 10) * i / (n - 1)
            else:
                x = CHART_X + CHART_WIDTH / 2
            y = CHART_Y + 5 + (CHART_HEIGHT - 10) * (1 - (v - min_val) / span)
            return int(x), int(y)

        # Сетка
        for j in range(1, 4):
            gy = CHART_Y + CHART_HEIGHT * j / 4
            pygame.draw.line(self.screen, (60, 65, 85),
                             (CHART_X + 1, int(gy)),
                             (CHART_X + CHART_WIDTH - 1, int(gy)), 1)

        # Линии
        def draw_series(values, color, width=2):
            if len(values) < 2:
                return
            pts = [to_screen(i, v) for i, v in enumerate(values)]
            pygame.draw.lines(self.screen, color, False, pts, width)

        draw_series(history_min, COL_BAD, 1)
        draw_series(history_avg, COL_ACCENT2, 2)
        draw_series(history_max, COL_GOOD, 2)

        # Легенда
        legend_y = CHART_Y + CHART_HEIGHT + 6
        for i, (label, color) in enumerate([
            ("max", COL_GOOD),
            ("avg", COL_ACCENT2),
            ("min", COL_BAD),
        ]):
            lx = CHART_X + i * 60
            pygame.draw.line(self.screen, color,
                             (lx, legend_y + 6), (lx + 18, legend_y + 6), 2)
            lbl = self.font_tiny.render(label, True, COL_TEXT_DIM)
            self.screen.blit(lbl, (lx + 22, legend_y))

    def _draw_stats(self):
        stats = self.ga.get_stats()

        x = PANEL_X + 15
        y = CHART_Y + CHART_HEIGHT + 28

        stage = get_stage(self.game.score)
        pipes_to_next = 100 - (self.game.score % 100) if stage < 10 else 0

        lines = [
            (f"Поколение: {stats['generation']}", COL_TEXT, self.font_large),
            (f"Лучший фитнес поколения: {stats['max']:.1f}", COL_GOOD, self.font_small),
            (f"Средний фитнес: {stats['avg']:.1f}", COL_ACCENT2, self.font_small),
            (f"Рекорд за всё время: {stats['best_ever']:.1f}", COL_ACCENT, self.font_small),
            (f"Скорость симуляции: x{self.sim_speed}", COL_TEXT_DIM, self.font_small),
            (f"Стадия: {stage}/10", COL_ACCENT2, self.font_small),
            (f"До след. стадии: {pipes_to_next} труб", COL_TEXT_DIM, self.font_small),
            (f"Финишировало: {self.game.finished_count}", COL_GOOD, self.font_small),
        ]
        for text, color, font in lines:
            surf = font.render(text, True, color)
            self.screen.blit(surf, (x, y))
            y += surf.get_height() + 1

    def _draw_finish_banner(self):
        """Баннер «ФИНИШ» когда птицы дошли до 1000 труб."""
        if self.game.finished_count == 0:
            return
        overlay = pygame.Surface((GAME_WIDTH, 70), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 170))
        self.screen.blit(overlay, (0, GAME_HEIGHT // 2 + 30))
        text = self.font_huge.render(
            f"ФИНИШ!  {self.game.finished_count} птиц(ы)  |  Лучший спидран!", True, COL_ACCENT
        )
        self.screen.blit(text, text.get_rect(center=(GAME_WIDTH // 2, GAME_HEIGHT // 2 + 65)))

    def _draw_help(self):
        """Подсказки управления внизу игровой области."""
        y = WINDOW_HEIGHT - 28
        text = ("ПРОБЕЛ — пауза   ·   F — fast forward   ·   "
                "S — сенсоры   ·   +/- — скорость   ·   R — рестарт")
        surf = self.font_small.render(text, True, COL_TEXT_DIM)
        self.screen.blit(surf, (15, y))

    # Главный цикл
    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                # Слайдеры
                for slider in self.sliders.values():
                    slider.handle_event(event)

                # Кнопки
                for btn in self.buttons:
                    action = btn.handle_event(event)
                    if action == "reset":
                        self._reset_evolution()
                    elif action == "selection":
                        if self.selection_method == "roulette":
                            self.selection_method = "tournament"
                            btn.label = "Метод: Турнир"
                        else:
                            self.selection_method = "roulette"
                            btn.label = "Метод: Рулетка"
                        self.ga.selection_method = self.selection_method
                    elif action == "fastforward":
                        self.fast_forward = not self.fast_forward
                        btn.label = "FF: ON" if self.fast_forward else "FF: OFF"

                # Клавиатура
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif event.key == pygame.K_f:
                        self.fast_forward = not self.fast_forward
                        for btn in self.buttons:
                            if btn.action_id == "fastforward":
                                btn.label = "FF: ON" if self.fast_forward else "FF: OFF"
                    elif event.key == pygame.K_s:
                        self.show_sensors = not self.show_sensors
                    elif event.key == pygame.K_r:
                        self._reset_evolution()
                    elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                        self.sim_speed = min(self.sim_speed + 1, 50)
                    elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                        self.sim_speed = max(1, self.sim_speed - 1)

            # Шаги симуляции
            if not self.paused:
                steps = self.sim_speed * (10 if self.fast_forward else 1)
                for _ in range(steps):
                    self._step_simulation()

            # Отрисовка
            self.screen.fill(COL_BG)
            self._draw_game_area()
            self._draw_finish_banner()
            self._draw_top_bar()
            self._draw_panel()
            self._draw_help()

            if self.paused:
                pause_surf = self.font_huge.render("ПАУЗА", True, COL_ACCENT)
                rect = pause_surf.get_rect(center=(GAME_WIDTH / 2, GAME_HEIGHT / 2 + 30))
                self.screen.blit(pause_surf, rect)

            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    App().run()

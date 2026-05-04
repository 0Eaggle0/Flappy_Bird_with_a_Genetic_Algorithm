import sys
import os

# Без дисплея для pygame
os.environ["SDL_VIDEODRIVER"] = "dummy"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from genetic_algorithm import GeneticAlgorithm
from game import FlappyGame


def run_generation(ga: GeneticAlgorithm, max_frames: int = 5000):
    """Прогоняет одно поколение и возвращает (max_fit, avg_fit, max_score)."""
    game = FlappyGame(ga.population)
    frame = 0
    while frame < max_frames:
        if not game.step():
            break
        frame += 1

    for i, bird in enumerate(game.birds):
        ga.set_fitness(i, bird.fitness)

    fits = ga.fitness_scores
    max_score = max(b.score for b in game.birds)
    return max(fits), sum(fits) / len(fits), max_score, frame


def main():
    print("Тест нейроэволюции Flappy Bird")
    print("=" * 60)

    ga = GeneticAlgorithm(
        population_size=50,
        mutation_rate=0.05,
        mutation_strength=0.3,
        crossover_rate=0.7,
        elite_count=2,
    )

    print(f"{'Поколение':<10}{'Max фитнес':<14}{'Avg фитнес':<14}"
          f"{'Max труб':<10}{'Кадров':<10}")
    print("-" * 60)

    for gen in range(30):
        max_f, avg_f, max_s, frames = run_generation(ga)
        print(f"{gen:<10}{max_f:<14.1f}{avg_f:<14.1f}{max_s:<10}{frames:<10}")
        ga.evolve()

    print("-" * 60)
    print(f"Лучший рекорд: {ga.best_ever_fitness:.1f}")


if __name__ == "__main__":
    main()

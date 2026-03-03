# Author: Joseph Lorenzo Bautista
# Class: CSCI 580 - Artificial Intelligence
# Assignment: Assignment #4 - SA + GA

import math
import random
from dataclasses import dataclass
from typing import List, Tuple
from pathlib import Path

import matplotlib.pyplot as plt

# Formatting and saving the figures
plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

# Formatting and saving the images/plots
IMAGES_PATH = Path() / "Assignment_4_Images"
IMAGES_PATH.mkdir(parents=True, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


# Part 1: TSP Utilities
Point = Tuple[float, float]
Tour = List[int]

def make_cities(n: int = 40, seed: int = 7) -> List[Point]:
    # Reproducible 2D city coordinates in [0, 1]x[0, 1].
    rng = random.Random(seed)
    return [(rng.random(), rng.random()) for _ in range(n)]

def dist(a: Point, b: Point) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.hypot(dx, dy)

def tour_length(cities: List[Point], tour: Tour) -> float:
    # Total length of a closed tour.
    n = len(tour)
    total = 0.0
    for i in range(n):
        total += dist(cities[tour[i]], cities[tour[(i + 1) % n]])
    return total

def is_valid_tour(tour: Tour, n: int) -> bool:
    return len(tour) == n and set(tour) == set(range(n))

def random_tour(n: int, rng: random.Random) -> Tour:
    t = list(range(n))
    rng.shuffle(t)
    return t

def nearest_neighbor_tour(cities: List[Point], start: int = 0) -> Tour:
    # Deterministic baseline: greedy nearest neighbor.
    n = len(cities)
    unvisited = set(range(n))
    tour = [start]
    unvisited.remove(start)
    cur = start
    while unvisited:
        nxt = min(unvisited, key=lambda j: dist(cities[cur], cities[j]))
        tour.append(nxt)
        unvisited.remove(nxt)
        cur = nxt
    return tour


# Part 2: Neighborhood Operation (2-Opt)
def two_opt_swap(tour: Tour, i: int, k: int) -> Tour:
    # Return a new tour where the segment [i:k] is reversed.
    return tour[:i] + list(reversed(tour[i:k + 1])) + tour[k + 1:]

def random_two_opt_neighbor(tour: Tour, rng: random.Random) -> Tour:
    # Pick a random 2-opt move.
    n = len(tour)
    i = rng.randrange(0, n - 1)
    k = rng.randrange(i + 1, n)
    return two_opt_swap(tour, i, k)


# Part 3: Visualization Helper
def plot_tour(cities: List[Point], tour: Tour, title: str = "") -> None:
    xs = [cities[i][0] for i in tour] + [cities[tour[0]][0]]
    ys = [cities[i][1] for i in tour] + [cities[tour[0]][1]]

    plt.figure(figsize=(6, 6))
    plt.plot(xs, ys, marker="o")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.show()

def plot_history(series: List[float], title: str, ylabel: str = "Value") -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(series)
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel(ylabel)
    plt.show()

def plot_compare(sa_hist: List[float], ga_hist: List[float]) -> None:
    # Overlay histories (x-axis is not identical meaning; this is a visual comparison).
    plt.figure(figsize=(8, 4))
    plt.plot(sa_hist, label="SA best-so-far")
    plt.plot([i * (len(sa_hist) / max(1, len(ga_hist)-1)) for i in range(len(ga_hist))], ga_hist, label="GA best-by-gen")
    plt.title("SA vs GA: Best tour length over time (scaled x-axis)")
    plt.xlabel("Progress (scaled)")
    plt.ylabel("Length")
    plt.legend()
    plt.show()


# Part 4: Baseline Tour
cities = make_cities(n=40, seed=7)

nn_tour = nearest_neighbor_tour(cities, start=0)
nn_len = tour_length(cities, nn_tour)
print(f"Nearest-Neighbor Length: {nn_len:.4f}")

plot_tour(cities, nn_tour, title=f"Nearest Neighbor (L={nn_len:.4f})")


# Part 5A: Code Skeleton - Simulated Annealing (SA)
@dataclass
class SAConfig:
    iters: int = 20_000
    t0: float = 0.2
    alpha: float = 0.9995
    seed: int = 123
    report_every: int = 2000

def simulated_annealing_tsp(cities: List[Point], init_tour: Tour, cfg: SAConfig) -> Tuple[Tour, float, List[float]]:
    rng = random.Random(cfg.seed)
    n = len(cities)
    assert is_valid_tour(init_tour, n), "init_tour must be a valid permutation"

    cur_tour = init_tour[:]
    cur_len = tour_length(cities, cur_tour)

    best_tour = cur_tour[:]
    best_len = cur_len
    history = [best_len]

    T = cfg.t0

    for it in range(cfg.iters):
        cand_tour = random_two_opt_neighbor(cur_tour, rng)
        cand_len = tour_length(cities, cand_tour)
        delta = cand_len - cur_len

        # TODO: acceptance rule
        if delta <= 0:
            accept = True
        else:
            accept = rng.random() < math.exp(-delta / T)
        
        if accept:
            cur_tour, cur_len = cand_tour, cand_len

        if cur_len < best_len:
            best_tour, best_len = cur_tour[:], cur_len

        history.append(best_len)

        # TODO: cooling schedule
        T = T * cfg.alpha
        T = max(T, 1e-12)

        if cfg.report_every and (it + 1) % cfg.report_every == 0:
            print(f"[SA] iter={it+1:6d}  T={T:.4g}  cur={cur_len:.4f}  best={best_len:.4f}")

    return best_tour, best_len, history

# Part 5B: 
@dataclass
class GAConfig:
    pop_size: int = 200
    generations: int = 400
    tournament_k: int = 5
    crossover_rate: float = 0.9
    mutation_rate: float = 0.2
    elite_size: int = 5
    seed: int = 999
    report_every: int = 50

def tournament_select(pop: List[Tour], lengths: List[float], k: int, rng: random.Random) -> Tour:
    # TODO: tournament selection (minimize length)
    # Randomly selects and picks k individuals and returns the best one which is the shortest tour
    candidates = []
    for _ in range(k):
        idx = rng.randrange(len(pop))
        candidates.append(idx)

    # Finds the winner which is the lowest length
    best = candidates[0]
    for c in candidates:
        if lengths[c] < lengths[best]:
            best = c

    return pop[best][:]

def order_crossover_ox(parent1: Tour, parent2: Tour, rng: random.Random) -> Tour:
    # TODO: order crossover (OX)
    n = len(parent1)

    # Picks two points
    i = rng.randrange(n)
    j = rng.randrange(n)

    if i > j:
        temp = i
        i = j
        j = temp

    # Starts with an empty child
    child = [None] * n

    # Copies the part from parent1 into the child
    for x in range(i, j + 1):
        child[x] = parent1[x]

    # Looks and figures out which cities are already in the child
    used = set()
    for x in range(i, j + 1):
        used.add(parent1[x])

    # Retrieves and gets the remaining cities from parent2 (which keeps the order)
    remaining = []
    for city in parent2:
        if city not in used:
            remaining.append(city)

    # Completes and fills in the empty spots
    r = 0
    for x in range(n):
        if child[x] is None:
            child[x] = remaining[r]
            r += 1
    return child

def mutate_swap(tour: Tour, rng: random.Random) -> Tour:
    # TODO: swap mutation
    # Swaps two random cities in the tour
    t = tour[:]
    i = rng.randrange(len(t))
    j = rng.randrange(len(t))
    
    # Swaps here
    temp = t[i]
    t[i] = t[j]
    t[j] = temp
    return t

def genetic_algorithm_tsp(cities: List[Point], init_seed_tours: List[Tour], cfg: GAConfig) -> Tuple[Tour, float, List[float]]:
    rng = random.Random(cfg.seed)
    n = len(cities)

    pop: List[Tour] = []
    for t in init_seed_tours:
        assert is_valid_tour(t, n)
        pop.append(t[:])
    while len(pop) < cfg.pop_size:
        pop.append(random_tour(n, rng))

    lengths = [tour_length(cities, t) for t in pop]
    best_idx = min(range(len(pop)), key=lambda i: lengths[i])
    best_tour = pop[best_idx][:]
    best_len = lengths[best_idx]
    history = [best_len]

    for gen in range(cfg.generations):
        elite_indices = sorted(range(len(pop)), key=lambda i: lengths[i])[: cfg.elite_size]
        next_pop = [pop[i][:] for i in elite_indices]

        while len(next_pop) < cfg.pop_size:
            p1 = tournament_select(pop, lengths, cfg.tournament_k, rng)
            p2 = tournament_select(pop, lengths, cfg.tournament_k, rng)

            if rng.random() < cfg.crossover_rate:
                child = order_crossover_ox(p1, p2, rng)
            else:
                child = p1[:]

            if rng.random() < cfg.mutation_rate:
                child = mutate_swap(child, rng)

            next_pop.append(child)

        pop = next_pop
        lengths = [tour_length(cities, t) for t in pop]

        gen_best_idx = min(range(len(pop)), key=lambda i: lengths[i])
        gen_best_len = lengths[gen_best_idx]
        if gen_best_len < best_len:
            best_len = gen_best_len
            best_tour = pop[gen_best_idx][:]

        history.append(best_len)

        if cfg.report_every and (gen + 1) % cfg.report_every == 0:
            print(f"[GA] gen={gen+1:4d}  best={best_len:.4f}")

    assert is_valid_tour(best_tour, n)
    return best_tour, best_len, history


# Part 6: Data Collection, Visualization and Comparison

# Instrumented SA that can/is tracks delta E values as well as the current solution.
def simulated_annealing_tsp_instrumented(cities, init_tour, cfg):
    rng = random.Random(cfg.seed)
    n = len(cities)
    assert is_valid_tour(init_tour, n)

    cur_tour = init_tour[:]
    cur_len = tour_length(cities, cur_tour)

    best_tour = cur_tour[:]
    best_len = cur_len
    history = [best_len]

    # Tracks other data/info
    accepted_deltas = []
    rejected_deltas = []
    cur_history     = [cur_len]
    temp_history    = [cfg.t0]
    num_evals       = 0

    T = cfg.t0

    for it in range(cfg.iters):
        cand_tour = random_two_opt_neighbor(cur_tour, rng)
        cand_len = tour_length(cities, cand_tour)
        num_evals += 1
        delta = cand_len - cur_len

        # The acceptance rule
        if delta <= 0:
            accept = True
        else:
            accept = rng.random() < math.exp(-delta / T)

        # Track accepted vs Rejected deltas
        if accept:
            accepted_deltas.append(delta)
            cur_tour, cur_len = cand_tour, cand_len
        else:
            rejected_deltas.append(delta)

        if cur_len < best_len:
            best_tour, best_len = cur_tour[:], cur_len

        history.append(best_len)
        cur_history.append(cur_len)
        temp_history.append(T)

        # Cooling Function
        T = T * cfg.alpha
        T = max(T, 1e-12)

        if cfg.report_every and (it + 1) % cfg.report_every == 0:
            print(f"[SA] iter={it+1:6d}  T={T:.4g}  cur={cur_len:.4f}  best={best_len:.4f}")

    sa_data = {
        "accepted_deltas": accepted_deltas,
        "rejected_deltas": rejected_deltas,
        "cur_history": cur_history,
        "temp_history": temp_history,
        "num_evals": num_evals,
    }
    return best_tour, best_len, history, sa_data


# Instrumented GA that also keeps track of population stats
def genetic_algorithm_tsp_instrumented(cities, init_seed_tours, cfg):
    rng = random.Random(cfg.seed)
    n = len(cities)

    pop = []
    for t in init_seed_tours:
        assert is_valid_tour(t, n)
        pop.append(t[:])
    while len(pop) < cfg.pop_size:
        pop.append(random_tour(n, rng))

    lengths = [tour_length(cities, t) for t in pop]
    best_idx = min(range(len(pop)), key=lambda i: lengths[i])
    best_tour = pop[best_idx][:]
    best_len = lengths[best_idx]
    history = [best_len]

    # Tracking other data/info
    avg_fitness_history = [sum(lengths) / len(lengths)]
    worst_fitness_history = [max(lengths)]
    diversity_history = []
    num_evals = 0

    # Calculates the initial diversity which is the std dev of tour lengths
    std = 0.0
    mean = sum(lengths) / len(lengths)
    for l in lengths:
        std += (l - mean) ** 2
    std = math.sqrt(std / len(lengths))
    diversity_history.append(std)

    # Initial population 
    num_evals += len(pop)

    for gen in range(cfg.generations):
        elite_indices = sorted(range(len(pop)), key=lambda i: lengths[i])[: cfg.elite_size]
        next_pop = [pop[i][:] for i in elite_indices]

        while len(next_pop) < cfg.pop_size:
            p1 = tournament_select(pop, lengths, cfg.tournament_k, rng)
            p2 = tournament_select(pop, lengths, cfg.tournament_k, rng)

            if rng.random() < cfg.crossover_rate:
                child = order_crossover_ox(p1, p2, rng)
            else:
                child = p1[:]

            if rng.random() < cfg.mutation_rate:
                child = mutate_swap(child, rng)

            next_pop.append(child)

        pop = next_pop
        lengths = [tour_length(cities, t) for t in pop]
        num_evals += len(pop)

        # Tracks the avg and worst fitness
        avg_len = sum(lengths) / len(lengths)
        avg_fitness_history.append(avg_len)
        worst_fitness_history.append(max(lengths))

        # Tracks the std dev of the tour lengths
        mean = avg_len
        variance = 0.0
        for l in lengths:
            variance += (l - mean) ** 2
        variance = variance / len(lengths)
        diversity_history.append(math.sqrt(variance))

        gen_best_idx = min(range(len(pop)), key=lambda i: lengths[i])
        gen_best_len = lengths[gen_best_idx]
        if gen_best_len < best_len:
            best_len = gen_best_len
            best_tour = pop[gen_best_idx][:]

        history.append(best_len)

        if cfg.report_every and (gen + 1) % cfg.report_every == 0:
            print(f"[GA] gen={gen+1:4d} best={best_len:.4f} avg={avg_len:.4f} std={diversity_history[-1]:.4f}")

    assert is_valid_tour(best_tour, n)

    ga_data = {
        "avg_fitness_history": avg_fitness_history,
        "worst_fitness_history": worst_fitness_history,
        "diversity_history": diversity_history,
        "num_evals": num_evals,
    }
    return best_tour, best_len, history, ga_data


# Runs both SA and GA instrumented version

print("=" * 60)
print("Running Instrumented Simulated Annealing")
print("=" * 60)
sa_cfg = SAConfig(iters=20_000, t0=0.2, alpha=0.9995, seed=123, report_every=2000)
sa_best_tour, sa_best_len, sa_hist, sa_data = simulated_annealing_tsp_instrumented(cities, nn_tour[:], sa_cfg)

print(f"\nSA Best length: {sa_best_len:.4f}")
print(f"SA Improvement vs NN: {(nn_len - sa_best_len) / nn_len * 100:.2f}%")

assert is_valid_tour(sa_best_tour, len(cities))

plot_tour(cities, sa_best_tour, title=f"SA Best (L={sa_best_len:.4f})")
plot_history(sa_hist, title="SA Best-So-Far Length", ylabel="Length")

print()
print("=" * 60)
print("Running Instrumented Genetic Algorithm")
print("=" * 60)
ga_cfg = GAConfig(pop_size=200, generations=400, tournament_k=5, crossover_rate=0.9,
                   mutation_rate=0.2, elite_size=5, seed=999, report_every=50)
ga_best_tour, ga_best_len, ga_hist, ga_data = genetic_algorithm_tsp_instrumented(cities, init_seed_tours=[nn_tour], cfg=ga_cfg)

print(f"\nGA Best Length: {ga_best_len:.4f}")
print(f"GA Improvement vs NN: {(nn_len - ga_best_len) / nn_len * 100:.2f}%")

plot_tour(cities, ga_best_tour, title=f"GA Best (L={ga_best_len:.4f})")
plot_history(ga_hist, title="GA Best-By-Generation Length", ylabel="Length")


# Visualization for SA and GA

# Figure 1: SA - Histogram of delta E for accepted vs rejected moves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Accepted deltas
ax1.hist(sa_data["accepted_deltas"], bins=50, color="green", alpha=0.7)
ax1.set_title("SA: Accepted Delta Moves")
ax1.set_xlabel("Delta E")
ax1.set_ylabel("Count")
ax1.axvline(x=0, color="red", linestyle="--", label="delta=0")
ax1.legend()

# Rejected deltas
ax2.hist(sa_data["rejected_deltas"], bins=50, color="red", alpha=0.7)
ax2.set_title("SA: Rejected Delta Moves")
ax2.set_xlabel("Delta E")
ax2.set_ylabel("Count")
ax2.axvline(x=0, color="red", linestyle="--", label="delta=0")
ax2.legend()

plt.tight_layout()
save_fig("sa_delta_histogram")
plt.show()

# Figure 2: SA - Current vs Best solution over time
plt.figure(figsize=(10, 4))
plt.plot(sa_data["cur_history"], alpha=0.4, label="Current Solution", color="orange")
plt.plot(sa_hist, label="Best So Far", color="blue", linewidth=2)
plt.title("SA: Current vs Best Solution Over Time")
plt.xlabel("Iterations")
plt.ylabel("Tour Length")
plt.legend()
save_fig("sa_current_vs_best_solution")
plt.show()

# Figure 3: SA - Temperature Schedule
plt.figure(figsize=(10, 4))
plt.plot(sa_data["temp_history"], color="red")
plt.title("SA: Temperature Over Time")
plt.xlabel("Iterations")
plt.ylabel("Temperature")
save_fig("sa_temperature_schedule")
plt.show()

# Figure 4: GA - Population Average Fitness Over Generations
plt.figure(figsize=(10, 4))
plt.plot(ga_data["avg_fitness_history"], label="Avg Fitness", color="orange")
plt.plot(ga_hist, label="Best Fitness", color="blue", linewidth=2)
plt.plot(ga_data["worst_fitness_history"], label="Worst Fitness", color="red", alpha=0.5)
plt.title("GA: Population Fitness Over Generations")
plt.xlabel("Generations")
plt.ylabel("Tour Length")
plt.legend()
save_fig("ga_pop_avg_fitness")
plt.show()

# Figure 5: GA: Diversity (std dev of tour lengths) Over Generations
plt.figure(figsize=(8, 3))
plt.plot(ga_data["diversity_history"], color="purple")
plt.title("GA: Population Diversity (Std Dev of Lengths)")
plt.xlabel("Generations")
plt.ylabel("Standard Deviation")
save_fig("ga_diversity_over_gen")
plt.show()

# Figure 6: SA vs GA Side by Side Comparison
sa_evals = sa_data["num_evals"]
ga_evals = ga_data["num_evals"]
print(f"\nTotal Evaluations - SA: {sa_evals} and GA: {ga_evals}")

# Normalization for both SA and GA
sa_x_norm = [i / max(1, len(sa_hist) - 1) for i in range(len(sa_hist))]
ga_x_norm = [i / max(1, len(ga_hist) - 1) for i in range(len(ga_hist))]

plt.figure(figsize=(10, 5))
plt.plot(sa_x_norm, sa_hist, label=f"SA Best-So-Far (Final = {sa_best_len:.4f})", linewidth=2)
plt.plot(ga_x_norm, ga_hist, label=f"GA Best-So-Far (Final = {ga_best_len:.4f})", linewidth=2)
plt.title("SA vs GA: Best Tour Length (Normalized)")
plt.xlabel("Fraction of Total Evaluations")
plt.ylabel("Tour Length")
plt.legend()
save_fig("sa_vs_ga_comparison")
plt.show()

# Figure 7: Final Tour Plot Side by Side
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

# NN Tour
xs = [cities[i][0] for i in nn_tour] + [cities[nn_tour[0]][0]]
ys = [cities[i][1] for i in nn_tour] + [cities[nn_tour[0]][1]]
ax1.plot(xs, ys, marker="o")
ax1.set_title(f"Nearest Neighbor (L = {nn_len:.4f})")
ax1.set_aspect("equal")

# SA Tour
xs = [cities[i][0] for i in sa_best_tour] + [cities[sa_best_tour[0]][0]]
ys = [cities[i][1] for i in sa_best_tour] + [cities[sa_best_tour[0]][1]]
ax2.plot(xs, ys, marker="o", color="green")
ax2.set_title(f"SA Best (L = {sa_best_len:.4f})")
ax2.set_aspect("equal")

# GA Tour
xs = [cities[i][0] for i in ga_best_tour] + [cities[ga_best_tour[0]][0]]
ys = [cities[i][1] for i in ga_best_tour] + [cities[ga_best_tour[0]][1]]
ax3.plot(xs, ys, marker="o", color="red")
ax3.set_title(f"GA Best (L = {ga_best_len:.4f})")
ax3.set_aspect("equal")

plt.tight_layout()
save_fig("final_tour_plot")
plt.show()


# Grade Check
def grade_check(nn_len, method_len, min_improvement_pct=10.0, label="Method"):
    improvement = (nn_len - method_len) / nn_len * 100.0
    print(f"{label} Length: {method_len:.4f} | Improvement vs NN: {improvement:.2f}%")
    assert improvement >= min_improvement_pct, (
        f"{label} improvement {improvement:.2f}% is below required {min_improvement_pct:.2f}%"
    )
    print(f"✅ {label} passed improvement threshold!")

print("\n" + "=" * 60)
print("Grade Check")
print("=" * 60)
grade_check(nn_len, sa_best_len, min_improvement_pct=10.0, label="SA")
grade_check(nn_len, ga_best_len, min_improvement_pct=1.0, label="GA")

# Summary of the Data/Output
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print(f"NN: {nn_len:.4f} | SA: {sa_best_len:.4f} | GA: {ga_best_len:.4f}")
print()

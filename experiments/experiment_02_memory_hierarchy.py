import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ─────────────────────────────────────────────
# EXPERIMENT 2 — Memory Hierarchy
# Question: How much does memory access pattern affect performance?
# Hardware: CPU only (this is a CPU architecture experiment)
# ─────────────────────────────────────────────

def measure_sequential(matrix, repetitions=5):
    """
    Access every element row by row — left to right, top to bottom.
    This is cache-friendly: each access is next to the previous one.
    The CPU prefetcher predicts this pattern and loads ahead.
    """
    times = []
    rows, cols = matrix.shape

    for _ in range(repetitions):
        start = time.perf_counter()
        total = 0.0
        for i in range(rows):
            for j in range(cols):
                total += matrix[i, j]
        end = time.perf_counter()
        times.append(end - start)

    return min(times), total  # min = best case, removes OS scheduling noise


def measure_strided(matrix, stride=16, repetitions=5):
    """
    Access every 16th element — jumps past the cache line boundary.
    Cache line = 64 bytes = 16 float32 values.
    So every access lands on a new cache line → more misses than sequential.
    """
    times = []
    rows, cols = matrix.shape

    for _ in range(repetitions):
        start = time.perf_counter()
        total = 0.0
        for i in range(0, rows, stride):
            for j in range(0, cols, stride):
                total += matrix[i, j]
        end = time.perf_counter()
        times.append(end - start)

    return min(times), total


def measure_random(matrix, repetitions=5):
    """
    Access elements in a random shuffled order.
    The CPU prefetcher cannot predict the next address.
    Almost every access = cache miss = wait ~100ns for RAM.
    This is the worst case.
    """
    times = []
    rows, cols = matrix.shape
    total_elements = rows * cols

    # Pre-generate random indices OUTSIDE the timing loop
    # We're measuring memory access, not random number generation
    indices = np.random.permutation(total_elements)
    flat = matrix.flatten()

    for _ in range(repetitions):
        start = time.perf_counter()
        total = 0.0
        for idx in indices:
            total += flat[idx]
        end = time.perf_counter()
        times.append(end - start)

    return min(times), total


def run_experiment():
    """
    Run all three access patterns across increasing matrix sizes.
    Watch the gap between sequential and random GROW as matrix exceeds cache size.
    """

    # Matrix sizes to test
    # 100x100   = 40 KB   → fits in L1/L2 cache
    # 500x500   = 1 MB    → fits in L3 cache
    # 1000x1000 = 4 MB    → fits in L3 cache (barely)
    # 2000x2000 = 16 MB   → exceeds L3, spills to RAM
    # 3000x3000 = 36 MB   → fully in RAM territory

    sizes = [100, 500, 1000, 2000, 3000]
    results = {
        'sequential': [],
        'strided': [],
        'random': [],
        'size_mb': []
    }

    print("=" * 60)
    print("EXPERIMENT 2 — Memory Hierarchy")
    print("Measuring how access pattern affects performance")
    print("=" * 60)
    print(f"\n{'Size':>10} {'MB':>8} {'Sequential':>14} {'Strided':>14} {'Random':>14} {'Slowdown':>10}")
    print("-" * 75)

    for size in sizes:
        # Create matrix with float32 (4 bytes per element)
        matrix = np.random.rand(size, size).astype(np.float32)
        size_mb = (matrix.nbytes) / (1024 ** 2)

        # Run all three patterns
        seq_time, _ = measure_sequential(matrix)
        str_time, _ = measure_strided(matrix)
        rnd_time, _ = measure_random(matrix)

        results['sequential'].append(seq_time * 1000)   # convert to ms
        results['strided'].append(str_time * 1000)
        results['random'].append(rnd_time * 1000)
        results['size_mb'].append(round(size_mb, 1))

        slowdown = rnd_time / seq_time if seq_time > 0 else 0

        print(f"{size:>7}x{size:<3} {size_mb:>7.1f}MB "
              f"{seq_time*1000:>12.1f}ms "
              f"{str_time*1000:>12.1f}ms "
              f"{rnd_time*1000:>12.1f}ms "
              f"{slowdown:>8.1f}x")

    print("-" * 75)
    print("\nNote: 'Slowdown' = Random ÷ Sequential")
    print("Watch how slowdown increases as matrix size grows past L3 cache\n")

    return results, sizes


def plot_results(results, sizes):
    """
    Two plots:
    1. Raw time (ms) per access pattern per matrix size
    2. Slowdown ratio (random / sequential) — the cache miss story
    """
    sns.set_theme(style="darkgrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Experiment 2 — Memory Hierarchy: Access Pattern vs Performance",
        fontsize=14, fontweight='bold', y=1.02
    )

    size_labels = [f"{s}×{s}\n({mb}MB)" for s, mb in zip(sizes, results['size_mb'])]
    x = np.arange(len(sizes))
    width = 0.25

    # ── Plot 1: Raw time per pattern ──
    bars1 = ax1.bar(x - width, results['sequential'], width,
                    label='Sequential', color='#2ecc71', alpha=0.85)
    bars2 = ax1.bar(x, results['strided'], width,
                    label='Strided (stride=16)', color='#f39c12', alpha=0.85)
    bars3 = ax1.bar(x + width, results['random'], width,
                    label='Random', color='#e74c3c', alpha=0.85)

    ax1.set_xlabel('Matrix Size', fontsize=11)
    ax1.set_ylabel('Time (ms)', fontsize=11)
    ax1.set_title('Raw Access Time by Pattern', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(size_labels, fontsize=8)
    ax1.legend(fontsize=9)
    ax1.set_yscale('log')  # log scale — random is orders of magnitude slower

    # Annotate cache regions
    ax1.axvline(x=1.5, color='white', linestyle='--', alpha=0.5, linewidth=1)
    ax1.text(0.7, ax1.get_ylim()[1] * 0.6, '← In Cache', color='white',
             fontsize=8, alpha=0.7)
    ax1.text(1.7, ax1.get_ylim()[1] * 0.6, 'RAM territory →', color='white',
             fontsize=8, alpha=0.7)

    # ── Plot 2: Slowdown ratio ──
    slowdown = [r / s if s > 0 else 0
                for r, s in zip(results['random'], results['sequential'])]

    ax2.plot(size_labels, slowdown, 'o-', color='#e74c3c',
             linewidth=2.5, markersize=8, markerfacecolor='white',
             markeredgewidth=2.5, label='Random / Sequential')

    ax2.axhline(y=1, color='#2ecc71', linestyle='--', alpha=0.6, label='No slowdown (baseline)')
    ax2.fill_between(range(len(sizes)), slowdown, 1,
                     alpha=0.15, color='#e74c3c')

    ax2.set_xlabel('Matrix Size', fontsize=11)
    ax2.set_ylabel('Slowdown Factor (×)', fontsize=11)
    ax2.set_title('Cache Miss Penalty: Random vs Sequential', fontsize=12)
    ax2.set_xticks(range(len(sizes)))
    ax2.set_xticklabels(size_labels, fontsize=8)
    ax2.legend(fontsize=9)

    # Annotate the worst point
    max_idx = slowdown.index(max(slowdown))
    ax2.annotate(f'{slowdown[max_idx]:.1f}× slower',
                 xy=(max_idx, slowdown[max_idx]),
                 xytext=(max_idx - 0.8, slowdown[max_idx] * 0.85),
                 fontsize=9, color='#e74c3c',
                 arrowprops=dict(arrowstyle='->', color='#e74c3c'))

    plt.tight_layout()
    plt.savefig('experiment_02_results.png', dpi=150, bbox_inches='tight')
    print("Plot saved as experiment_02_results.png")
    plt.show()


def print_architectural_explanation(results, sizes):
    """
    After seeing the numbers, explain what caused them.
    This is the most important part.
    """
    slowdowns = [r / s if s > 0 else 0
                 for r, s in zip(results['random'], results['sequential'])]

    print("\n" + "=" * 60)
    print("ARCHITECTURAL EXPLANATION")
    print("=" * 60)

    print(f"""
What you just measured:

SMALL MATRICES (100×100 = 40KB → fits in L1/L2 cache):
  Slowdown = {slowdowns[0]:.1f}x
  Why: Even random access hits cache — the whole matrix fits.
  The CPU loaded it once, now everything is warm in cache.

MEDIUM MATRICES (1000×1000 = 4MB → fits in L3 cache):
  Slowdown = {slowdowns[2]:.1f}x
  Why: L3 cache can hold it, but random access starts thrashing.
  Each random jump may land in a different cache line.

LARGE MATRICES (3000×3000 = 36MB → RAM territory):
  Slowdown = {slowdowns[4]:.1f}x
  Why: Matrix doesn't fit in any cache.
  Sequential: CPU prefetcher loads ahead → smooth stream from RAM
  Random: Every access = cache miss = 100ns round trip to RAM
  This is the DRAM latency wall.

THE KEY INSIGHT:
  Sequential access didn't get faster — the CPU just hid the
  latency through prefetching. Random access exposed the true
  cost of going to RAM: ~100ns per access.

CONNECTION TO NEURAL NETWORKS:
  → CNN convolutions: cache-friendly (im2col transform)
  → Transformer attention: random-ish access across n×n matrix
  → This is WHY attention is memory-bandwidth bound at long sequences
  → This is EXACTLY what the Roofline model will quantify in Week 3
""")


if __name__ == "__main__":
    print("Starting Experiment 2 — this will take 2-5 minutes on CPU\n")
    results, sizes = run_experiment()
    print_architectural_explanation(results, sizes)
    plot_results(results, sizes)
    print("\nExperiment 2 complete. Check experiment_02_results.png")
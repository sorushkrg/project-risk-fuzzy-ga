import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import warnings

warnings.filterwarnings('ignore')


class Tee:
    """کلاسی برای نوشتن هم‌زمان خروجی روی ترمینال و داخل فایل متنی"""

    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def run_ga_feature_selection_optimized():
    """
    ✅ Genetic Algorithm برای Feature Selection
    ✅ بهینه و optimized
    ✅ Early stopping
    ✅ Convergence curve
    """
    
    # ✅ Seed برای reproducibility
    np.random.seed(42)
    
    # ---------------------------------------------------------
    # 0. SETUP REPORT DIRECTORY
    # ---------------------------------------------------------
    report_dir = "../../report"
    os.makedirs(report_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_path = os.path.join(report_dir, f"ga_feature_selection_{timestamp}.txt")

    sys.stdout = Tee(report_path)

    print("=" * 70)
    print("--- GENETIC ALGORITHM FEATURE SELECTION (OPTIMIZED) ---")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. LOAD TRAINING DATA
    # ---------------------------------------------------------
    print("\n[STEP 1] Loading training data...")
    print("-" * 70)

    x_train_path = "../../data/processed/X_train.csv"
    y_train_path = "../../data/processed/y_train.csv"

    if not os.path.exists(x_train_path) or not os.path.exists(y_train_path):
        print("[ERROR] Training data not found. Please run data_prep.py first.")
        sys.exit(1)

    X_train = pd.read_csv(x_train_path)
    y_train = pd.read_csv(y_train_path)

    feature_names = X_train.columns.tolist()
    num_features = len(feature_names)

    print(f"✓ Data Loaded Successfully")
    print(f"  X_train shape: {X_train.shape}")
    print(f"  Total Initial Features: {num_features}")
    print(f"  Training samples: {len(X_train)}")

    X_data = X_train.values
    y_data = y_train.values.ravel()

    # ---------------------------------------------------------
    # 2. GA PARAMETERS SETUP (OPTIMIZED)
    # ---------------------------------------------------------
    print("\n[STEP 2] Genetic Algorithm Parameters Setup (Optimized)")
    print("-" * 70)

    population_size = 50  # ✅ بهینه
    generations = 100
    mutation_rate = 0.1  # ✅ 10% mutation
    crossover_rate = 0.8  # ✅ 80% crossover
    elite_size = 5  # ✅ نگاه‌داری بهترین‌ها (Elitism)
    tournament_size = 3  # ✅ Tournament selection

    min_features = 8
    max_features = 15
    early_stopping_patience = 25  # ✅ اضافه شد

    print(f"Population Size: {population_size}")
    print(f"Generations: {generations}")
    print(f"Mutation Rate: {mutation_rate:.1%}")
    print(f"Crossover Rate: {crossover_rate:.1%}")
    print(f"Elite Size (Elitism): {elite_size}")
    print(f"Tournament Size: {tournament_size}")
    print(f"Feature constraints: [{min_features}, {max_features}]")
    print(f"Early Stopping Patience: {early_stopping_patience}")

    # ---------------------------------------------------------
    # 3. FITNESS FUNCTION
    # ---------------------------------------------------------
    def fitness_function(individual):
        """
        محاسبه fitness برای یک chromosome
        individual: binary array (1 = selected, 0 = not selected)
        """
        selected_count = np.sum(individual)

        # ✅ Constraint: حداقل و حداکثر فیچرها
        if selected_count < min_features:
            penalty = 0.5 * (min_features - selected_count) / min_features
            return 0.5 - penalty

        if selected_count > max_features:
            penalty = 0.3 * (selected_count - max_features) / max_features
            return 0.5 - penalty

        # ✅ Evaluate using Decision Tree CV
        selected_indices = np.where(individual == 1)[0]
        X_sel = X_data[:, selected_indices]

        clf = DecisionTreeClassifier(
            random_state=42,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5
        )

        cv_scores = cross_val_score(clf, X_sel, y_data, cv=5, scoring='accuracy')
        base_accuracy = cv_scores.mean()
        complexity_penalty = 0.01 * (selected_count / num_features)

        return base_accuracy - complexity_penalty

    # ---------------------------------------------------------
    # 4. INITIALIZATION - Create Initial Population
    # ---------------------------------------------------------
    print("\n[STEP 3] Initializing population...")
    print("-" * 70)

    population = []
    for _ in range(population_size):
        individual = np.random.randint(2, size=num_features)
        # ✅ Ensure constraints
        while np.sum(individual) < min_features or np.sum(individual) > max_features:
            individual = np.random.randint(2, size=num_features)
        population.append(individual)

    population = np.array(population)
    fitness_values = np.array([fitness_function(ind) for ind in population])

    print(f"✓ Initial population created: {population_size} individuals")
    print(f"✓ Initial best fitness: {np.max(fitness_values):.4f}")

    # ---------------------------------------------------------
    # 5. TOURNAMENT SELECTION
    # ---------------------------------------------------------
    def tournament_selection(fitness_vals, tournament_sz=3):
        """
        Tournament selection برای انتخاب والدین
        """
        idx1 = np.random.randint(len(fitness_vals))
        idx2 = np.random.randint(len(fitness_vals))
        
        if tournament_sz > 2:
            for _ in range(tournament_sz - 2):
                idx_candidate = np.random.randint(len(fitness_vals))
                if fitness_vals[idx_candidate] > fitness_vals[idx1]:
                    idx1 = idx_candidate

        return idx1 if fitness_vals[idx1] > fitness_vals[idx2] else idx2

    # ---------------------------------------------------------
    # 6. CROSSOVER (Uniform Crossover)
    # ---------------------------------------------------------
    def crossover(parent1, parent2):
        """
        Uniform crossover: هر gene از یکی از والدین انتخاب می‌شود
        """
        child1 = np.zeros(num_features, dtype=int)
        child2 = np.zeros(num_features, dtype=int)

        for i in range(num_features):
            if np.random.rand() < 0.5:
                child1[i] = parent1[i]
                child2[i] = parent2[i]
            else:
                child1[i] = parent2[i]
                child2[i] = parent1[i]

        return child1, child2

    # ---------------------------------------------------------
    # 7. MUTATION (Bit Flip Mutation)
    # ---------------------------------------------------------
    def mutate(individual):
        """
        Mutation: تغییر random bit
        """
        mutated = individual.copy()
        for i in range(num_features):
            if np.random.rand() < mutation_rate:
                mutated[i] = 1 - mutated[i]

        # ✅ Ensure constraints بعد از mutation
        while np.sum(mutated) < min_features or np.sum(mutated) > max_features:
            mutated = individual.copy()
            for i in range(num_features):
                if np.random.rand() < mutation_rate:
                    mutated[i] = 1 - mutated[i]

        return mutated

    # ---------------------------------------------------------
    # 8. MAIN GA LOOP (WITH EARLY STOPPING)
    # ---------------------------------------------------------
    print("\n[STEP 4] Starting Genetic Algorithm Optimization...")
    print("-" * 70)

    fitness_history = []
    best_fitness_history = []
    no_improve_count = 0
    best_individual = population[np.argmax(fitness_values)]
    best_fitness = np.max(fitness_values)

    for generation in range(generations):
        # ✅ Elitism: نگاه‌داری بهترین‌ها
        sorted_indices = np.argsort(fitness_values)[::-1]
        elite_indices = sorted_indices[:elite_size]
        elite_population = population[elite_indices].copy()
        elite_fitness = fitness_values[elite_indices].copy()

        # ✅ Create new population
        new_population = []
        new_fitness_values = []

        # اضافه کردن elites
        new_population.extend(elite_population)
        new_fitness_values.extend(elite_fitness)

        # Generate بقیه population
        while len(new_population) < population_size:
            # Selection
            parent1_idx = tournament_selection(fitness_values, tournament_size)
            parent2_idx = tournament_selection(fitness_values, tournament_size)

            parent1 = population[parent1_idx].copy()
            parent2 = population[parent2_idx].copy()

            # Crossover
            if np.random.rand() < crossover_rate:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1, parent2

            # Mutation
            if np.random.rand() < mutation_rate:
                child1 = mutate(child1)
            if np.random.rand() < mutation_rate:
                child2 = mutate(child2)

            # Evaluate
            fitness_child1 = fitness_function(child1)
            fitness_child2 = fitness_function(child2)

            new_population.append(child1)
            new_fitness_values.append(fitness_child1)

            if len(new_population) < population_size:
                new_population.append(child2)
                new_fitness_values.append(fitness_child2)

        # Update population
        population = np.array(new_population[:population_size])
        fitness_values = np.array(new_fitness_values[:population_size])

        # ✅ Track best fitness
        current_best = np.max(fitness_values)
        current_best_idx = np.argmax(fitness_values)

        fitness_history.append(np.mean(fitness_values))
        best_fitness_history.append(current_best)

        if current_best > best_fitness:
            best_fitness = current_best
            best_individual = population[current_best_idx].copy()
            no_improve_count = 0
        else:
            no_improve_count += 1

        # Print progress
        if (generation + 1) % 5 == 0 or generation == 0:
            avg_fitness = np.mean(fitness_values)
            active_features = np.sum(best_individual)
            print(
                f"Generation {generation + 1:03d}/{generations} | Best: {best_fitness:.4f} | Avg: {avg_fitness:.4f} | Features: {active_features}")

        # ✅ Early stopping
        if no_improve_count >= early_stopping_patience:
            print(f"\n>>> Early Stopping: No improvement for {early_stopping_patience} generations")
            print(f">>> Stopping at generation {generation + 1}/{generations}")
            break

    # ---------------------------------------------------------
    # 6. EXTRACT SELECTED FEATURES
    # ---------------------------------------------------------
    print("\n[STEP 5] Extracting selected features...")
    print("-" * 70)

    selected_indices = np.where(best_individual == 1)[0]
    selected_feature_names = [feature_names[i] for i in selected_indices]

    print(f"✓ Final Best Fitness: {best_fitness:.4f}")
    print(f"✓ Total Selected Features: {len(selected_indices)} out of {num_features}")
    print(f"✓ Reduction: {((num_features - len(selected_indices)) / num_features * 100):.1f}%")
    print(f"\nSelected Features:")
    for i, name in enumerate(selected_feature_names, 1):
        print(f"  {i}. {name}")

    # ---------------------------------------------------------
    # 7. SAVE CONVERGENCE CURVE
    # ---------------------------------------------------------
    print("\n[STEP 6] Saving convergence curve...")
    print("-" * 70)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # ✅ Best fitness over generations
    ax1.plot(best_fitness_history, linewidth=2.5, color='#2E86AB', marker='o', markersize=4, markevery=5)
    ax1.set_xlabel('Generation', fontsize=12)
    ax1.set_ylabel('Best Fitness', fontsize=12)
    ax1.set_title('GA Best Fitness Convergence', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # ✅ Average fitness over generations
    ax2.plot(fitness_history, linewidth=2.5, color='#A23B72', marker='s', markersize=4, markevery=5)
    ax2.set_xlabel('Generation', fontsize=12)
    ax2.set_ylabel('Average Population Fitness', fontsize=12)
    ax2.set_title('GA Average Population Fitness', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    convergence_curve_path = os.path.join("../../report", f"ga_convergence_curve_{timestamp}.png")
    plt.savefig(convergence_curve_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Convergence curve saved to: {convergence_curve_path}")

    # ---------------------------------------------------------
    # 8. FILTER X_TRAIN
    # ---------------------------------------------------------
    print("\n[STEP 7] Filtering X_train...")
    print("-" * 70)

    X_train_selected = X_train[selected_feature_names]
    print(f"✓ X_train filtered: {X_train_selected.shape}")

    # ---------------------------------------------------------
    # 9. LOAD AND FILTER X_TEST
    # ---------------------------------------------------------
    print("\n[STEP 8] Loading and filtering X_test...")
    print("-" * 70)

    x_test_path = "../../data/processed/X_test.csv"

    if not os.path.exists(x_test_path):
        print("[ERROR] Test data not found.")
        sys.exit(1)

    X_test = pd.read_csv(x_test_path)
    print(f"✓ X_test loaded: {X_test.shape}")

    X_test_selected = X_test[selected_feature_names]
    print(f"✓ X_test filtered: {X_test_selected.shape}")
    print(f"✅ X_train and X_test both have {len(selected_feature_names)} features now!")

    # ---------------------------------------------------------
    # 10. SAVE FILTERED DATASETS
    # ---------------------------------------------------------
    print("\n[STEP 9] Saving filtered datasets...")
    print("-" * 70)

    train_out_path = "../../data/processed/X_train_selected_ga.csv"
    X_train_selected.to_csv(train_out_path, index=False)
    print(f"✓ Saved: {train_out_path}")

    test_out_path = "../../data/processed/X_test_selected_ga.csv"
    X_test_selected.to_csv(test_out_path, index=False)
    print(f"✓ Saved: {test_out_path}")

    features_out_path = "../../data/processed/selected_feature_names_ga.txt"
    with open(features_out_path, 'w', encoding='utf-8') as f:
        for i, fname in enumerate(selected_feature_names, 1):
            f.write(f"{i}. {fname}\n")
    print(f"✓ Saved: {features_out_path}")

    # ---------------------------------------------------------
    # 11. SUMMARY
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("--- GENETIC ALGORITHM FEATURE SELECTION (OPTIMIZED) COMPLETED ---")
    print("=" * 70)

    print(f"\n📊 SUMMARY:")
    print(f"  Original features: {num_features}")
    print(f"  Selected features: {len(selected_feature_names)}")
    print(f"  Reduction: {((num_features - len(selected_indices)) / num_features * 100):.1f}%")
    print(f"  Best Fitness: {best_fitness:.4f}")
    print(f"  Total generations: {len(best_fitness_history)}")

    print(f"\n📁 OUTPUT FILES:")
    print(f"  ✓ X_train_selected_ga.csv")
    print(f"  ✓ X_test_selected_ga.csv")
    print(f"  ✓ selected_feature_names_ga.txt")
    print(f"  ✓ ga_convergence_curve_*.png (visualization)")

    print(f"\n✅ Ready for Phase 3: Decision Tree (GA version)!")
    print(f"📝 Report: {report_path}")
    print("=" * 70)


if __name__ == "__main__":
    run_ga_feature_selection_optimized()

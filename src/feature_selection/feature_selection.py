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


def run_bpso_feature_selection_optimized():
    # ✅ اضافه شد: Seed برای reproducibility
    np.random.seed(42)

    # ---------------------------------------------------------
    # 0. SETUP REPORT DIRECTORY
    # ---------------------------------------------------------
    report_dir = "../../report"
    os.makedirs(report_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_path = os.path.join(report_dir, f"bpso_report_optimized_{timestamp}.txt")

    sys.stdout = Tee(report_path)

    print("=" * 70)
    print("--- BPSO FEATURE SELECTION (OPTIMIZED) ---")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. LOAD TRAINING DATA (21 features)
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
    # 2. BPSO PARAMETERS SETUP (OPTIMIZED)
    # ---------------------------------------------------------
    print("\n[STEP 2] BPSO Parameters Setup (Optimized)")
    print("-" * 70)

    num_particles = 50  # ✅ بهبود: 30 → 50
    max_iter = 100
    w = 0.7
    c1 = 2.0
    c2 = 2.0

    min_features = 8
    max_features = 15
    diversity_threshold = 20
    early_stopping_patience = 30  # ✅ اضافه شد: Early stopping
    restart_ratio = 0.5  # ✅ بهبود: 0.7 → 0.5

    print(f"Particles: {num_particles} (بهتر شده: 30 → 50)")
    print(f"Iterations: {max_iter}")
    print(f"Inertia (w): {w}")
    print(f"Cognitive (c1): {c1}")
    print(f"Social (c2): {c2}")
    print(f"Feature constraints: [{min_features}, {max_features}]")
    print(f"Early Stopping Patience: {early_stopping_patience} (اضافه شد)")
    print(f"Restart Ratio: {restart_ratio} (بهتر شده: 0.7 → 0.5)")

    # ---------------------------------------------------------
    # 3. FITNESS FUNCTION
    # ---------------------------------------------------------
    def fitness_function(position):
        selected_count = np.sum(position)

        if selected_count < min_features:
            penalty = 0.5 * (min_features - selected_count) / min_features
            return 0.5 - penalty

        if selected_count > max_features:
            penalty = 0.3 * (selected_count - max_features) / max_features
            return 0.5 - penalty

        selected_indices = np.where(position == 1)[0]
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
    # 4. INITIALIZATION
    # ---------------------------------------------------------
    print("\n[STEP 3] Initializing particles...")
    print("-" * 70)

    particles_pos = np.random.randint(2, size=(num_particles, num_features))
    particles_vel = np.zeros((num_particles, num_features))

    pbest_pos = particles_pos.copy()
    pbest_fitness = np.zeros(num_particles)
    gbest_pos = np.zeros(num_features)
    gbest_fitness = 0.0

    for i in range(num_particles):
        pbest_fitness[i] = fitness_function(particles_pos[i])
        if pbest_fitness[i] > gbest_fitness:
            gbest_fitness = pbest_fitness[i]
            gbest_pos = particles_pos[i].copy()

    print(f"✓ {num_particles} particles initialized")

    # ---------------------------------------------------------
    # 5. MAIN BPSO LOOP (OPTIMIZED WITH EARLY STOPPING)
    # ---------------------------------------------------------
    print("\n[STEP 4] Starting BPSO Optimization...")
    print("-" * 70)

    # ✅ اضافه شد: Fitness history برای convergence curve
    fitness_history = []
    no_improve_count_global = 0  # ✅ اضافه شد: برای early stopping
    diversity_restart_count = 0  # ✅ اضافه شد: شمارنده restart‌ها

    for iteration in range(max_iter):
        prev_best = gbest_fitness

        for i in range(num_particles):
            r1, r2 = np.random.rand(), np.random.rand()
            particles_vel[i] = (w * particles_vel[i] +
                                c1 * r1 * (pbest_pos[i] - particles_pos[i]) +
                                c2 * r2 * (gbest_pos - particles_pos[i]))

            sigmoid_vel = 1 / (1 + np.exp(-np.clip(particles_vel[i], -10, 10)))
            random_probs = np.random.rand(num_features)
            particles_pos[i] = (random_probs < sigmoid_vel).astype(int)

            current_fitness = fitness_function(particles_pos[i])

            if current_fitness > pbest_fitness[i]:
                pbest_fitness[i] = current_fitness
                pbest_pos[i] = particles_pos[i].copy()

                if current_fitness > gbest_fitness:
                    gbest_fitness = current_fitness
                    gbest_pos = particles_pos[i].copy()

        # ✅ اضافه شد: بررسی تغییر fitness
        if gbest_fitness == prev_best:
            no_improve_count_global += 1
        else:
            no_improve_count_global = 0

        # ✅ اضافه شد: Diversity restart (with optimized ratio)
        if no_improve_count_global >= diversity_threshold:
            print(f"\n>>> No improvement for {diversity_threshold} iterations. Restarting particles...")
            restart_count = int(restart_ratio * num_particles)  # ✅ بهتر شده: 0.5 بجای 0.7
            for i in range(restart_count):
                particles_pos[i] = np.random.randint(2, size=num_features)
                particles_vel[i] = np.zeros(num_features)
                pbest_pos[i] = particles_pos[i].copy()
                pbest_fitness[i] = fitness_function(particles_pos[i])
            no_improve_count_global = 0
            diversity_restart_count += 1

        # ✅ اضافه شد: Store fitness history
        fitness_history.append(gbest_fitness)

        # Print progress
        if (iteration + 1) % 5 == 0 or iteration == 0:
            active_features = np.sum(gbest_pos)
            print(
                f"Iteration {iteration + 1:03d}/{max_iter} | Best CV Accuracy: {gbest_fitness:.4f} | Selected Features: {active_features}")

        # ✅ اضافه شد: Early stopping
        if no_improve_count_global >= early_stopping_patience:
            print(f"\n>>> Early Stopping: No improvement for {early_stopping_patience} iterations")
            print(f">>> Stopping at iteration {iteration + 1}/{max_iter}")
            break

    # ---------------------------------------------------------
    # 6. EXTRACT SELECTED FEATURES
    # ---------------------------------------------------------
    print("\n[STEP 5] Extracting selected features...")
    print("-" * 70)

    selected_indices = np.where(gbest_pos == 1)[0]
    selected_feature_names = [feature_names[i] for i in selected_indices]

    print(f"✓ Final Best CV Accuracy: {gbest_fitness:.4f}")
    print(f"✓ Total Selected Features: {len(selected_indices)} out of {num_features}")
    print(f"✓ Reduction: {((num_features - len(selected_indices)) / num_features * 100):.1f}%")
    print(f"✓ Diversity Restarts: {diversity_restart_count}")
    print(f"\nSelected Features:")
    for i, name in enumerate(selected_feature_names, 1):
        print(f"  {i}. {name}")

    # ---------------------------------------------------------
    # 7. SAVE CONVERGENCE CURVE
    # ---------------------------------------------------------
    print("\n[STEP 6] Saving convergence curve...")
    print("-" * 70)

    # ✅ اضافه شد: Plot convergence curve
    plt.figure(figsize=(12, 6))
    plt.plot(fitness_history, linewidth=2.5, color='#2E86AB', marker='o', markersize=4, markevery=5)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Best CV Accuracy', fontsize=12)
    plt.title('BPSO Convergence Curve - Feature Selection', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    convergence_curve_path = os.path.join("../../report", f"convergence_curve_{timestamp}.png")
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
    print(f"✓ X_test loaded: {X_test.shape} (21 features)")

    # ✅ FILTER X_TEST TO SAME SELECTED FEATURES
    X_test_selected = X_test[selected_feature_names]
    print(f"✓ X_test filtered: {X_test_selected.shape}")
    print(f"✅ X_train and X_test both have {len(selected_feature_names)} features now!")

    # ---------------------------------------------------------
    # 10. SAVE FILTERED DATASETS
    # ---------------------------------------------------------
    print("\n[STEP 9] Saving filtered datasets...")
    print("-" * 70)

    train_out_path = "../../data/processed/X_train_selected.csv"
    X_train_selected.to_csv(train_out_path, index=False)
    print(f"✓ Saved: {train_out_path}")

    test_out_path = "../../data/processed/X_test_selected.csv"
    X_test_selected.to_csv(test_out_path, index=False)
    print(f"✓ Saved: {test_out_path}")

    features_out_path = "../../data/processed/selected_feature_names.txt"
    with open(features_out_path, 'w', encoding='utf-8') as f:
        for i, fname in enumerate(selected_feature_names, 1):
            f.write(f"{i}. {fname}\n")
    print(f"✓ Saved: {features_out_path}")

    # ---------------------------------------------------------
    # 11. SUMMARY
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("--- BPSO FEATURE SELECTION (OPTIMIZED) COMPLETED ---")
    print("=" * 70)

    print(f"\n📊 SUMMARY:")
    print(f"  Original features: {num_features}")
    print(f"  Selected features: {len(selected_feature_names)}")
    print(f"  Reduction: {((num_features - len(selected_indices)) / num_features * 100):.1f}%")
    print(f"  CV Accuracy: {gbest_fitness:.4f}")
    print(f"  Total iterations executed: {len(fitness_history)}")
    print(f"  Diversity restarts: {diversity_restart_count}")

    print(f"\n📁 OUTPUT FILES:")
    print(f"  ✓ X_train_selected.csv")
    print(f"  ✓ X_test_selected.csv")
    print(f"  ✓ selected_feature_names.txt")
    print(f"  ✓ convergence_curve_*.png (visualization)")

    print(f"\n✅ Ready for Phase 3: Decision Tree!")
    print(f"📝 Report: {report_path}")
    print("=" * 70)


if __name__ == "__main__":
    run_bpso_feature_selection_optimized()
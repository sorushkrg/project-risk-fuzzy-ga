import os
import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import pygad

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score , matthews_corrcoef

from simpful import FuzzySystem

try:
    from membership_function import (
        calculate_mf_parameters,
        build_simpful_variables,
        build_mfs_from_anchors,
        get_mf_anchors,
    )
except ImportError:
    from membership_builder import (
        calculate_mf_parameters,
        build_simpful_variables,
        build_mfs_from_anchors,
        get_mf_anchors,
    )

# =========================================================
# REUSE FROM PHASE 4 (بدون بازنویسی)
# =========================================================

from fuzzy_system import (
    get_project_paths,
    sanitize_feature_names,
    build_mamdani_output_variable,
    extract_cart_leaf_rules,
    merge_fuzzy_rules,
    build_mamdani_rule_strings,
    evaluate_predictions,
    OUTPUT_VARIABLE,
    CLASSIFICATION_THRESHOLD,
)


# =========================================================
# CONFIGURATION
# =========================================================

RANDOM_STATE = 42

# نسبت جدا شدن Validation از Train
# (Test دست‌نخورده می‌ماند)
VALIDATION_FRACTION = 0.20

# ---------------------------------------------------------
# GA hyperparameters
#
# این اعداد شروع محافظه‌کارانه هستند.
# بعد از تست سرعت روی سیستم خودت قابل افزایش‌اند.
# ---------------------------------------------------------

GA_POPULATION_SIZE = 20
GA_NUM_GENERATIONS = 15
GA_NUM_PARENTS_MATING = 8
GA_MUTATION_PERCENT_GENES = 15
GA_KEEP_ELITISM = 2

# subdivisions پایین‌تر فقط برای سرعت حین جستجو
GA_SEARCH_SUBDIVISIONS = 100

# subdivisions کامل برای ارزیابی نهایی
FINAL_SUBDIVISIONS = 1000

# حداقل فاصله‌ی نسبی مجاز بین anchorهای داخلی
MIN_INTERIOR_GAP_RATIO = 0.02


# =========================================================
# GA-SPECIFIC MAMDANI INFERENCE
#
# نسخه‌ی مشابه run_mamdani_inference در fuzzy_system.py
# ولی با subdivisions قابل‌تنظیم (برای سرعت در GA).
# =========================================================

def ga_mamdani_inference(
    fuzzy_system,
    X,
    feature_names,
    subdivisions,
):

    import io
    from contextlib import redirect_stdout

    scores = []

    for _, row in X.iterrows():

        for feature in feature_names:

            fuzzy_system.set_variable(
                feature,
                float(row[feature]),
            )

        with redirect_stdout(io.StringIO()):

            result = fuzzy_system.Mamdani_inference(
                [OUTPUT_VARIABLE],
                subdivisions=subdivisions,
                ignore_errors=True,
                ignore_warnings=True,
                verbose=False,
            )

        score = result.get(OUTPUT_VARIABLE, None)

        if score is None:
            scores.append(np.nan)
            continue

        try:
            score = float(score)
        except (TypeError, ValueError):
            scores.append(np.nan)
            continue

        scores.append(
            float(np.clip(score / 100.0, 0.0, 1.0))
        )

    return np.asarray(scores, dtype=float)


# =========================================================
# CHROMOSOME DECODING
#
# هر feature فقط 3 ژن دارد (x1, x2, x3 داخلی).
# x0 و x4 (vmin/vmax) ثابت هستند و توسط GA تغییر نمی‌کنند.
# =========================================================

def decode_chromosome(
    solution,
    feature_names,
    fixed_bounds,
):

    anchors = {}

    index = 0

    for feature in feature_names:

        vmin, vmax = fixed_bounds[feature]

        interior = np.sort(
            solution[index:index + 3]
        )

        index += 3

        data_range = float(vmax - vmin)

        min_gap = max(
            data_range * MIN_INTERIOR_GAP_RATIO,
            1e-8,
        )

        x1, x2, x3 = interior

        x1 = float(
            np.clip(x1, vmin + min_gap, vmax - 3 * min_gap)
        )

        x2 = float(
            np.clip(
                max(x2, x1 + min_gap),
                x1 + min_gap,
                vmax - 2 * min_gap,
            )
        )

        x3 = float(
            np.clip(
                max(x3, x2 + min_gap),
                x2 + min_gap,
                vmax - min_gap,
            )
        )

        anchors[feature] = np.array(
            [vmin, x1, x2, x3, vmax],
            dtype=float,
        )

    return anchors


# =========================================================
# MAIN
# =========================================================

def run_ga_optimization():

    paths = get_project_paths()

    # -----------------------------------------------------
    # STEP 1 - Load data (همان مسیر فاز ۴)
    # -----------------------------------------------------

    print("[STEP 1] Loading data...")

    X_train = pd.read_csv(
        os.path.join(paths["data_dir"], "X_train_selected.csv")
    )

    y_train = pd.read_csv(
        os.path.join(paths["data_dir"], "y_train.csv")
    ).values.ravel()

    X_test = pd.read_csv(
        os.path.join(paths["data_dir"], "X_test_selected.csv")
    )

    y_test = pd.read_csv(
        os.path.join(paths["data_dir"], "y_test.csv")
    ).values.ravel()

    feature_names = sanitize_feature_names(
        X_train.columns.tolist()
    )

    X_train.columns = feature_names
    X_test.columns = feature_names

    print(f"✓ X_train: {X_train.shape}")
    print(f"✓ X_test:  {X_test.shape}")

    # -----------------------------------------------------
    # STEP 2 - Validation split (فقط از Train)
    # -----------------------------------------------------

    print("\n[STEP 2] Splitting Train into GA-train / Validation...")

    X_train_ga, X_val, y_train_ga, y_val = train_test_split(
        X_train,
        y_train,
        test_size=VALIDATION_FRACTION,
        stratify=y_train,
        random_state=RANDOM_STATE,
    )

    X_train_ga = X_train_ga.reset_index(drop=True)
    X_val = X_val.reset_index(drop=True)

    print(f"✓ GA-train: {X_train_ga.shape}")
    print(f"✓ Validation: {X_val.shape}")
    print(
        f"✓ Validation class distribution: "
        f"{pd.Series(y_val).value_counts().to_dict()}"
    )

    # -----------------------------------------------------
    # STEP 3 - Load CART
    # -----------------------------------------------------

    print("\n[STEP 3] Loading CART model...")

    tree_model = joblib.load(
        os.path.join(
            paths["models_dir"],
            "decision_tree_model_optimized.pkl",
        )
    )

    # -----------------------------------------------------
    # STEP 4 - Baseline membership functions + FROZEN rules
    #
    # این‌ها فقط یک‌بار ساخته می‌شوند و در طول GA ثابت
    # می‌مانند. GA فقط anchorهای داخلی را عوض می‌کند.
    # -----------------------------------------------------

    print("\n[STEP 4] Building baseline MFs and frozen rule base...")

    baseline_mfs_params = calculate_mf_parameters(
        tree_model,
        X_train_ga,
        feature_names,
    )

    anchors = get_mf_anchors(
        baseline_mfs_params,
        feature_names,
    )

    fixed_bounds = {
        feature: (
            float(anchors[feature][0]),
            float(anchors[feature][4]),
        )
        for feature in feature_names
    }

    raw_rules = extract_cart_leaf_rules(
        tree_model,
        X_train_ga,
        feature_names,
        baseline_mfs_params,
    )

    structured_rules, duplicate_count = merge_fuzzy_rules(
        raw_rules
    )

    structured_rules = [
        rule for rule in structured_rules if rule["conditions"]
    ]

    frozen_fuzzy_rules = build_mamdani_rule_strings(
        structured_rules
    )

    print(f"✓ Frozen rules: {len(frozen_fuzzy_rules)}")

    if not frozen_fuzzy_rules:
        raise RuntimeError("No rules generated — cannot run GA.")

    # -----------------------------------------------------
    # STEP 5 - GA setup
    # -----------------------------------------------------

    print("\n[STEP 5] Setting up GA...")

    num_genes = len(feature_names) * 3

    gene_space = []

    for feature in feature_names:

        vmin, vmax = fixed_bounds[feature]

        for _ in range(3):

            gene_space.append(
                {"low": vmin, "high": vmax}
            )

    def fitness_func(ga_instance, solution, solution_idx):

        candidate_anchors = decode_chromosome(
            solution,
            feature_names,
            fixed_bounds,
        )

        candidate_mfs_params = build_mfs_from_anchors(
            candidate_anchors,
            feature_names,
        )

        linguistic_variables = build_simpful_variables(
            candidate_mfs_params
        )

        FS = FuzzySystem(show_banner=False)

        for feature, ling_var in linguistic_variables.items():
            FS.add_linguistic_variable(feature, ling_var)

        FS.add_linguistic_variable(
            OUTPUT_VARIABLE,
            build_mamdani_output_variable(),
        )

        FS.add_rules(frozen_fuzzy_rules)

        scores = ga_mamdani_inference(
            FS,
            X_val,
            feature_names,
            subdivisions=GA_SEARCH_SUBDIVISIONS,
        )

        valid_mask = np.isfinite(scores)

        if not np.any(valid_mask):
            return -1.0

        y_pred = (
            scores[valid_mask] >= CLASSIFICATION_THRESHOLD
        ).astype(int)

        return float(
            matthews_corrcoef(
                y_val[valid_mask],
                y_pred,
            )
        )

    ga_instance = pygad.GA(
        num_generations=GA_NUM_GENERATIONS,
        num_parents_mating=GA_NUM_PARENTS_MATING,
        fitness_func=fitness_func,
        sol_per_pop=GA_POPULATION_SIZE,
        num_genes=num_genes,
        gene_space=gene_space,
        gene_type=float,
        parent_selection_type="tournament",
        K_tournament=3,
        crossover_type="uniform",
        mutation_type="random",
        mutation_percent_genes=GA_MUTATION_PERCENT_GENES,
        keep_elitism=GA_KEEP_ELITISM,
        random_seed=RANDOM_STATE,
        save_best_solutions=True,
    )

    # -----------------------------------------------------
    # STEP 6 - Run GA
    # -----------------------------------------------------

    print(
        f"\n[STEP 6] Running GA "
        f"(population={GA_POPULATION_SIZE}, "
        f"generations={GA_NUM_GENERATIONS})..."
    )

    ga_instance.run()

    best_solution, best_fitness, _ = ga_instance.best_solution()

    print(
        f"✓ Best validation MCC: "
        f"{best_fitness:.4f}"
    )
    # -----------------------------------------------------
    # STEP 7 - Final evaluation on Test (فقط یک‌بار)
    # -----------------------------------------------------

    print("\n[STEP 7] Final evaluation on Test set...")

    final_anchors = decode_chromosome(
        best_solution,
        feature_names,
        fixed_bounds,
    )

    final_mfs_params = build_mfs_from_anchors(
        final_anchors,
        feature_names,
    )

    final_linguistic_variables = build_simpful_variables(
        final_mfs_params
    )

    FS_final = FuzzySystem(show_banner=False)

    for feature, ling_var in final_linguistic_variables.items():
        FS_final.add_linguistic_variable(feature, ling_var)

    FS_final.add_linguistic_variable(
        OUTPUT_VARIABLE,
        build_mamdani_output_variable(),
    )

    FS_final.add_rules(frozen_fuzzy_rules)

    test_scores = ga_mamdani_inference(
        FS_final,
        X_test,
        feature_names,
        subdivisions=FINAL_SUBDIVISIONS,
    )

    metrics, cm, y_pred, y_eval = evaluate_predictions(
        y_test,
        test_scores,
        CLASSIFICATION_THRESHOLD,
    )

    print("\n" + "=" * 80)
    print("PHASE 5 GA — FINAL TEST RESULTS")
    print("=" * 80)

    for key in [
        "accuracy", "balanced_accuracy", "precision",
        "recall", "f1_score", "mcc", "roc_auc", "pr_auc",
    ]:
        print(f"{key}: {metrics[key]:.4f}")

    # -----------------------------------------------------
    # STEP 8 - Save results
    # -----------------------------------------------------

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    results = {
        "phase": 5,
        "validation_mcc": float(best_fitness),
        "final_test_metrics": metrics,
        "structured_rules": structured_rules,
        "fuzzy_rules": frozen_fuzzy_rules,
        "final_anchors": {
            feature: final_anchors[feature].tolist()
            for feature in feature_names
        },
        "ga_config": {
            "population_size": GA_POPULATION_SIZE,
            "num_generations": GA_NUM_GENERATIONS,
            "num_parents_mating": GA_NUM_PARENTS_MATING,
            "mutation_percent_genes": GA_MUTATION_PERCENT_GENES,
        },
        "fitness_history": [
            float(f) for f in ga_instance.best_solutions_fitness
        ],
    }

    json_path = os.path.join(
        paths["report_dir"],
        f"phase5_ga_results_{timestamp}.json",
    )

    # =========================================================
    # SAVE RULES TO CSV
    # =========================================================

    rules_csv_path = os.path.join(
        paths["report_dir"],
        f"phase5_fuzzy_rules_{timestamp}.csv"
    )

    rules_rows = []

    for idx, rule in enumerate(structured_rules, start=1):
        conditions = rule.get("conditions", {})

        row = {
            "rule_id": idx,
            "output": rule.get("output"),
            "samples": rule.get("samples", 0),
            "weighted_samples": rule.get(
                "weighted_samples",
                0.0
            ),
            "probability": rule.get(
                "probability",
                0.0
            ),
            "weight": rule.get(
                "weight",
                0.0
            ),
            "conditions": "; ".join(
                f"{feature}={term}"
                for feature, term in conditions.items()
            ),
        }

        rules_rows.append(row)

    rules_df = pd.DataFrame(rules_rows)

    rules_df.to_csv(
        rules_csv_path,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f"✓ Rules saved to CSV: {rules_csv_path}"
    )


    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"\n✓ Results saved: {json_path}")


if __name__ == "__main__":
    run_ga_optimization()
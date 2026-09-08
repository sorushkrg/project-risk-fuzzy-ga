import pandas as pd
import numpy as np
import os
import sys
import json
import joblib
import matplotlib.pyplot as plt
from datetime import datetime

from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

import warnings

warnings.filterwarnings('ignore')


# =========================================================
# LOGGING
# =========================================================

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

    def close(self):
        self.log.close()


# =========================================================
# DECISION TREE TRAINING
# =========================================================

def train_decision_tree_optimized():
    """
    ✅ Decision Tree Classification - CART Algorithm (Optimized)
    ✅ نوع درخت: Binary Classification Decision Tree (CART)
    ✅ GridSearchCV برای بهترین پارامترها
    ✅ Threshold Optimization بدون استفاده از Test برای انتخاب Threshold
    ✅ Feature Importance
    ✅ Rule Extraction
    ✅ Visualization
    ✅ Logging با کلاس Tee
    """

    report_dir = "../report"
    models_dir = "../models"

    os.makedirs(report_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_path = os.path.join(
        report_dir,
        f"decision_tree_optimized_{timestamp}.txt"
    )

    # ---------------------------------------------------------
    # شروع Logging
    # ---------------------------------------------------------

    original_stdout = sys.stdout
    tee = Tee(report_path)
    sys.stdout = tee

    try:

        print("=" * 80)
        print("--- DECISION TREE TRAINING & OPTIMIZATION (PHASE 3) - CART ALGORITHM ---")
        print("=" * 80)

        # ---------------------------------------------------------
        # نوع درخت تصمیم گیری
        # ---------------------------------------------------------

        print("\n[INFO] Decision Tree Type:")
        print("-" * 80)

        print("""
🌳 ALGORITHM: CART (Classification and Regression Trees)
────────────────────────────────────────────────────────
• نوع: Binary Classification Decision Tree
• معیار تقسیم: Gini Index یا Entropy
• روش: Greedy (بهترین تقسیم در هر گره)
• خصوصیات:
  ✅ تک درخت (غیر ensemble)
  ✅ قابل تفسیر (Interpretable)
  ✅ قوانین استخراج شدنی
  ✅ برای Fuzzification مناسب

تفاوت Gini vs Entropy:
• Gini: معیار impurity با محاسبات معمولاً ساده‌تر
• Entropy: مبتنی بر Information Gain
• انتخاب نهایی: بر اساس Cross-Validation
""")

        # ---------------------------------------------------------
        # STEP 1: Load Data
        # ---------------------------------------------------------

        print("\n[STEP 1] Loading filtered data...")
        print("-" * 80)

        x_train_path = "../data/processed/X_train_selected.csv"
        x_test_path = "../data/processed/X_test_selected.csv"
        y_train_path = "../data/processed/y_train.csv"
        y_test_path = "../data/processed/y_test.csv"

        required_files = [
            x_train_path,
            x_test_path,
            y_train_path,
            y_test_path
        ]

        missing_files = [
            path for path in required_files
            if not os.path.exists(path)
        ]

        if missing_files:
            print("[ERROR] Required data files not found:")
            for path in missing_files:
                print(f"  - {path}")

            print("\nRun the previous data preparation / feature selection phase first.")
            sys.exit(1)

        X_train_selected = pd.read_csv(x_train_path)
        X_test_selected = pd.read_csv(x_test_path)

        y_train = pd.read_csv(y_train_path).values.ravel()
        y_test = pd.read_csv(y_test_path).values.ravel()

        feature_names = X_train_selected.columns.tolist()

        print("✓ Data Loaded Successfully")
        print(f"  X_train_selected shape: {X_train_selected.shape}")
        print(f"  X_test_selected shape: {X_test_selected.shape}")
        print(f"  Total Features: {len(feature_names)}")
        print(f"  Features: {feature_names}")
        print(
            f"\n  y_train distribution: "
            f"{pd.Series(y_train).value_counts().to_dict()}"
        )
        print(
            f"  y_test distribution: "
            f"{pd.Series(y_test).value_counts().to_dict()}"
        )

        # ---------------------------------------------------------
        # STEP 2: GridSearchCV
        # ---------------------------------------------------------

        print("\n[STEP 2] GridSearchCV - Finding Best Parameters...")
        print("-" * 80)

        # ---------------------------------------------------------
        # Base classifier
        # ---------------------------------------------------------

        base_clf = DecisionTreeClassifier(
            class_weight='balanced',
            random_state=42
        )

        # ---------------------------------------------------------
        # Cost Complexity Pruning Path
        # ---------------------------------------------------------

        path = base_clf.cost_complexity_pruning_path(
            X_train_selected,
            y_train
        )

        ccp_alphas = path.ccp_alphas

        # آخرین alpha معمولاً درخت را به یک root-only tree تبدیل می‌کند.
        # برای جلوگیری از تست غیرضروری آن را حذف می‌کنیم.
        if len(ccp_alphas) > 1:
            ccp_alphas = ccp_alphas[:-1]

        # اگر تعداد alphaها خیلی زیاد باشد، GridSearch بیش از حد سنگین می‌شود.
        # چند مقدار نماینده از بازه alpha انتخاب می‌کنیم.
        max_alpha_values = 30

        if len(ccp_alphas) > max_alpha_values:
            ccp_alphas = np.unique(
                np.quantile(
                    ccp_alphas,
                    np.linspace(0, 1, max_alpha_values)
                )
            )

        # ---------------------------------------------------------
        # Parameter Grid
        # ---------------------------------------------------------

        param_grid = {
            'ccp_alpha': ccp_alphas,

            'max_depth' : [5, 6, 7, 8 ],

            'min_samples_split': [5, 10, 20, 30],

            'min_samples_leaf': [5, 10, 15, 20],

            'criterion': ['gini', 'entropy']
        }
        print("🔍 Parameter Grid:")

        for param, values in param_grid.items():
            print(f"  {param}: {values}")

        total_combinations = 1

        for values in param_grid.values():
            total_combinations *= len(values)

        print(
            f"\n  Total combinations to test: "
            f"{total_combinations}"
        )

        print(
            f"  With 5-fold CV: "
            f"{total_combinations * 5} model trainings"
        )

        # ---------------------------------------------------------
        # Cross Validation Strategy
        # ---------------------------------------------------------

        cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=42
        )

        # ---------------------------------------------------------
        # GridSearchCV
        # ---------------------------------------------------------

        print(
            "\n⏳ Running GridSearchCV "
            "(this may take a few minutes)..."
        )
        print("-" * 80)

        grid_search = GridSearchCV(
            estimator=base_clf,
            param_grid=param_grid,
            scoring='f1',
            cv=cv,
            n_jobs=-1,
            refit=True,
            verbose=1,
            return_train_score=True
        )

        grid_search.fit(
            X_train_selected,
            y_train
        )

        # ---------------------------------------------------------
        # Analyze max_depth performance
        # ---------------------------------------------------------

        cv_results_df = pd.DataFrame(
            grid_search.cv_results_
        )

        depth_analysis = (
            cv_results_df
            .groupby('param_max_depth')
            .agg(
                mean_cv_f1=('mean_test_score', 'max'),
                mean_train_f1=('mean_train_score', 'max')
            )
            .reset_index()
        )

        depth_analysis['f1_gap'] = (
                depth_analysis['mean_train_f1']
                - depth_analysis['mean_cv_f1']
        )

        print("\n[DEPTH ANALYSIS]")
        print("-" * 80)

        print(
            depth_analysis.to_string(index=False)
        )

        print("\n✅ GridSearchCV Completed!")

        print("\n  Best Parameters:")

        for param, value in grid_search.best_params_.items():
            print(f"    • {param}: {value}")

        print(
            f"\n  Best CV F1-Score: "
            f"{grid_search.best_score_:.4f}"
        )

        # ---------------------------------------------------------
        # Overfitting Check: Train F1 vs CV F1
        # ---------------------------------------------------------

        best_index = grid_search.best_index_

        mean_train_f1 = (
            grid_search.cv_results_['mean_train_score'][best_index]
        )

        mean_cv_f1 = (
            grid_search.cv_results_['mean_test_score'][best_index]
        )

        f1_gap = mean_train_f1 - mean_cv_f1

        print("\n[OVERFITTING CHECK]")
        print("-" * 80)

        print(
            f"  Mean Training F1: {mean_train_f1:.4f}"
        )

        print(
            f"  Mean CV F1:       {mean_cv_f1:.4f}"
        )

        print(
            f"  Train-CV Gap:     {f1_gap:.4f}"
        )

        if f1_gap <= 0.03:
            print("  ✓ Overfitting risk: LOW")

        elif f1_gap <= 0.07:
            print("  ⚠ Overfitting risk: MODERATE")

        else:
            print("  ❌ Overfitting risk: HIGH")

        # ---------------------------------------------------------
        # بهترین مدل
        # ---------------------------------------------------------

        best_clf = grid_search.best_estimator_

        print("\n[STEP 3] Training Best Decision Tree...")
        print("-" * 80)

        print("✓ Decision Tree trained with best parameters")
        print(f"✓ Tree depth: {best_clf.get_depth()}")
        print(f"✓ Tree leaf nodes: {best_clf.get_n_leaves()}")

        # ---------------------------------------------------------
        # STEP 4: Threshold Optimization
        # ---------------------------------------------------------
        # مهم:
        # Test نباید برای پیدا کردن threshold استفاده شود.
        # threshold با Out-of-Fold predictions روی Training پیدا می‌شود.
        # سپس Test فقط برای ارزیابی نهایی استفاده خواهد شد.
        # ---------------------------------------------------------

        print("\n[STEP 4] Optimizing Decision Threshold...")
        print("-" * 80)

        oof_proba = cross_val_predict(
            best_clf,
            X_train_selected,
            y_train,
            cv=cv,
            method='predict_proba',
            n_jobs=-1
        )[:, 1]

        best_threshold = 0.5
        best_f1 = -1.0
        threshold_results = []

        for t in np.arange(0.20, 0.901, 0.01):

            y_pred_t = (
                    oof_proba >= t
            ).astype(int)

            f1_t = f1_score(
                y_train,
                y_pred_t,
                zero_division=0
            )

            precision_t = precision_score(
                y_train,
                y_pred_t,
                zero_division=0
            )

            recall_t = recall_score(
                y_train,
                y_pred_t,
                zero_division=0
            )

            threshold_results.append({
                'threshold': round(float(t), 2),
                'f1': float(f1_t),
                'precision': float(precision_t),
                'recall': float(recall_t)
            })

            if f1_t > best_f1:
                best_f1 = f1_t
                best_threshold = float(t)

        threshold_df = pd.DataFrame(threshold_results)

        default_oof_pred = (
                oof_proba >= 0.5
        ).astype(int)

        default_oof_f1 = f1_score(
            y_train,
            default_oof_pred,
            zero_division=0
        )

        print(
            f"✓ Default threshold (0.50) "
            f"OOF F1: {default_oof_f1:.4f}"
        )

        print(
            f"✓ Best threshold found: "
            f"{best_threshold:.2f} "
            f"→ OOF F1: {best_f1:.4f}"
        )

        # ---------------------------------------------------------
        # STEP 5: Model Evaluation on Test Set
        # ---------------------------------------------------------

        print("\n[STEP 5] Model Evaluation on Test Set...")
        print("-" * 80)

        # Test فقط اینجا استفاده می‌شود.
        y_proba = best_clf.predict_proba(
            X_test_selected
        )[:, 1]

        y_pred_default = (
                y_proba >= 0.5
        ).astype(int)

        y_pred = (
                y_proba >= best_threshold
        ).astype(int)

        default_test_f1 = f1_score(
            y_test,
            y_pred_default,
            zero_division=0
        )

        print(
            f"✓ Default threshold (0.50) "
            f"Test F1: {default_test_f1:.4f}"
        )

        print(
            f"✓ Optimized threshold ({best_threshold:.2f}) "
            f"selected from training OOF predictions"
        )

        # ---------------------------------------------------------
        # Metrics
        # ---------------------------------------------------------

        acc = accuracy_score(
            y_test,
            y_pred
        )

        prec = precision_score(
            y_test,
            y_pred,
            zero_division=0
        )

        rec = recall_score(
            y_test,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0
        )

        try:
            auc = roc_auc_score(
                y_test,
                y_proba
            )
        except ValueError:
            auc = 0.0

        print(f"\n{'=' * 80}")
        print("FINAL PERFORMANCE METRICS:")
        print(f"{'=' * 80}")

        print(
            f"Accuracy:  "
            f"{acc:.4f} ({acc * 100:.2f}%)"
        )

        print(
            f"Precision: "
            f"{prec:.4f} ({prec * 100:.2f}%)"
        )

        print(
            f"Recall:    "
            f"{rec:.4f} ({rec * 100:.2f}%)"
        )

        print(
            f"F1-Score:  "
            f"{f1:.4f}"
        )

        print(
            f"ROC-AUC:   "
            f"{auc:.4f}"
        )

        # ---------------------------------------------------------
        # Confusion Matrix
        # ---------------------------------------------------------

        cm = confusion_matrix(
            y_test,
            y_pred,
            labels=[0, 1]
        )

        tn, fp, fn, tp = cm.ravel()

        print("\nConfusion Matrix:")
        print(f"  True Negatives:  {tn}")
        print(f"  False Positives: {fp}")
        print(f"  False Negatives: {fn}")
        print(f"  True Positives:  {tp}")

        specificity = (
            tn / (tn + fp)
            if (tn + fp) > 0
            else 0.0
        )

        sensitivity = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else 0.0
        )

        print("\nAdditional Metrics:")
        print(
            f"  Specificity (True Negative Rate): "
            f"{specificity:.4f}"
        )

        print(
            f"  Sensitivity (True Positive Rate): "
            f"{sensitivity:.4f}"
        )

        print("\nDetailed Classification Report:")

        print(
            classification_report(
                y_test,
                y_pred,
                zero_division=0
            )
        )

        # ---------------------------------------------------------
        # STEP 6: Feature Importance
        # ---------------------------------------------------------

        print("\n[STEP 6] Feature Importance Analysis...")
        print("-" * 80)

        importances = best_clf.feature_importances_

        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances,
            'importance_percent': importances * 100
        }).sort_values(
            'importance',
            ascending=False
        )

        print(
            feature_importance_df.to_string(
                index=False
            )
        )

        importance_path = os.path.join(
            models_dir,
            "feature_importance.csv"
        )

        feature_importance_df.to_csv(
            importance_path,
            index=False
        )

        print(
            f"\n✓ Feature importance saved to: "
            f"{importance_path}"
        )

        # ---------------------------------------------------------
        # STEP 7: Extract Rules
        # ---------------------------------------------------------

        print("\n[STEP 7] Extracting Decision Rules...")
        print("-" * 80)

        tree_rules_text = export_text(
            best_clf,
            feature_names=feature_names
        )

        rules_text_path = os.path.join(
            models_dir,
            "tree_rules_text.txt"
        )

        with open(
                rules_text_path,
                "w",
                encoding="utf-8"
        ) as f:
            f.write(tree_rules_text)

        print(
            f"✓ Text rules saved to: "
            f"{rules_text_path}"
        )

        # ---------------------------------------------------------
        # Structured Rule Extraction
        # ---------------------------------------------------------

        def extract_structured_rules(tree, feature_names):

            tree_ = tree.tree_

            rules = []
            rule_id = 0

            def recurse(node, depth, condition):

                nonlocal rule_id

                if tree_.feature[node] != -2:

                    name = feature_names[
                        tree_.feature[node]
                    ]

                    threshold = tree_.threshold[node]

                    left_cond = (
                        f"{name} <= {threshold:.4f}"
                    )

                    new_left_cond = (
                        left_cond
                        if not condition
                        else f"{condition} AND {left_cond}"
                    )

                    recurse(
                        tree_.children_left[node],
                        depth + 1,
                        new_left_cond
                    )

                    right_cond = (
                        f"{name} > {threshold:.4f}"
                    )

                    new_right_cond = (
                        right_cond
                        if not condition
                        else f"{condition} AND {right_cond}"
                    )

                    recurse(
                        tree_.children_right[node],
                        depth + 1,
                        new_right_cond
                    )

                else:

                    value = tree_.value[node][0]

                    class_index = int(
                        np.argmax(value)
                    )

                    class_pred = tree.classes_[
                        class_index
                    ]

                    value_sum = float(
                        value.sum()
                    )

                    confidence = (
                        float(value[class_index] / value_sum)
                        if value_sum > 0
                        else 0.0
                    )

                    # تعداد واقعی نمونه‌های رسیده به leaf
                    samples = int(
                        tree_.n_node_samples[node]
                    )

                    # احتمال کلاس 1
                    if 1 in tree.classes_:

                        defect_index = list(
                            tree.classes_
                        ).index(1)

                        defect_prob = (
                            float(
                                value[defect_index]
                                / value_sum
                            )
                            if value_sum > 0
                            else 0.0
                        )

                    else:
                        defect_prob = 0.0

                    rules.append({
                        'rule_id': rule_id,
                        'condition': (
                            condition
                            if condition
                            else "TRUE"
                        ),
                        'predicted_class': int(
                            class_pred
                        ),
                        'confidence': confidence,
                        'samples': samples,
                        'defect_prob': defect_prob
                    })

                    rule_id += 1

            recurse(
                0,
                0,
                ""
            )

            return rules

        structured_rules = extract_structured_rules(
            best_clf,
            feature_names
        )

        rules_df = pd.DataFrame(
            structured_rules
        )

        rules_csv_path = os.path.join(
            models_dir,
            "extracted_rules.csv"
        )

        rules_df.to_csv(
            rules_csv_path,
            index=False
        )

        print(
            f"\n✓ Extracted "
            f"{len(rules_df)} rules"
        )

        print(
            f"✓ Rules saved to: "
            f"{rules_csv_path}"
        )

        print("\nFirst 5 Rules:")

        print(
            rules_df.head().to_string(
                index=False
            )
        )

        # ---------------------------------------------------------
        # STEP 8: Save Visualizations
        # ---------------------------------------------------------

        print("\n[STEP 8] Creating Visualizations...")
        print("-" * 80)

        # Threshold Optimization Curve
        fig, (ax1, ax2) = plt.subplots(
            1,
            2,
            figsize=(14, 5)
        )

        # Plot 1: Threshold vs F1-Score

        thresholds = (
            threshold_df['threshold'].values
        )

        f1_scores = (
            threshold_df['f1'].values
        )

        ax1.plot(
            thresholds,
            f1_scores,
            linewidth=2.5,
            marker='o',
            color='#2E86AB'
        )

        ax1.axvline(
            x=best_threshold,
            color='red',
            linestyle='--',
            linewidth=2,
            label=f'Best: {best_threshold:.2f}'
        )

        ax1.set_xlabel(
            'Decision Threshold',
            fontsize=12
        )

        ax1.set_ylabel(
            'F1-Score',
            fontsize=12
        )

        ax1.set_title(
            'Threshold Optimization Curve',
            fontsize=14,
            fontweight='bold'
        )

        ax1.grid(
            True,
            alpha=0.3
        )

        ax1.legend()

        # Plot 2: Feature Importance

        top_features = (
            feature_importance_df.head(8)
        )

        ax2.barh(
            range(len(top_features)),
            top_features[
                'importance_percent'
            ].values,
            color='#A23B72'
        )

        ax2.set_yticks(
            range(len(top_features))
        )

        ax2.set_yticklabels(
            top_features['feature'].values
        )

        ax2.set_xlabel(
            'Importance (%)',
            fontsize=12
        )

        ax2.set_title(
            'Top Features Importance',
            fontsize=14,
            fontweight='bold'
        )

        ax2.grid(
            True,
            alpha=0.3,
            axis='x'
        )

        plt.tight_layout()

        viz_path = os.path.join(
            report_dir,
            f"decision_tree_visualizations_{timestamp}.png"
        )

        plt.savefig(
            viz_path,
            dpi=300,
            bbox_inches='tight'
        )

        plt.close()

        print(
            f"✓ Visualizations saved to: "
            f"{viz_path}"
        )

        # ---------------------------------------------------------
        # STEP 9: Save Model & Configuration
        # ---------------------------------------------------------

        print("\n[STEP 9] Saving Model & Configuration...")
        print("-" * 80)

        model_path = os.path.join(
            models_dir,
            "decision_tree_model_optimized.pkl"
        )

        joblib.dump(
            best_clf,
            model_path
        )

        print(
            f"✓ Model saved to: "
            f"{model_path}"
        )

        # max_depth ممکن است None باشد
        max_depth_value = (
            None
            if best_clf.max_depth is None
            else int(best_clf.max_depth)
        )

        config = {
            'model_info': {
                'model_type':
                    'DecisionTreeClassifier (CART Algorithm)',
                'selected_features':
                    feature_names,
                'feature_count':
                    len(feature_names),
                'training_date':
                    timestamp
            },

            'model_parameters': {
                'max_depth':
                    max_depth_value,
                'min_samples_split':
                    int(best_clf.min_samples_split),
                'min_samples_leaf':
                    int(best_clf.min_samples_leaf),
                'criterion':
                    best_clf.criterion,
                'ccp_alpha':
                    float(best_clf.ccp_alpha),
                'class_weight':
                    'balanced',
                'decision_threshold':
                    float(best_threshold)
            },

            'gridsearch_info': {
                'best_cv_score':
                    float(grid_search.best_score_),
                'cv_folds':
                    5,
                'scoring_metric':
                    'f1_weighted',
                'total_combinations_tested':
                    total_combinations
            },

            'threshold_optimization': {
                'method':
                    'Out-of-Fold predictions on training data',
                'default_threshold':
                    0.5,
                'optimized_threshold':
                    float(best_threshold),
                'best_oof_f1':
                    float(best_f1)
            },

            'model_structure': {
                'tree_depth':
                    int(best_clf.get_depth()),
                'tree_leaf_nodes':
                    int(best_clf.get_n_leaves()),
                'total_rules':
                    len(structured_rules)
            },

            'performance_metrics': {
                'accuracy':
                    float(acc),
                'precision':
                    float(prec),
                'recall':
                    float(rec),
                'f1_score':
                    float(f1),
                'roc_auc':
                    float(auc),
                'specificity':
                    float(specificity),
                'sensitivity':
                    float(sensitivity)
            },

            'confusion_matrix': {
                'true_negatives':
                    int(tn),
                'false_positives':
                    int(fp),
                'false_negatives':
                    int(fn),
                'true_positives':
                    int(tp)
            }
        }

        config_path = os.path.join(
            models_dir,
            "decision_tree_config.json"
        )

        with open(
                config_path,
                "w",
                encoding="utf-8"
        ) as f:

            json.dump(
                config,
                f,
                indent=2,
                ensure_ascii=False
            )

        print(
            f"✓ Configuration saved to: "
            f"{config_path}"
        )

        # ---------------------------------------------------------
        # FINAL SUMMARY
        # ---------------------------------------------------------

        print("\n" + "=" * 80)
        print("--- DECISION TREE OPTIMIZATION COMPLETED ---")
        print("=" * 80)

        print("\n📊 FINAL SUMMARY:")

        print(
            "  Algorithm: "
            "CART (Classification and Regression Trees)"
        )

        print(
            f"  Features: "
            f"{len(feature_names)}"
        )

        print(
            f"  Accuracy: "
            f"{acc:.4f} ({acc * 100:.2f}%)"
        )

        print(
            f"  Precision: "
            f"{prec:.4f} ({prec * 100:.2f}%)"
        )

        print(
            f"  Recall: "
            f"{rec:.4f} ({rec * 100:.2f}%)"
        )

        print(
            f"  F1-Score: "
            f"{f1:.4f}"
        )

        print(
            f"  ROC-AUC: "
            f"{auc:.4f}"
        )

        print(
            f"  Rules extracted: "
            f"{len(structured_rules)}"
        )

        print(
            f"  Tree depth: "
            f"{best_clf.get_depth()}"
        )

        print(
            f"  Decision threshold: "
            f"{best_threshold:.2f}"
        )

        print("\n📁 OUTPUT FILES:")

        print(
            "  ✓ decision_tree_model_optimized.pkl"
        )

        print(
            "  ✓ extracted_rules.csv"
        )

        print(
            "  ✓ tree_rules_text.txt"
        )

        print(
            "  ✓ feature_importance.csv"
        )

        print(
            "  ✓ decision_tree_config.json"
        )

        print(
            "  ✓ decision_tree_visualizations_*.png"
        )

        print(
            "\n✅ Ready for Phase 4: Fuzzification!"
        )

        print(
            f"📝 Report saved to: "
            f"{report_path}"
        )

        print("=" * 80)

    finally:
        # ---------------------------------------------------------
        # Restore stdout and close log file
        # ---------------------------------------------------------

        sys.stdout = original_stdout
        tee.close()


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    train_decision_tree_optimized()
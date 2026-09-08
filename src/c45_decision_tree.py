import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

from datetime import datetime
from collections import Counter


# =========================================================
# LOGGING
# =========================================================

class Tee:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        try:
            self.log.close()
        except Exception:
            pass


# =========================================================
# C4.5 NODE
# =========================================================

class C45Node:
    def __init__(
            self,
            is_leaf=False,
            prediction=None,
            probabilities=None,
            raw_probabilities=None,
            feature=None,
            threshold=None,
            information_gain=0.0,
            gain_ratio=0.0,
            split_information=0.0,
            samples=0,
            weighted_samples=0.0,
            depth=0,
    ):
        self.is_leaf = is_leaf
        self.prediction = prediction
        self.probabilities = probabilities or {}
        self.raw_probabilities = raw_probabilities or {}
        self.feature = feature
        self.threshold = threshold
        self.information_gain = float(information_gain)
        self.gain_ratio = float(gain_ratio)
        self.split_information = float(split_information)
        self.samples = int(samples)
        self.weighted_samples = float(weighted_samples)
        self.depth = int(depth)
        self.left = None
        self.right = None


# =========================================================
# C4.5 DECISION TREE - FROM SCRATCH
# =========================================================

class C45DecisionTree:
    """
    v5 fixes (per the 3-point plan):
        FIX A: Candidate thresholds are no longer restricted to points
               where the class label changes between adjacent sorted
               values. Any two adjacent DISTINCT values are now a valid
               candidate (same search space CART uses).
        FIX B: Wider max_depth / smaller min_samples_leaf so the tree can
               actually split the previously-dominant 5303-sample leaf.
        FIX C: Threshold search is no longer pure-F1 maximization. It only
               considers thresholds whose OOF specificity is >=
               MIN_SPECIFICITY, and picks the best F1 among those.
    """

    MIN_SPECIFICITY = 0.30  # FIX C constraint, tune as needed

    def __init__(
            self,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            min_gain_ratio=1e-7,
            min_information_gain=1e-7,
            pruning=True,
            confidence_factor=0.25,
            class_weight=None,
            random_state=42,
            prediction_threshold=0.5,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_gain_ratio = min_gain_ratio
        self.min_information_gain = min_information_gain
        self.pruning = pruning
        self.confidence_factor = confidence_factor
        self.class_weight = class_weight
        self.random_state = random_state
        self.prediction_threshold = prediction_threshold

        self.root = None
        self.classes_ = None
        self.feature_names_ = None
        self.feature_importances_ = None
        self.n_features_in_ = None
        self.tree_depth_ = 0
        self.n_leaves_ = 0
        self.n_nodes_ = 0
        self.n_nodes_before_pruning_ = 0
        self.n_leaves_before_pruning_ = 0
        self.pruning_changes_ = 0
        self.class_weights_ = None

    def _validate_input(self, X, y=None):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2-dimensional array.")
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or infinite values.")
        if y is not None:
            y = np.asarray(y).ravel()
            if len(X) != len(y):
                raise ValueError("X and y must contain the same number of samples.")
            if len(y) == 0:
                raise ValueError("Training data is empty.")
        return X, y

    def _prepare_class_weights(self):
        self.class_weights_ = {int(cls): 1.0 for cls in self.classes_}
        if self.class_weight is None:
            return
        if self.class_weight == "balanced":
            counts = Counter(self._y_train.tolist())
            n = len(self._y_train)
            k = len(self.classes_)
            for cls in self.classes_:
                count = counts.get(cls, 0)
                self.class_weights_[int(cls)] = (n / (k * count) if count > 0 else 1.0)
            return
        if isinstance(self.class_weight, dict):
            for cls in self.classes_:
                self.class_weights_[int(cls)] = float(self.class_weight.get(int(cls), 1.0))
            return
        raise ValueError("class_weight must be None, 'balanced', or a dictionary.")

    def _sample_weights(self, y):
        return np.asarray([self.class_weights_.get(int(cls), 1.0) for cls in y], dtype=float)

    def _weighted_counts(self, y, sample_weights=None):
        counts = {int(cls): 0.0 for cls in self.classes_}
        if sample_weights is None:
            sample_weights = np.ones(len(y), dtype=float)
        for cls, weight in zip(y, sample_weights):
            counts[int(cls)] = counts.get(int(cls), 0.0) + float(weight)
        return counts

    def entropy(self, y, sample_weights=None):
        if len(y) == 0:
            return 0.0
        counts = self._weighted_counts(y, sample_weights)
        total = sum(counts.values())
        if total <= 0:
            return 0.0
        probabilities = np.asarray([v / total for v in counts.values() if v > 0], dtype=float)
        return float(-np.sum(probabilities * np.log2(probabilities)))

    def class_distribution(self, y, sample_weights=None):
        counts = self._weighted_counts(y, sample_weights)
        total = sum(counts.values())
        if total <= 0:
            return {int(cls): 0.0 for cls in self.classes_}
        return {int(cls): float(counts.get(int(cls), 0.0) / total) for cls in self.classes_}

    def majority_class(self, y, sample_weights=None):
        counts = self._weighted_counts(y, sample_weights)
        return max(counts, key=counts.get)

    @staticmethod
    def _entropy_from_counts(counts):
        counts = np.asarray(counts, dtype=float)
        total = float(np.sum(counts))
        if total <= 0:
            return 0.0
        p = counts[counts > 0] / total
        return float(-np.sum(p * np.log2(p)))

    def _best_split_for_feature(self, X, y, sample_weights, feature_index):
        """FIX A: candidate thresholds = every boundary between two
        DISTINCT adjacent values (like CART), not only class-change
        boundaries. This lets nodes with overlapping/noisy labels still
        be split further instead of stopping early."""
        values = X[:, feature_index]
        order = np.argsort(values, kind="mergesort")
        sv = values[order]
        sy = y[order]
        sw = sample_weights[order]

        classes = self.classes_
        class_to_idx = {int(c): i for i, c in enumerate(classes)}
        k = len(classes)

        total_counts = np.zeros(k, dtype=float)
        for cls, w in zip(sy, sw):
            total_counts[class_to_idx[int(cls)]] += w

        total_weight = float(np.sum(sw))
        if total_weight <= 0:
            return None

        parent_entropy = self._entropy_from_counts(total_counts)
        if parent_entropy <= 0:
            return None

        left_counts = np.zeros(k, dtype=float)
        left_weight = 0.0
        n_left = 0
        best = None

        for i in range(len(sv) - 1):
            cls_idx = class_to_idx[int(sy[i])]
            left_counts[cls_idx] += sw[i]
            left_weight += float(sw[i])
            n_left += 1
            n_right = len(sv) - n_left

            if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                continue
            # FIX A: only skip identical values; no longer require class change.
            if sv[i] == sv[i + 1]:
                continue

            right_counts = total_counts - left_counts
            right_weight = total_weight - left_weight
            if left_weight <= 0 or right_weight <= 0:
                continue

            left_entropy = self._entropy_from_counts(left_counts)
            right_entropy = self._entropy_from_counts(right_counts)
            gain = parent_entropy - (
                (left_weight / total_weight) * left_entropy
                + (right_weight / total_weight) * right_entropy
            )
            if gain <= self.min_information_gain:
                continue

            p_left = left_weight / total_weight
            p_right = right_weight / total_weight
            split_info = float(-(p_left * np.log2(p_left) + p_right * np.log2(p_right)))
            if split_info <= 0:
                continue

            ratio = float(gain / split_info)
            threshold = float((sv[i] + sv[i + 1]) / 2.0)

            candidate = {
                "feature": feature_index, "threshold": threshold, "gain": float(gain),
                "split_info": split_info, "gain_ratio": ratio,
            }
            if best is None or (candidate["gain"], candidate["gain_ratio"]) > (best["gain"], best["gain_ratio"]):
                best = candidate

        if best is None:
            return None

        values = X[:, feature_index]
        left_mask = values <= best["threshold"]
        best["left_mask"] = left_mask
        best["right_mask"] = ~left_mask
        return best

    def find_best_split(self, X, y, sample_weights):
        parent_entropy = self.entropy(y, sample_weights)
        if parent_entropy <= 0:
            return None

        attribute_candidates = []
        for feature_index in range(X.shape[1]):
            candidate = self._best_split_for_feature(X, y, sample_weights, feature_index)
            if candidate is not None:
                attribute_candidates.append(candidate)

        if not attribute_candidates:
            return None

        mean_gain_ratio = float(np.mean([c["gain_ratio"] for c in attribute_candidates]))
        eligible = [c for c in attribute_candidates if c["gain_ratio"] >= mean_gain_ratio]
        if not eligible:
            eligible = attribute_candidates

        eligible.sort(key=lambda c: (c["gain_ratio"], c["gain"], -c["feature"]), reverse=True)
        best = eligible[0]

        if best["gain_ratio"] <= self.min_gain_ratio:
            return None
        return best

    def _leaf(self, y, sample_weights, depth):
        return C45Node(
            is_leaf=True, prediction=self.majority_class(y, sample_weights),
            probabilities=self.class_distribution(y, sample_weights),
            raw_probabilities=self.class_distribution(y, sample_weights=None),
            samples=len(y), weighted_samples=float(np.sum(sample_weights)), depth=depth,
        )

    def _build_tree(self, X, y, sample_weights, depth=0):
        n_samples = len(y)
        if n_samples == 0:
            raise ValueError("Internal error: empty node encountered.")
        if len(np.unique(y)) == 1:
            return self._leaf(y, sample_weights, depth)
        if self.max_depth is not None and depth >= self.max_depth:
            return self._leaf(y, sample_weights, depth)
        if n_samples < self.min_samples_split:
            return self._leaf(y, sample_weights, depth)

        best = self.find_best_split(X, y, sample_weights)
        if best is None:
            return self._leaf(y, sample_weights, depth)
        if best["gain_ratio"] <= self.min_gain_ratio:
            return self._leaf(y, sample_weights, depth)

        left_mask, right_mask = best["left_mask"], best["right_mask"]
        X_left, y_left, w_left = X[left_mask], y[left_mask], sample_weights[left_mask]
        X_right, y_right, w_right = X[right_mask], y[right_mask], sample_weights[right_mask]

        node = C45Node(
            is_leaf=False, prediction=self.majority_class(y, sample_weights),
            probabilities=self.class_distribution(y, sample_weights),
            raw_probabilities=self.class_distribution(y, sample_weights=None),
            feature=best["feature"], threshold=best["threshold"],
            information_gain=best["gain"], gain_ratio=best["gain_ratio"],
            split_information=best["split_info"], samples=n_samples,
            weighted_samples=float(np.sum(sample_weights)), depth=depth,
        )
        node.left = self._build_tree(X_left, y_left, w_left, depth + 1)
        node.right = self._build_tree(X_right, y_right, w_right, depth + 1)
        return node

    def _calculate_depth(self, node):
        if node is None:
            return 0
        if node.is_leaf:
            return node.depth
        return max(self._calculate_depth(node.left), self._calculate_depth(node.right))

    def _count_leaves(self, node):
        if node is None:
            return 0
        if node.is_leaf:
            return 1
        return self._count_leaves(node.left) + self._count_leaves(node.right)

    def _count_nodes(self, node):
        if node is None:
            return 0
        return 1 + self._count_nodes(node.left) + self._count_nodes(node.right)

    @staticmethod
    def _binomial_upper_error_rate(errors, samples, confidence_factor=0.25):
        if samples <= 0:
            return 0.0
        if errors <= 0:
            return min(0.5, 0.5 / samples)
        p = errors / samples
        cf = min(max(float(confidence_factor), 1e-6), 0.5)
        z_values = {0.01: 2.32635, 0.05: 1.64485, 0.10: 1.28155,
                    0.20: 0.84162, 0.25: 0.67449, 0.50: 0.0}
        closest_cf = min(z_values, key=lambda x: abs(x - cf))
        z = z_values[closest_cf]
        denominator = 1.0 + (z * z / samples)
        centre = p + (z * z / (2.0 * samples))
        margin = z * np.sqrt((p * (1.0 - p) / samples) + (z * z / (4.0 * samples * samples)))
        upper = (centre + margin) / denominator
        return float(min(max(upper, p), 1.0))

    def _subtree_leaf_error(self, node):
        if node.is_leaf:
            minority = 1.0 - max(node.raw_probabilities.values())
            return minority * node.samples
        return self._subtree_leaf_error(node.left) + self._subtree_leaf_error(node.right)

    def _prune_node(self, node):
        if node is None or node.is_leaf:
            return node, False
        node.left, left_changed = self._prune_node(node.left)
        node.right, right_changed = self._prune_node(node.right)
        children_changed = left_changed or right_changed

        if node.left.is_leaf and node.right.is_leaf:
            subtree_errors = self._subtree_leaf_error(node)
            leaf_error_count = (1.0 - max(node.raw_probabilities.values())) * node.samples
            subtree_rate = self._binomial_upper_error_rate(subtree_errors, node.samples, self.confidence_factor)
            leaf_rate = self._binomial_upper_error_rate(leaf_error_count, node.samples, self.confidence_factor)
            if leaf_rate <= subtree_rate + 1e-12:
                return (C45Node(
                    is_leaf=True, prediction=node.prediction, probabilities=node.probabilities,
                    raw_probabilities=node.raw_probabilities, samples=node.samples,
                    weighted_samples=node.weighted_samples, depth=node.depth,
                ), True)
        return node, children_changed

    def post_prune(self):
        if self.root is None or not self.pruning:
            return
        before_nodes = self._count_nodes(self.root)
        before_leaves = self._count_leaves(self.root)
        self.root, _ = self._prune_node(self.root)
        after_nodes = self._count_nodes(self.root)
        after_leaves = self._count_leaves(self.root)
        self.pruning_changes_ = before_nodes - after_nodes
        self.n_nodes_before_pruning_ = before_nodes
        self.n_leaves_before_pruning_ = before_leaves
        self.n_nodes_ = after_nodes
        self.n_leaves_ = after_leaves
        self.tree_depth_ = self._calculate_depth(self.root)

    def fit(self, X, y, feature_names=None):
        X, y = self._validate_input(X, y)
        if not np.issubdtype(y.dtype, np.number):
            raise ValueError("y must contain numeric class labels.")
        if len(np.unique(y)) < 2:
            raise ValueError("y must contain at least two classes.")
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        self._y_train = y.copy()
        if feature_names is None:
            self.feature_names_ = [f"feature_{i}" for i in range(self.n_features_in_)]
        else:
            self.feature_names_ = list(feature_names)
            if len(self.feature_names_) != self.n_features_in_:
                raise ValueError("feature_names length must match number of features.")
        self._prepare_class_weights()
        sample_weights = self._sample_weights(y)
        self.root = self._build_tree(X, y, sample_weights, depth=0)
        self.n_nodes_before_pruning_ = self._count_nodes(self.root)
        self.n_leaves_before_pruning_ = self._count_leaves(self.root)
        self.n_nodes_ = self.n_nodes_before_pruning_
        self.n_leaves_ = self.n_leaves_before_pruning_
        self.tree_depth_ = self._calculate_depth(self.root)
        if self.pruning:
            self.post_prune()
        else:
            self.n_nodes_ = self._count_nodes(self.root)
            self.n_leaves_ = self._count_leaves(self.root)
            self.tree_depth_ = self._calculate_depth(self.root)
        self.feature_importances_ = self._calculate_feature_importance(X, y)
        return self

    def _calculate_feature_importance(self, X, y):
        importances = np.zeros(self.n_features_in_, dtype=float)
        total_weight = float(np.sum(self._sample_weights(y)))
        if total_weight <= 0:
            return importances
        def recurse(node):
            if node is None or node.is_leaf:
                return
            if node.feature is not None:
                importances[node.feature] += (node.weighted_samples / total_weight) * node.information_gain
            recurse(node.left)
            recurse(node.right)
        recurse(self.root)
        total = float(np.sum(importances))
        if total > 0:
            importances /= total
        return importances

    def _predict_proba_one(self, x, node):
        if node.is_leaf:
            return np.asarray([node.raw_probabilities.get(int(cls), 0.0) for cls in self.classes_], dtype=float)
        if x[node.feature] <= node.threshold:
            return self._predict_proba_one(x, node.left)
        return self._predict_proba_one(x, node.right)

    def predict_proba(self, X):
        X, _ = self._validate_input(X)
        return np.asarray([self._predict_proba_one(x, self.root) for x in X], dtype=float)

    def predict(self, X, threshold=None):
        X, _ = self._validate_input(X)
        if threshold is None:
            threshold = self.prediction_threshold
        threshold = float(threshold)
        if threshold < 0.0 or threshold > 1.0:
            raise ValueError("threshold must be between 0 and 1.")
        proba = self.predict_proba(X)
        if 1 in self.classes_:
            idx = list(self.classes_).index(1)
            p1 = proba[:, idx]
            return np.where(p1 >= threshold, 1, 0)
        return np.asarray([self.classes_[int(np.argmax(row))] for row in proba])

    def extract_rules(self):
        rules = []
        rule_id = 0
        def recurse(node, conditions):
            nonlocal rule_id
            if node.is_leaf:
                probabilities = {int(k): float(v) for k, v in node.raw_probabilities.items()}
                predicted_class = int(node.prediction)
                confidence = float(max(probabilities.values())) if probabilities else 0.0
                defect_prob = float(probabilities.get(1, 0.0))
                condition_text = " AND ".join(conditions) if conditions else "TRUE"
                rules.append({
                    "rule_id": rule_id, "condition": condition_text, "predicted_class": predicted_class,
                    "confidence": confidence, "samples": int(node.samples),
                    "weighted_samples": float(node.weighted_samples), "defect_prob": defect_prob,
                    "depth": int(node.depth),
                })
                rule_id += 1
                return
            feature_name = self.feature_names_[node.feature]
            recurse(node.left, conditions + [f"{feature_name} <= {node.threshold:.6f}"])
            recurse(node.right, conditions + [f"{feature_name} > {node.threshold:.6f}"])
        recurse(self.root, [])
        return rules

    def export_text(self):
        lines = []
        def recurse(node, indent=""):
            if node.is_leaf:
                probabilities = {int(k): round(float(v), 6) for k, v in node.raw_probabilities.items()}
                lines.append(f"{indent}LEAF class={node.prediction} samples={node.samples} probabilities={probabilities}")
                return
            feature_name = self.feature_names_[node.feature]
            lines.append(f"{indent}{feature_name} <= {node.threshold:.6f} "
                          f"(InformationGain={node.information_gain:.6f}, GainRatio={node.gain_ratio:.6f}, "
                          f"SplitInfo={node.split_information:.6f})")
            recurse(node.left, indent + "    ")
            lines.append(f"{indent}{feature_name} > {node.threshold:.6f} "
                          f"(InformationGain={node.information_gain:.6f}, GainRatio={node.gain_ratio:.6f}, "
                          f"SplitInfo={node.split_information:.6f})")
            recurse(node.right, indent + "    ")
        recurse(self.root)
        return "\n".join(lines)


# =========================================================
# EVALUATION
# =========================================================

def evaluate_model(model, X_test, y_test, threshold=None):
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, precision_score, recall_score,
        f1_score, confusion_matrix, classification_report, roc_auc_score,
        matthews_corrcoef, average_precision_score,
    )
    X_test, y_test = model._validate_input(X_test, y_test)
    y_proba = model.predict_proba(X_test)
    y_pred = model.predict(X_test, threshold=threshold)

    if 1 in model.classes_:
        class_1_index = list(model.classes_).index(1)
        y_probability = y_proba[:, class_1_index]
    else:
        y_probability = np.zeros(len(X_test), dtype=float)

    accuracy = accuracy_score(y_test, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)
    try:
        roc_auc = roc_auc_score(y_test, y_probability)
    except ValueError:
        roc_auc = 0.0
    try:
        pr_auc = average_precision_score(y_test, y_probability)
    except ValueError:
        pr_auc = 0.0

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    actual_threshold = model.prediction_threshold if threshold is None else float(threshold)

    print("\n" + "=" * 80)
    print("C4.5 FINAL TEST PERFORMANCE")
    print("=" * 80)
    print(f"Prediction threshold: {actual_threshold:.3f}")
    print(f"Accuracy:             {accuracy:.4f}")
    print(f"Balanced Accuracy:    {balanced_accuracy:.4f}")
    print(f"Precision:            {precision:.4f}")
    print(f"Recall/Sensitivity:   {recall:.4f}")
    print(f"Specificity:          {specificity:.4f}")
    print(f"F1-Score:             {f1:.4f}")
    print(f"ROC-AUC:              {roc_auc:.4f}")
    print(f"PR-AUC:               {pr_auc:.4f}")
    print(f"MCC:                  {mcc:.4f}")
    print("\nConfusion Matrix:")
    print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    return {
        "prediction_threshold": actual_threshold, "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy), "precision": float(precision),
        "recall": float(recall), "f1_score": float(f1), "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc), "mcc": float(mcc), "specificity": float(specificity),
        "sensitivity": float(sensitivity), "true_negatives": int(tn), "false_positives": int(fp),
        "false_negatives": int(fn), "true_positives": int(tp),
    }


def cross_validate_c45(X, y, feature_names, params, cv):
    from sklearn.metrics import f1_score
    val_scores, train_scores = [], []
    oof_proba = np.zeros(len(y), dtype=float)
    for train_idx, valid_idx in cv.split(X, y):
        m = C45DecisionTree(
            max_depth=params['max_depth'], min_samples_split=params['min_samples_split'],
            min_samples_leaf=params['min_samples_leaf'], min_gain_ratio=1e-7,
            min_information_gain=1e-7, pruning=True, confidence_factor=params['confidence_factor'],
            class_weight='balanced', random_state=42, prediction_threshold=0.5,
        )
        m.fit(X[train_idx], y[train_idx], feature_names)
        train_pred = m.predict(X[train_idx], threshold=0.5)
        valid_pred = m.predict(X[valid_idx], threshold=0.5)
        train_scores.append(f1_score(y[train_idx], train_pred, zero_division=0))
        val_scores.append(f1_score(y[valid_idx], valid_pred, zero_division=0))
        p = m.predict_proba(X[valid_idx])
        oof_proba[valid_idx] = p[:, list(m.classes_).index(1)]
    mean_val = float(np.mean(val_scores))
    mean_train = float(np.mean(train_scores))
    return {
        'mean_cv_f1': mean_val, 'std_cv_f1': float(np.std(val_scores)),
        'mean_train_f1': mean_train, 'f1_gap': mean_train - mean_val, 'oof_proba': oof_proba,
    }


def find_best_threshold_with_specificity_floor(y_true, oof_proba, min_specificity):
    """FIX C: only consider thresholds whose OOF specificity is >=
    min_specificity, then pick the best F1 among those. Prevents the
    degenerate 'predict almost everyone positive' solution."""
    from sklearn.metrics import f1_score
    y_true = np.asarray(y_true)
    rows = []
    for t in np.arange(0.05, 0.951, 0.01):
        pred = (oof_proba >= t).astype(int)
        tn = int(np.sum((pred == 0) & (y_true == 0)))
        fp = int(np.sum((pred == 1) & (y_true == 0)))
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = f1_score(y_true, pred, zero_division=0)
        rows.append({'threshold': round(float(t), 2), 'f1': float(f1), 'specificity': float(spec)})

    df = pd.DataFrame(rows)
    eligible = df[df['specificity'] >= min_specificity]

    if eligible.empty:
        best_row = df.sort_values('specificity', ascending=False).iloc[0]
        warning = (f"WARNING: no threshold reached the {min_specificity:.2f} specificity floor; "
                   f"falling back to the highest-specificity threshold "
                   f"({best_row['specificity']:.4f}).")
    else:
        best_row = eligible.sort_values('f1', ascending=False).iloc[0]
        warning = None

    return float(best_row['threshold']), float(best_row['f1']), float(best_row['specificity']), df, warning


def train_c45():
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    report_dir = os.path.join(project_root, "report")
    models_dir = os.path.join(project_root, "models")
    os.makedirs(report_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_path = os.path.join(report_dir, f"c45_report_v5_{timestamp}.txt")

    original_stdout = sys.stdout
    tee = Tee(report_path)
    sys.stdout = tee
    try:
        print("=" * 80)
        print("C4.5 DECISION TREE FROM SCRATCH - V5")
        print("=" * 80)
        print("\nV5 fixes:")
        print("  FIX A: candidate thresholds no longer restricted to class-change boundaries")
        print("  FIX B: wider max_depth / smaller min_samples_leaf")
        print(f"  FIX C: threshold search enforces specificity floor >= {C45DecisionTree.MIN_SPECIFICITY:.2f}")

        print("\n[STEP 1] Loading data...")
        paths = {
            'X_train': os.path.join(project_root, "data/processed/X_train_selected_ga.csv"),
            'X_test': os.path.join(project_root, "data/processed/X_test_selected_ga.csv"),
            'y_train': os.path.join(project_root, "data/processed/y_train.csv"),
            'y_test': os.path.join(project_root, "data/processed/y_test.csv"),
        }
        if not os.path.exists(paths['X_train']):
            paths['X_train'] = os.path.join(project_root, "data/processed/X_train_selected.csv")
            paths['X_test'] = os.path.join(project_root, "data/processed/X_test_selected.csv")

        missing = [p for p in paths.values() if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError("Missing files:\n" + "\n".join(missing))
        X_train = pd.read_csv(paths['X_train'])
        X_test = pd.read_csv(paths['X_test'])
        y_train = pd.read_csv(paths['y_train']).values.ravel()
        y_test = pd.read_csv(paths['y_test']).values.ravel()
        feature_names = X_train.columns.tolist()
        if list(X_train.columns) != list(X_test.columns):
            raise ValueError("X_train/X_test columns do not match.")
        if len(X_train) != len(y_train) or len(X_test) != len(y_test):
            raise ValueError("X/y sample counts do not match.")
        if X_train.isnull().any().any() or X_test.isnull().any().any():
            raise ValueError("NaN values found in data.")
        X_train_np = X_train.astype(float).values
        X_test_np = X_test.astype(float).values
        if not np.isfinite(X_train_np).all() or not np.isfinite(X_test_np).all():
            raise ValueError("Infinite values found in data.")
        print(f"✓ Using: {paths['X_train']}")
        print(f"✓ X_train: {X_train_np.shape}")
        print(f"✓ X_test:  {X_test_np.shape}")
        print(f"✓ Features: {feature_names}")
        print(f"✓ y_train: {pd.Series(y_train).value_counts().to_dict()}")
        print(f"✓ y_test:  {pd.Series(y_test).value_counts().to_dict()}")

        print("\n[STEP 2] C4.5 Hyperparameter Optimization")
        print("-" * 80)
        param_grid = {
            'max_depth': [7, 8, 9],
            'min_samples_split': [10, 15],
            'min_samples_leaf': [3, 5],
            'confidence_factor': [0.25],
        }
        combinations = int(np.prod([len(v) for v in param_grid.values()]))
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        print("Parameter Grid:")
        for k, v in param_grid.items():
            print(f"  {k}: {v}")
        print(f"Total combinations: {combinations}")
        print(f"Total model fits: {combinations * 5}")

        results = []
        best_score = -np.inf
        best_params = None
        best_details = None
        combos = [(d, s, l, c) for d in param_grid['max_depth']
                  for s in param_grid['min_samples_split']
                  for l in param_grid['min_samples_leaf']
                  for c in param_grid['confidence_factor']]

        for i, (d, sp, leaf, cf) in enumerate(combos, 1):
            params = {'max_depth': d, 'min_samples_split': sp, 'min_samples_leaf': leaf, 'confidence_factor': cf}
            details = cross_validate_c45(X_train_np, y_train, feature_names, params, cv)
            results.append({**params, **{k: v for k, v in details.items() if k != 'oof_proba'}})
            better = (details['mean_cv_f1'] > best_score + 1e-12 or
                      (abs(details['mean_cv_f1'] - best_score) <= 1e-12 and
                       best_details is not None and details['f1_gap'] < best_details['f1_gap']))
            if better or best_params is None:
                best_score = details['mean_cv_f1']
                best_params = params.copy()
                best_details = details.copy()
            print(f"[{i:02d}/{len(combos)}] depth={d}, split={sp}, leaf={leaf}, CF={cf:.2f} | "
                  f"CV F1={details['mean_cv_f1']:.4f} ±{details['std_cv_f1']:.4f} | "
                  f"Train F1={details['mean_train_f1']:.4f} | Gap={details['f1_gap']:.4f}")

        results_df = pd.DataFrame(results).sort_values(['mean_cv_f1', 'f1_gap'], ascending=[False, True])
        grid_results_path = os.path.join(models_dir, 'c45_gridsearch_results_v5.csv')
        results_df.to_csv(grid_results_path, index=False)
        print("\n[GRID SEARCH TOP 10]")
        print(results_df.head(10).to_string(index=False))
        print("\n✓ Best Parameters:")
        for k, v in best_params.items():
            print(f"  {k}: {v}")
        print(f"✓ Best CV F1: {best_details['mean_cv_f1']:.4f}")
        print(f"✓ Train-CV Gap: {best_details['f1_gap']:.4f}")

        gap = best_details['f1_gap']
        risk = 'LOW' if gap < 0.03 else ('MODERATE' if gap < 0.06 else 'HIGH')
        print("\n[OVERFITTING CHECK]")
        print(f"Overfitting Risk: {risk}")

        print("\n[STEP 3] Training final C4.5 model...")
        model = C45DecisionTree(
            max_depth=best_params['max_depth'], min_samples_split=best_params['min_samples_split'],
            min_samples_leaf=best_params['min_samples_leaf'], min_gain_ratio=1e-7,
            min_information_gain=1e-7, pruning=True, confidence_factor=best_params['confidence_factor'],
            class_weight='balanced', random_state=42, prediction_threshold=0.5,
        )
        model.fit(X_train_np, y_train, feature_names)
        print(f"✓ Tree depth: {model.tree_depth_}")
        print(f"✓ Nodes after pruning:  {model.n_nodes_}")
        print(f"✓ Leaves after pruning:  {model.n_leaves_}")
        print(f"✓ Pruning changes: {model.pruning_changes_}")

        print("\n[STEP 4] Optimizing decision threshold (specificity-constrained)...")
        oof_proba = best_details['oof_proba']
        best_threshold, best_threshold_f1, best_threshold_spec, thr_df, warn = (
            find_best_threshold_with_specificity_floor(y_train, oof_proba, C45DecisionTree.MIN_SPECIFICITY)
        )
        model.prediction_threshold = best_threshold
        threshold_path = os.path.join(models_dir, 'c45_threshold_results_v5.csv')
        thr_df.to_csv(threshold_path, index=False)
        if warn:
            print(warn)
        print(f"✓ Chosen threshold: {best_threshold:.2f} | OOF F1: {best_threshold_f1:.4f} | "
              f"OOF Specificity: {best_threshold_spec:.4f}")

        print("\n[STEP 5] Final Test Evaluation")
        metrics = evaluate_model(model, X_test_np, y_test, threshold=best_threshold)

        print("\n[STEP 6] Feature importance")
        importance_df = pd.DataFrame({
            'feature': feature_names, 'importance': model.feature_importances_,
            'importance_percent': model.feature_importances_ * 100,
        }).sort_values('importance', ascending=False)
        print(importance_df.to_string(index=False))
        importance_path = os.path.join(models_dir, 'c45_feature_importance_v5.csv')
        importance_df.to_csv(importance_path, index=False)

        print("\n[STEP 7] Extracting rules")
        rules = model.extract_rules()
        rules_df = pd.DataFrame(rules)
        rules_path = os.path.join(models_dir, 'c45_extracted_rules_v5.csv')
        rules_df.to_csv(rules_path, index=False)
        print(f"✓ Rules extracted: {len(rules_df)}")
        if not rules_df.empty:
            print(rules_df.head(10).to_string(index=False))

        tree_path = os.path.join(models_dir, 'c45_tree_rules_v5.txt')
        with open(tree_path, 'w', encoding='utf-8') as f:
            f.write(model.export_text())

        print("\n[STEP 9] Saving model and configuration")
        model_path = os.path.join(models_dir, 'c45_model_v5.pkl')
        joblib.dump(model, model_path)
        config = {
            'model_info': {
                'algorithm': 'C4.5-style (v5)', 'implementation': 'From Scratch',
                'training_date': timestamp, 'features': feature_names, 'data_path': paths['X_train'],
            },
            'v5_fixes': {
                'A_candidate_thresholds': 'all distinct adjacent value midpoints (like CART)',
                'B_param_grid': 'max_depth up to 9, min_samples_leaf down to 3',
                'C_threshold_search': f'specificity floor >= {C45DecisionTree.MIN_SPECIFICITY:.2f}',
            },
            'optimization': {
                'best_params': best_params, 'best_mean_cv_f1': best_details['mean_cv_f1'],
                'train_cv_gap': gap, 'overfitting_risk': risk,
            },
            'threshold_optimization': {
                'method': 'specificity-constrained OOF F1 maximization',
                'min_specificity_floor': C45DecisionTree.MIN_SPECIFICITY,
                'chosen_threshold': best_threshold, 'oof_f1_at_threshold': best_threshold_f1,
                'oof_specificity_at_threshold': best_threshold_spec,
            },
            'tree_structure': {
                'depth': model.tree_depth_, 'leaf_nodes': model.n_leaves_,
                'pruning_changes': model.pruning_changes_, 'rules': len(rules),
            },
            'metrics': metrics,
        }
        config_path = os.path.join(models_dir, 'c45_config_v5.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"✓ Model saved: {model_path}")
        print(f"✓ Config saved: {config_path}")

        print("\n" + "=" * 80)
        print("C4.5 V5 TRAINING COMPLETED")
        print("=" * 80)
        print(f"Best parameters: {best_params}")
        print(f"Threshold: {best_threshold:.2f}")
        print(f"Test F1: {metrics['f1_score']:.4f}")
        print(f"Test Specificity: {metrics['specificity']:.4f}")
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"Rules: {len(rules)}")
        print(f"Tree depth: {model.tree_depth_}")
        print(f"Report: {report_path}")
    finally:
        sys.stdout = original_stdout
        tee.close()


if __name__ == "__main__":
    train_c45()
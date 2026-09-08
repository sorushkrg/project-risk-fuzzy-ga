import os
import sys
import json
import io

from contextlib import redirect_stdout
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
)

from simpful import (
    FuzzySystem,
    TriangleFuzzySet,
    LinguisticVariable,
)


# =========================================================
# MEMBERSHIP FUNCTION IMPORT
# =========================================================

try:
    from membership_function import (
        calculate_mf_parameters,
        build_simpful_variables,
    )
except ImportError:
    from membership_builder import (
        calculate_mf_parameters,
        build_simpful_variables,
    )


# =========================================================
# CONFIGURATION
# =========================================================

FUZZY_TERMS = (
    "Low",
    "Medium",
    "High",
)

OUTPUT_VARIABLE = "Risk"

OUTPUT_TERMS = (
    "Low",
    "Medium",
    "High",
)

# ---------------------------------------------------------
# Mamdani output universe
# ---------------------------------------------------------

RISK_MIN = 0.0
RISK_MAX = 100.0

# ---------------------------------------------------------
# Classification threshold
#
# فقط baseline است.
# از Test برای انتخاب threshold استفاده نمی‌شود.
# ---------------------------------------------------------

CLASSIFICATION_THRESHOLD = 0.50

# ---------------------------------------------------------
# Simpful Mamdani numerical resolution
# ---------------------------------------------------------

MAMDANI_SUBDIVISIONS = 1000

# ---------------------------------------------------------
# Minimum membership
# ---------------------------------------------------------

MIN_RULE_MEMBERSHIP = 0.03

# ---------------------------------------------------------
# Minimum dominance over second-best term
# ---------------------------------------------------------

MIN_TERM_MARGIN = 0.02

# ---------------------------------------------------------
# Minimum dominance ratio
# ---------------------------------------------------------

MIN_TERM_RATIO = 1.05

# ---------------------------------------------------------
# Dominance rule mode
#
# False = OR
# True  = AND
# ---------------------------------------------------------

DOMINANCE_REQUIRE_BOTH = False

# ---------------------------------------------------------
# Output probability -> linguistic term
# ---------------------------------------------------------

LOW_RISK_LIMIT = 0.33
HIGH_RISK_LIMIT = 0.67

# ---------------------------------------------------------
# Minimum rule weight
# ---------------------------------------------------------

MIN_RULE_WEIGHT = 0.05


# =========================================================
# LOGGING
# =========================================================

class Tee:

    def __init__(self, filename):

        self.terminal = sys.stdout

        self.log = open(
            filename,
            "w",
            encoding="utf-8",
        )

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
# PROJECT PATHS
# =========================================================

def get_project_paths():

    src_dir = os.path.dirname(
        os.path.abspath(__file__)
    )

    project_root = os.path.dirname(
        src_dir
    )

    paths = {

        "project_root":
            project_root,

        "data_dir":
            os.path.join(
                project_root,
                "data",
                "processed",
            ),

        "models_dir":
            os.path.join(
                project_root,
                "models",
            ),

        "report_dir":
            os.path.join(
                project_root,
                "report",
            ),
    }

    os.makedirs(
        paths["report_dir"],
        exist_ok=True,
    )

    return paths


# =========================================================
# FEATURE NAME SANITIZATION
# =========================================================

def sanitize_feature_names(columns):

    sanitized = []

    for col in columns:

        name = str(col)

        name = name.replace(
            "(",
            "_",
        )

        name = name.replace(
            ")",
            "",
        )

        name = name.replace(
            " ",
            "_",
        )

        name = name.replace(
            "-",
            "_",
        )

        sanitized.append(
            name
        )

    return sanitized


# =========================================================
# TRIANGULAR MEMBERSHIP
# =========================================================

def triangular_membership(
    x,
    a,
    b,
    c,
):
    """
    triangular / shoulder membership.

    a == b -> left shoulder
    a < b < c -> triangle
    b == c -> right shoulder
    """

    x = float(x)
    a = float(a)
    b = float(b)
    c = float(c)

    # -----------------------------------------------------
    # Completely degenerate
    # -----------------------------------------------------

    if (
        np.isclose(a, b)
        and
        np.isclose(b, c)
    ):

        return (
            1.0
            if np.isclose(x, b)
            else 0.0
        )

    # -----------------------------------------------------
    # Left shoulder
    # -----------------------------------------------------

    if np.isclose(a, b):

        if x <= b:
            return 1.0

        if x >= c:
            return 0.0

        denominator = c - b

        if denominator <= 0:
            return 0.0

        return float(
            (c - x) / denominator
        )

    # -----------------------------------------------------
    # Right shoulder
    # -----------------------------------------------------

    if np.isclose(b, c):

        if x >= b:
            return 1.0

        if x <= a:
            return 0.0

        denominator = b - a

        if denominator <= 0:
            return 0.0

        return float(
            (x - a) / denominator
        )

    # -----------------------------------------------------
    # Standard triangle
    # -----------------------------------------------------

    if x <= a or x >= c:
        return 0.0

    if np.isclose(x, b):
        return 1.0

    if x < b:

        denominator = b - a

        if denominator <= 0:
            return 0.0

        return float(
            (x - a) / denominator
        )

    denominator = c - b

    if denominator <= 0:
        return 0.0

    return float(
        (c - x) / denominator
    )


# =========================================================
# MAMDANI OUTPUT VARIABLE
# =========================================================

def build_mamdani_output_variable():

    risk_low = TriangleFuzzySet(
        0.0,
        0.0,
        50.0,
        term="Low",
    )

    risk_medium = TriangleFuzzySet(
        0.0,
        50.0,
        100.0,
        term="Medium",
    )

    risk_high = TriangleFuzzySet(
        50.0,
        100.0,
        100.0,
        term="High",
    )

    return LinguisticVariable(
        [
            risk_low,
            risk_medium,
            risk_high,
        ],
        universe_of_discourse=[
            RISK_MIN,
            RISK_MAX,
        ],
        concept="Defect Risk",
    )


# =========================================================
# OUTPUT PROBABILITY -> FUZZY TERM
# =========================================================

def defect_probability_to_term(
    defect_probability,
):

    p = float(
        np.clip(
            defect_probability,
            0.0,
            1.0,
        )
    )

    if p < LOW_RISK_LIMIT:
        return "Low"

    if p < HIGH_RISK_LIMIT:
        return "Medium"

    return "High"


# =========================================================
# INITIAL FEATURE BOUNDS
# =========================================================

def create_initial_bounds(
    feature_names,
):

    return {
        feature: (
            -np.inf,
            np.inf,
        )
        for feature in feature_names
    }


# =========================================================
# FEATURE-SPECIFIC PATH MASK
# =========================================================

def get_feature_interval_mask(
    X_train,
    feature,
    lower,
    upper,
):
    """
    فقط samples مربوط به بازه همان feature.
    """

    values = (
        X_train[feature]
        .to_numpy(
            dtype=float
        )
    )

    mask = np.ones(
        len(X_train),
        dtype=bool,
    )

    if np.isfinite(lower):

        mask &= (
            values > lower
        )

    if np.isfinite(upper):

        mask &= (
            values <= upper
        )

    return mask


# =========================================================
# FULL PATH SAMPLE MASK
# =========================================================

def get_path_mask(
    X_train,
    feature_bounds,
):
    """
    samples مربوط به کل path CART.
    """

    mask = np.ones(
        len(X_train),
        dtype=bool,
    )

    for feature, (
        lower,
        upper,
    ) in feature_bounds.items():

        values = (
            X_train[feature]
            .to_numpy(
                dtype=float
            )
        )

        if np.isfinite(lower):

            mask &= (
                values > lower
            )

        if np.isfinite(upper):

            mask &= (
                values <= upper
            )

    return mask


# =========================================================
# FEATURE INTERVAL -> FUZZY TERM
# =========================================================

def interval_to_term_from_data(
    feature,
    feature_bounds,
    X_train,
    mfs_params,
    path_mask,
):
    """
    تبدیل بازه یک feature در CART
    به Low / Medium / High.

    Membership برای انتخاب term فقط روی
    نمونه‌های واقعی همان CART leaf محاسبه می‌شود.
    """

    if feature not in mfs_params:

        return (
            None,
            0.0,
            0,
            {},
            None,
        )

    lower, upper = feature_bounds[feature]

    # -----------------------------------------------------
    # اگر این feature در path محدود نشده باشد،
    # antecedent مربوط به آن feature ساخته نمی‌شود.
    # -----------------------------------------------------

    if (
        lower == -np.inf
        and
        upper == np.inf
    ):

        return (
            None,
            0.0,
            0,
            {},
            None,
        )

    # -----------------------------------------------------
    # اعتبارسنجی path mask
    # -----------------------------------------------------

    if len(path_mask) != len(X_train):

        raise ValueError(
            "path_mask length does not match X_train."
        )

    # -----------------------------------------------------
    # فقط نمونه‌های واقعی همان CART leaf
    # -----------------------------------------------------

    feature_values = (
        X_train.loc[
            path_mask,
            feature,
        ]
        .astype(float)
        .to_numpy()
    )

    if len(feature_values) == 0:

        return (
            None,
            0.0,
            0,
            {},
            "empty_leaf",
        )

    # -----------------------------------------------------
    # Membership scores
    # -----------------------------------------------------

    scores = {}

    for term, params in (
        mfs_params[feature].items()
    ):

        a, b, c = params

        if term == "Low":

            memberships = np.asarray(
                [
                    trapezoidal_membership(
                        value,
                        a,
                        a,
                        b,
                        c,
                    )
                    for value in feature_values
                ],
                dtype=float,
            )

        elif term == "High":

            memberships = np.asarray(
                [
                    trapezoidal_membership(
                        value,
                        a,
                        b,
                        c,
                        c,
                    )
                    for value in feature_values
                ],
                dtype=float,
            )

        else:

            memberships = np.asarray(
                [
                    triangular_membership(
                        value,
                        a,
                        b,
                        c,
                    )
                    for value in feature_values
                ],
                dtype=float,
            )

        scores[term] = (
            float(np.mean(memberships))
            if len(memberships) > 0
            else 0.0
        )

    # -----------------------------------------------------
    # Sort terms by score
    # -----------------------------------------------------

    ranked_terms = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    best_term = ranked_terms[0][0]

    best_score = float(
        ranked_terms[0][1]
    )

    second_score = (
        float(ranked_terms[1][1])
        if len(ranked_terms) > 1
        else 0.0
    )

    # -----------------------------------------------------
    # Basic membership threshold
    # -----------------------------------------------------

    if best_score < MIN_RULE_MEMBERSHIP:

        return (
            None,
            best_score,
            len(feature_values),
            scores,
            "weak_membership",
        )

    # -----------------------------------------------------
    # Dominance checks
    # -----------------------------------------------------

    score_margin = (
        best_score
        - second_score
    )

    ratio_ok = (
        best_score
        >=
        max(
            second_score * MIN_TERM_RATIO,
            MIN_RULE_MEMBERSHIP,
        )
    )

    margin_ok = (
        score_margin
        >=
        MIN_TERM_MARGIN
    )

    # -----------------------------------------------------
    # Dominance mode
    # -----------------------------------------------------

    if DOMINANCE_REQUIRE_BOTH:

        dominance_ok = (
            ratio_ok
            and
            margin_ok
        )

    else:

        dominance_ok = (
            ratio_ok
            or
            margin_ok
        )

    # -----------------------------------------------------
    # Ambiguous mapping
    # -----------------------------------------------------

    if not dominance_ok:

        return (
            None,
            best_score,
            len(feature_values),
            scores,
            "ambiguous_membership",
        )

    return (
        best_term,
        best_score,
        len(feature_values),
        scores,
        None,
    )


# =========================================================
# TRAPEZOIDAL MEMBERSHIP
# =========================================================

def trapezoidal_membership(
    x,
    a,
    b,
    c,
    d,
):
    """
    trapezoidal membership.

    a == b -> left shoulder
    c == d -> right shoulder
    """

    x = float(x)
    a = float(a)
    b = float(b)
    c = float(c)
    d = float(d)

    if x < a or x > d:
        return 0.0

    if b > a and x < b:
        return float(
            (x - a) / (b - a)
        )

    if x <= c:
        return 1.0

    if d > c:
        return float(
            (d - x) / (d - c)
        )

    return 1.0


# =========================================================
# CHECK CONTRADICTIONS
# =========================================================

def find_rule_contradictions(
    structured_rules,
):

    groups = {}

    for rule in structured_rules:

        key = tuple(
            sorted(
                rule[
                    "conditions"
                ].items()
            )
        )

        groups.setdefault(
            key,
            set(),
        ).add(
            rule[
                "output_term"
            ]
        )

    contradictions = {

        key: outputs

        for key, outputs
        in groups.items()

        if len(outputs) > 1
    }

    return contradictions



def analyze_rule_duplicates_and_conflicts(
    raw_rules,
):
    """
    تفکیک duplicateهای واقعی از conflictها.

    duplicate:
        antecedent یکسان + output یکسان

    conflict:
        antecedent یکسان + output متفاوت
    """

    groups = {}

    for rule in raw_rules:

        conditions_key = tuple(
            sorted(
                rule["conditions"].items()
            )
        )

        output_term = rule["output_term"]

        groups.setdefault(
            conditions_key,
            [],
        ).append(
            output_term
        )

    duplicate_groups = {}
    conflict_groups = {}

    duplicate_count = 0
    conflict_count = 0

    for conditions_key, outputs in groups.items():

        if len(outputs) <= 1:
            continue

        unique_outputs = set(outputs)

        # ---------------------------------------------
        # Conflict
        # ---------------------------------------------

        if len(unique_outputs) > 1:

            conflict_groups[
                conditions_key
            ] = outputs

            conflict_count += (
                len(outputs) - 1
            )

        # ---------------------------------------------
        # Duplicate
        # ---------------------------------------------

        else:

            duplicate_groups[
                conditions_key
            ] = outputs

            duplicate_count += (
                len(outputs) - 1
            )

    return {
        "duplicate_groups": duplicate_groups,
        "conflict_groups": conflict_groups,
        "duplicate_count": int(duplicate_count),
        "conflict_count": int(conflict_count),
    }


def print_conflict_details(raw_rules):
    """
    نمایش جزئیات تمام conflictهای واقعی.

    conflict:
        antecedent یکسان + output متفاوت
    """

    analysis = analyze_rule_duplicates_and_conflicts(
        raw_rules
    )

    print("\n" + "=" * 80)
    print("RAW CONFLICT DETAILS")
    print("=" * 80)

    if not analysis["conflict_groups"]:
        print("\n✓ No raw conflicts found.")
        print("=" * 80)
        return

    for idx, (conditions_key, outputs) in enumerate(
        analysis["conflict_groups"].items(),
        start=1,
    ):

        print(
            f"\nConflict Group {idx}"
        )

        print("-" * 80)

        print("Conditions:")

        for feature, term in conditions_key:
            print(
                f"  {feature} IS {term}"
            )

        print("\nConflicting rules:")

        matching_rules = []

        for rule in raw_rules:

            key = tuple(
                sorted(
                    rule["conditions"].items()
                )
            )

            if key == conditions_key:
                matching_rules.append(
                    rule
                )

        for j, rule in enumerate(
            matching_rules,
            start=1,
        ):

            print(
                f"  Rule {j}: "
                f"output={rule['output_term']}, "
                f"p={rule['defect_probability']:.4f}, "
                f"samples={rule['samples']}, "
                f"depth={rule['depth']}, "
                f"path_samples={rule['path_sample_count']}, "
                f"avg_membership={rule['average_membership']:.4f}, "
                f"min_membership={rule['minimum_membership']:.4f}"
            )

    print("\n" + "=" * 80)




def calculate_conflict_scores(raw_rules):
    """
    محاسبه قدرت نسبی ruleها در conflictها.

    هدف:
        بررسی اینکه در هر antecedent متناقض،
        کدام rule شواهد قوی‌تری دارد.
    """

    analysis = analyze_rule_duplicates_and_conflicts(
        raw_rules
    )

    print("\n" + "=" * 80)
    print("CONFLICT STRENGTH ANALYSIS")
    print("=" * 80)

    for group_idx, (conditions_key, outputs) in enumerate(
        analysis["conflict_groups"].items(),
        start=1,
    ):

        matching_rules = []

        for rule in raw_rules:

            key = tuple(
                sorted(
                    rule["conditions"].items()
                )
            )

            if key == conditions_key:
                matching_rules.append(rule)

        if not matching_rules:
            continue

        total_samples = sum(
            rule["samples"]
            for rule in matching_rules
        )

        print(
            f"\nConflict Group {group_idx}"
        )

        print("-" * 80)

        print("Conditions:")

        for feature, term in conditions_key:
            print(
                f"  {feature} IS {term}"
            )

        print("\nRules:")

        for idx, rule in enumerate(
            matching_rules,
            start=1,
        ):

            sample_ratio = (
                rule["samples"] / total_samples
                if total_samples > 0
                else 0.0
            )

            probability = (
                rule["defect_probability"]
            )

            membership = (
                rule["average_membership"]
            )

            # امتیاز تشخیصی اولیه
            strength = (
                sample_ratio
                * membership
                * (
                    1.0
                    + abs(
                        probability - 0.5
                    )
                )
            )

            print(
                f"\n  Rule {idx}"
            )

            print(
                f"    Output:           "
                f"{rule['output_term']}"
            )

            print(
                f"    Probability:      "
                f"{probability:.4f}"
            )

            print(
                f"    Samples:          "
                f"{rule['samples']}"
            )

            print(
                f"    Sample ratio:     "
                f"{sample_ratio:.4f}"
            )

            print(
                f"    Avg membership:   "
                f"{membership:.4f}"
            )

            print(
                f"    Min membership:   "
                f"{rule['minimum_membership']:.4f}"
            )

            print(
                f"    Diagnostic score: "
                f"{strength:.6f}"
            )

    print("\n" + "=" * 80)

# =========================================================
# CART LEAF EXTRACTION
# =========================================================

def extract_cart_leaf_rules(
    tree_model,
    X_train,
    feature_names,
    mfs_params,
):
    """
    استخراج تمام leafهای CART.

    برای هر leaf:

        conditions
        output term
        defect probability
        sample count
        membership quality
    """

    if not hasattr(
        tree_model,
        "tree_",
    ):

        raise TypeError(
            "tree_model must expose '.tree_'."
        )

    tree_ = tree_model.tree_

    raw_rules = []

    def recurse(
        node,
        feature_bounds,
        conditions,
        depth,
    ):

        feature_index = int(
            tree_.feature[node]
        )

        # =================================================
        # LEAF
        # =================================================

        if feature_index < 0:

            value = np.asarray(
                tree_.value[node][0],
                dtype=float,
            )

            # -------------------------------------------------
            # Weighted / unweighted leaf support
            # -------------------------------------------------

            unweighted_samples = int(
                tree_.n_node_samples[node]
            )

            weighted_samples = float(
                tree_.weighted_n_node_samples[node]
            )

            total = float(
                np.sum(value)
            )

            if (
                total > 0
                and
                len(value) >= 2
            ):

                defect_probability = float(
                    value[1] / total
                )

            else:

                defect_probability = 0.0

            fuzzy_conditions = {}

            membership_scores = []

            feature_score_details = {}

            # -------------------------------------------------
            # Full CART path mask
            # -------------------------------------------------

            path_mask = get_path_mask(
                X_train,
                feature_bounds,
            )

            path_sample_count = int(
                np.sum(path_mask)
            )

            # -------------------------------------------------
            # Build fuzzy antecedents
            # -------------------------------------------------

            for feature in feature_names:

                (
                    lower,
                    upper,
                ) = feature_bounds[
                    feature
                ]

                if (
                    lower == -np.inf
                    and
                    upper == np.inf
                ):

                    continue

                (
                    term,
                    membership_score,
                    sample_count,
                    all_scores,
                    mapping_status,
                ) = interval_to_term_from_data(
                    feature,
                    feature_bounds,
                    X_train,
                    mfs_params,
                    path_mask,
                )

                feature_score_details[
                    feature
                ] = {

                    "lower":
                        float(lower)
                        if np.isfinite(lower)
                        else None,

                    "upper":
                        float(upper)
                        if np.isfinite(upper)
                        else None,

                    "selected_term":
                        term,

                    "selected_membership":
                        float(
                            membership_score
                        ),

                    "sample_count":
                        int(
                            sample_count
                        ),

                    "mapping_status":
                        mapping_status,

                    "all_memberships":
                        {
                            key: float(value)
                            for key, value
                            in all_scores.items()
                        },
                }

                if term is not None:

                    fuzzy_conditions[
                        feature
                    ] = term

                    membership_scores.append(
                        membership_score
                    )

            # -------------------------------------------------
            # Membership quality
            # -------------------------------------------------

            if membership_scores:

                average_path_membership = float(
                    np.mean(
                        membership_scores
                    )
                )

                minimum_path_membership = float(
                    np.min(
                        membership_scores
                    )
                )

            else:

                average_path_membership = 0.0
                minimum_path_membership = 0.0

            # -------------------------------------------------
            # Output term
            # -------------------------------------------------

            output_term = (
                defect_probability_to_term(
                    defect_probability
                )
            )

            raw_rules.append(
                {

                    "conditions":
                        fuzzy_conditions,

                    "output_term":
                        output_term,

                    "defect_probability":
                        float(
                            defect_probability
                        ),

                    # تعداد واقعی نمونه‌ها
                    "samples":
                        int(
                            unweighted_samples
                        ),

                    # تعداد weighted نمونه‌های leaf
                    # مطابق با class_weight مورد استفاده در CART
                    "weighted_samples":
                        float(
                            weighted_samples
                        ),

                    "depth":
                        int(
                            depth
                        ),

                    "path_sample_count":
                        path_sample_count,

                    "average_membership":
                        average_path_membership,

                    "minimum_membership":
                        minimum_path_membership,

                    "feature_score_details":
                        feature_score_details,
                }
            )

            return

        # =================================================
        # INTERNAL NODE
        # =================================================

        if not (
            0 <= feature_index
            < len(feature_names)
        ):

            return

        feature_name = (
            feature_names[
                feature_index
            ]
        )

        threshold = float(
            tree_.threshold[node]
        )

        # =================================================
        # LEFT CHILD
        #
        # feature <= threshold
        # =================================================

        left_bounds = dict(
            feature_bounds
        )

        old_lower, old_upper = (
            left_bounds[
                feature_name
            ]
        )

        left_bounds[
            feature_name
        ] = (
            old_lower,
            min(
                old_upper,
                threshold,
            ),
        )

        recurse(
            tree_.children_left[node],
            left_bounds,
            conditions + [
                (
                    feature_name,
                    "<=",
                    threshold,
                )
            ],
            depth + 1,
        )

        # =================================================
        # RIGHT CHILD
        #
        # feature > threshold
        # =================================================

        right_bounds = dict(
            feature_bounds
        )

        old_lower, old_upper = (
            right_bounds[
                feature_name
            ]
        )

        right_bounds[
            feature_name
        ] = (
            max(
                old_lower,
                threshold,
            ),
            old_upper,
        )

        recurse(
            tree_.children_right[node],
            right_bounds,
            conditions + [
                (
                    feature_name,
                    ">",
                    threshold,
                )
            ],
            depth + 1,
        )

    recurse(
        0,
        create_initial_bounds(
            feature_names
        ),
        [],
        0,
    )

    return raw_rules


# =========================================================
# MERGE IDENTICAL FUZZY ANTECEDENTS
# =========================================================

def merge_fuzzy_rules(
    raw_rules,
):
    """
    Merge only true duplicates.

    True duplicate:
        same antecedent + same output term

    Conflict:
        same antecedent + different output term

    Conflicting rules are preserved as separate rules.
    """

    if not raw_rules:
        return [], 0

    groups = {}

    # ---------------------------------------------------------
    # GROUP BY antecedent + output term
    # ---------------------------------------------------------

    for rule in raw_rules:

        conditions_key = tuple(
            sorted(
                rule["conditions"].items()
            )
        )

        output_term = rule[
            "output_term"
        ]

        group_key = (
            conditions_key,
            output_term,
        )

        groups.setdefault(
            group_key,
            [],
        ).append(
            rule
        )

    merged_rules = []

    # تعداد ruleهای واقعی که به دلیل duplicate merge می‌شوند
    duplicate_count = 0

    # ---------------------------------------------------------
    # MERGE TRUE DUPLICATES ONLY
    # ---------------------------------------------------------

    for (
        group_key,
        rules,
    ) in groups.items():

        conditions_key, output_term = (
            group_key
        )

        duplicate_count += max(
            0,
            len(rules) - 1,
        )

        # -----------------------------------------------------
        # Unweighted sample count
        # -----------------------------------------------------

        total_samples = sum(
            int(rule["samples"])
            for rule in rules
        )

        # -----------------------------------------------------
        # Weighted sample count
        # -----------------------------------------------------

        total_weighted_samples = sum(
            float(
                rule.get(
                    "weighted_samples",
                    rule["samples"],
                )
            )
            for rule in rules
        )

        if total_weighted_samples <= 0:
            total_weighted_samples = float(
                len(rules)
            )

        # -----------------------------------------------------
        # Weighted defect probability
        # -----------------------------------------------------

        weighted_probability = (
            sum(
                float(
                    rule[
                        "defect_probability"
                    ]
                )
                * float(
                    rule.get(
                        "weighted_samples",
                        rule["samples"],
                    )
                )
                for rule in rules
            )
            / total_weighted_samples
        )

        # -----------------------------------------------------
        # Weighted average membership
        # -----------------------------------------------------

        weighted_membership = (
            sum(
                float(
                    rule[
                        "average_membership"
                    ]
                )
                * float(
                    rule.get(
                        "weighted_samples",
                        rule["samples"],
                    )
                )
                for rule in rules
            )
            / total_weighted_samples
        )

        # -----------------------------------------------------
        # Minimum membership
        # -----------------------------------------------------

        minimum_membership = min(
            float(
                rule[
                    "minimum_membership"
                ]
            )
            for rule in rules
        )

        # -----------------------------------------------------
        # Maximum depth
        # -----------------------------------------------------

        max_depth = max(
            int(
                rule["depth"]
            )
            for rule in rules
        )

        # -----------------------------------------------------
        # Path sample count
        # -----------------------------------------------------

        path_sample_count = sum(
            int(
                rule.get(
                    "path_sample_count",
                    rule["samples"],
                )
            )
            for rule in rules
        )

        # -----------------------------------------------------
        # Source rules
        # -----------------------------------------------------

        source_rules = []

        for rule in rules:

            source_rules.append(
                {
                    "output_term":
                        rule[
                            "output_term"
                        ],

                    "defect_probability":
                        float(
                            rule[
                                "defect_probability"
                            ]
                        ),

                    "samples":
                        int(
                            rule["samples"]
                        ),

                    "weighted_samples":
                        float(
                            rule.get(
                                "weighted_samples",
                                rule["samples"],
                            )
                        ),

                    "depth":
                        int(
                            rule["depth"]
                        ),
                }
            )

        # -----------------------------------------------------
        # Strongest source rule
        # -----------------------------------------------------

        strongest_rule = max(
            rules,
            key=lambda r: (
                float(
                    r.get(
                        "weighted_samples",
                        r["samples"],
                    )
                ),
                float(
                    r[
                        "average_membership"
                    ]
                ),
            ),
        )

        feature_score_details = (
            strongest_rule.get(
                "feature_score_details",
                {},
            )
        )

        # -----------------------------------------------------
        # Build merged rule
        # -----------------------------------------------------

        merged_rule = {

            "conditions":
                dict(
                    conditions_key
                ),

            "output_term":
                output_term,

            "defect_probability":
                float(
                    weighted_probability
                ),

            "samples":
                int(
                    total_samples
                ),

            "weighted_samples":
                float(
                    total_weighted_samples
                ),

            "depth":
                int(
                    max_depth
                ),

            "path_sample_count":
                int(
                    path_sample_count
                ),

            "average_membership":
                float(
                    weighted_membership
                ),

            "minimum_membership":
                float(
                    minimum_membership
                ),

            "feature_score_details":
                feature_score_details,

            "source_rules":
                source_rules,

            "source_rule_count":
                len(rules),
        }

        merged_rules.append(
            merged_rule
        )

    # ---------------------------------------------------------
    # STABLE SORTING
    # ---------------------------------------------------------

    merged_rules.sort(
        key=lambda rule: (
            -float(
                rule.get(
                    "weighted_samples",
                    rule["samples"],
                )
            ),
            str(
                rule["output_term"]
            ),
            tuple(
                sorted(
                    rule[
                        "conditions"
                    ].items()
                )
            ),
        )
    )

    return (
        merged_rules,
        duplicate_count,
    )

# =========================================================
# BUILD MAMDANI RULE STRINGS
# =========================================================

def build_mamdani_rule_strings(
    structured_rules,
):
    """
    تبدیل Ruleها به syntax Mamdani در Simpful.
    """

    fuzzy_rules = []

    if not structured_rules:
        return fuzzy_rules

    max_weighted_samples = max(
        float(
            rule.get(
                "weighted_samples",
                rule["samples"],
            )
        )
        for rule in structured_rules
    )

    for rule in structured_rules:

        conditions = rule[
            "conditions"
        ]

        output_term = rule[
            "output_term"
        ]

        if not conditions:
            continue

        condition_strings = []

        for feature, term in sorted(
            conditions.items()
        ):

            condition_strings.append(
                f"({feature} IS {term})"
            )

        condition_clause = (
            " AND ".join(
                condition_strings
            )
        )

        # -------------------------------------------------
        # Evidence-based rule weight
        # -------------------------------------------------

        weighted_samples = float(
            rule.get(
                "weighted_samples",
                rule["samples"],
            )
        )

        if max_weighted_samples > 0:

            support = (
                weighted_samples
                / max_weighted_samples
            )

        else:

            support = 1.0

        probability = float(
            np.clip(
                rule[
                    "defect_probability"
                ],
                0.0,
                1.0,
            )
        )

        # 0.5 = کمترین certainty
        # 0.0 / 1.0 = بیشترین certainty
        certainty = max(
            probability,
            1.0 - probability,
        )

        raw_weight = (
            support
            * certainty
        )

        weight = max(
            raw_weight,
            MIN_RULE_WEIGHT,
        )

        rule_text = (
            f"IF {condition_clause} "
            f"THEN "
            f"({OUTPUT_VARIABLE} IS "
            f"{output_term}) "
            f"WEIGHT {weight:.4f}"
        )

        fuzzy_rules.append(
            rule_text
        )

        rule[
            "rule_text"
        ] = rule_text

        rule[
            "rule_weight"
        ] = float(
            weight
        )

        rule[
            "rule_support"
        ] = float(
            support
        )

        rule[
            "rule_certainty"
        ] = float(
            certainty
        )

    return fuzzy_rules


# =========================================================
# MAMDANI INFERENCE
# =========================================================

def run_mamdani_inference(
    fuzzy_system,
    X,
    feature_names,
):
    """
    اجرای Mamdani inference.
    """

    scores = []

    zero_firing_count = 0

    for _, row in X.iterrows():

        # -------------------------------------------------
        # Set crisp input
        # -------------------------------------------------

        for feature in feature_names:

            fuzzy_system.set_variable(
                feature,
                float(
                    row[
                        feature
                    ]
                ),
            )

        # -------------------------------------------------
        # Mamdani inference
        # -------------------------------------------------

        with redirect_stdout(
            io.StringIO()
        ):

            result = (
                fuzzy_system.Mamdani_inference(
                    [OUTPUT_VARIABLE],
                    subdivisions=(
                        MAMDANI_SUBDIVISIONS
                    ),
                    ignore_errors=True,
                    ignore_warnings=True,
                    verbose=False,
                )
            )

        score = result.get(
            OUTPUT_VARIABLE,
            None,
        )

        if score is None:

            zero_firing_count += 1

            scores.append(
                np.nan
            )

            continue

        try:

            score = float(
                score
            )

        except (
            TypeError,
            ValueError,
        ):

            zero_firing_count += 1

            scores.append(
                np.nan
            )

            continue

        score = float(
            np.clip(
                score / 100.0,
                0.0,
                1.0,
            )
        )

        scores.append(
            score
        )

    return (
        np.asarray(
            scores,
            dtype=float,
        ),
        zero_firing_count,
    )


# =========================================================
# EVALUATION
# =========================================================

def evaluate_predictions(
    y_true,
    probabilities,
    threshold,
):

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    y_true = np.asarray(
        y_true
    )

    valid_mask = np.isfinite(
        probabilities
    )

    invalid_count = int(
        np.sum(
            ~valid_mask
        )
    )

    if not np.any(
        valid_mask
    ):

        raise RuntimeError(
            "No valid fuzzy predictions were produced."
        )

    y_eval = (
        y_true[
            valid_mask
        ]
    )

    prob_eval = (
        probabilities[
            valid_mask
        ]
    )

    y_pred = (
        prob_eval >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_eval,
        y_pred,
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            y_eval,
            y_pred,
        )
    )

    precision = precision_score(
        y_eval,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_eval,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_eval,
        y_pred,
        zero_division=0,
    )

    mcc = matthews_corrcoef(
        y_eval,
        y_pred,
    )

    try:

        roc_auc = roc_auc_score(
            y_eval,
            prob_eval,
        )

    except ValueError:

        roc_auc = 0.0

    try:

        pr_auc = average_precision_score(
            y_eval,
            prob_eval,
        )

    except ValueError:

        pr_auc = 0.0

    cm = confusion_matrix(
        y_eval,
        y_pred,
        labels=[
            0,
            1,
        ],
    )

    tn, fp, fn, tp = cm.ravel()

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

    metrics = {

        "threshold":
            float(
                threshold
            ),

        "accuracy":
            float(
                accuracy
            ),

        "balanced_accuracy":
            float(
                balanced_accuracy
            ),

        "precision":
            float(
                precision
            ),

        "recall":
            float(
                recall
            ),

        "f1_score":
            float(
                f1
            ),

        "mcc":
            float(
                mcc
            ),

        "specificity":
            float(
                specificity
            ),

        "sensitivity":
            float(
                sensitivity
            ),

        "roc_auc":
            float(
                roc_auc
            ),

        "pr_auc":
            float(
                pr_auc
            ),

        "valid_samples":
            int(
                len(prob_eval)
            ),

        "invalid_samples":
            int(
                invalid_count
            ),

        "true_negatives":
            int(tn),

        "false_positives":
            int(fp),

        "false_negatives":
            int(fn),

        "true_positives":
            int(tp),
    }

    return (
        metrics,
        cm,
        y_pred,
        y_eval,
    )


# =========================================================
# SERIALIZE MEMBERSHIP PARAMETERS
# =========================================================

def serialize_membership_parameters(
    mfs_params,
):

    result = {}

    for feature, terms in (
        mfs_params.items()
    ):

        result[
            feature
        ] = {}

        for term, params in (
            terms.items()
        ):

            result[
                feature
            ][term] = [
                float(x)
                for x in params
            ]

    return result


# =========================================================
# RULE DISTRIBUTION
# =========================================================

def get_rule_output_distribution(
    structured_rules,
):

    distribution = {

        "Low": 0,
        "Medium": 0,
        "High": 0,
    }

    for rule in structured_rules:

        term = rule[
            "output_term"
        ]

        if term in distribution:

            distribution[
                term
            ] += 1

    return distribution


# =========================================================
# RULE MAPPING DIAGNOSTICS
# =========================================================

def get_rule_mapping_statistics(
    raw_rules,
):
    """
    بررسی کیفیت mapping اولیه.
    """

    accepted = 0
    weak = 0
    ambiguous = 0

    for rule in raw_rules:

        details = rule.get(
            "feature_score_details",
            {},
        )

        for _, info in (
            details.items()
        ):

            status = info.get(
                "mapping_status"
            )

            if status is None:

                if (
                    info.get(
                        "selected_term"
                    )
                    is not None
                ):

                    accepted += 1

            elif status == (
                "weak_membership"
            ):

                weak += 1

            elif status == (
                "ambiguous_membership"
            ):

                ambiguous += 1

    return {

        "accepted_mappings":
            int(
                accepted
            ),

        "weak_mappings":
            int(
                weak
            ),

        "ambiguous_mappings":
            int(
                ambiguous
            ),
    }


# =========================================================
# SCORE DISTRIBUTION
# =========================================================

def get_prediction_distribution(
    scores,
    threshold,
):

    valid_scores = np.asarray(
        scores,
        dtype=float,
    )

    valid_scores = valid_scores[
        np.isfinite(
            valid_scores
        )
    ]

    if len(
        valid_scores
    ) == 0:

        return {

            "mean": None,
            "std": None,
            "min": None,
            "max": None,

            "predicted_class_0":
                0,

            "predicted_class_1":
                0,
        }

    predictions = (
        valid_scores >= threshold
    ).astype(int)

    return {

        "mean":
            float(
                np.mean(
                    valid_scores
                )
            ),

        "std":
            float(
                np.std(
                    valid_scores
                )
            ),

        "min":
            float(
                np.min(
                    valid_scores
                )
            ),

        "max":
            float(
                np.max(
                    valid_scores
                )
            ),

        "predicted_class_0":
            int(
                np.sum(
                    predictions == 0
                )
            ),

        "predicted_class_1":
            int(
                np.sum(
                    predictions == 1
                )
            ),
    }


# =========================================================
# MAIN PIPELINE
# =========================================================

def run_fuzzy_pipeline():

    paths = get_project_paths()

    timestamp = (
        datetime.now().strftime(
            "%Y-%m-%d_%H-%M-%S"
        )
    )

    report_path = os.path.join(
        paths["report_dir"],
        f"phase4_mamdani_fuzzy_v5_"
        f"{timestamp}.txt",
    )

    original_stdout = sys.stdout

    tee = Tee(
        report_path
    )

    sys.stdout = tee

    try:

        # =================================================
        # HEADER
        # =================================================

        print("=" * 80)

        print(
            "PHASE 4 - MAMDANI FUZZY SYSTEM "
            "(CART-GUIDED V5)"
        )

        print("=" * 80)

        print("\nArchitecture:")

        print(
            "BPSO Feature Selection"
        )

        print("        ↓")

        print(
            "CART Decision Tree"
        )

        print("        ↓")

        print(
            "CART Leaf Paths"
        )

        print("        ↓")

        print(
            "Feature-specific "
            "Data-aware Fuzzy Mapping"
        )

        print("        ↓")

        print(
            "Ambiguity-aware "
            "Rule Construction"
        )

        print("        ↓")

        print(
            "True-Duplicate Rule Merging"
        )

        print("        ↓")

        print(
            "Conflict-Preserving "
            "Rule Construction"
        )

        print("        ↓")

        print(
            "Mamdani Fuzzy Inference"
        )

        print("        ↓")

        print(
            "Defuzzification"
        )

        print("        ↓")

        print(
            "Phase 5: GA Membership Optimization"
        )

        # =================================================
        # STEP 1 - DATA
        # =================================================

        print(
            "\n[STEP 1] Loading data..."
        )

        train_path = os.path.join(
            paths["data_dir"],
            "X_train_selected.csv",
        )

        test_path = os.path.join(
            paths["data_dir"],
            "X_test_selected.csv",
        )

        y_test_path = os.path.join(
            paths["data_dir"],
            "y_test.csv",
        )

        required_files = [
            train_path,
            test_path,
            y_test_path,
        ]

        missing_files = [
            path
            for path in required_files
            if not os.path.exists(path)
        ]

        if missing_files:

            raise FileNotFoundError(
                "Missing files:\n"
                +
                "\n".join(
                    missing_files
                )
            )

        X_train = pd.read_csv(
            train_path
        )

        X_test = pd.read_csv(
            test_path
        )

        y_test = pd.read_csv(
            y_test_path
        ).values.ravel()

        # -------------------------------------------------
        # Feature names
        # -------------------------------------------------

        original_feature_names = (
            X_train.columns.tolist()
        )

        feature_names = (
            sanitize_feature_names(
                original_feature_names
            )
        )

        if len(
            set(feature_names)
        ) != len(
            feature_names
        ):

            raise ValueError(
                "Feature-name sanitization "
                "produced duplicate names."
            )

        X_train.columns = (
            feature_names
        )

        X_test.columns = (
            feature_names
        )

        # -------------------------------------------------
        # Validation
        # -------------------------------------------------

        if list(
            X_train.columns
        ) != list(
            X_test.columns
        ):

            raise ValueError(
                "X_train and X_test columns do not match."
            )

        if len(X_test) != len(y_test):

            raise ValueError(
                "X_test and y_test lengths do not match."
            )

        if X_train.isnull().any().any():

            raise ValueError(
                "NaN values found in X_train."
            )

        if X_test.isnull().any().any():

            raise ValueError(
                "NaN values found in X_test."
            )

        print(
            f"✓ X_train: "
            f"{X_train.shape}"
        )

        print(
            f"✓ X_test:  "
            f"{X_test.shape}"
        )

        print(
            f"✓ Features: "
            f"{feature_names}"
        )

        print(
            "✓ Test class distribution: "
            f"{pd.Series(y_test).value_counts().to_dict()}"
        )

        # =================================================
        # STEP 2 - LOAD CART
        # =================================================

        print(
            "\n[STEP 2] Loading CART decision tree..."
        )

        cart_model_path = os.path.join(
            paths["models_dir"],
            "decision_tree_model_optimized.pkl",
        )

        if not os.path.exists(
            cart_model_path
        ):

            raise FileNotFoundError(
                "CART model not found:\n"
                f"{cart_model_path}"
            )

        tree_model = joblib.load(
            cart_model_path
        )

        if not hasattr(
            tree_model,
            "tree_",
        ):

            raise TypeError(
                "Loaded model is not a fitted "
                "sklearn decision tree."
            )

        print(
            f"✓ CART model: "
            f"{cart_model_path}"
        )

        print(
            f"✓ Tree depth: "
            f"{tree_model.get_depth()}"
        )

        print(
            f"✓ Tree leaves: "
            f"{tree_model.get_n_leaves()}"
        )

        # =================================================
        # STEP 3 - MEMBERSHIP FUNCTIONS
        # =================================================

        print(
            "\n[STEP 3] Building membership functions..."
        )

        mfs_params = (
            calculate_mf_parameters(
                tree_model,
                X_train,
                feature_names,
            )
        )

        linguistic_variables = (
            build_simpful_variables(
                mfs_params
            )
        )

        print(
            f"✓ Membership variables: "
            f"{len(linguistic_variables)}"
        )

        for feature, terms in (
            mfs_params.items()
        ):

            print(
                f"\n  {feature}:"
            )

            print(
                f"    Low    = "
                f"{terms['Low']}"
            )

            print(
                f"    Medium = "
                f"{terms['Medium']}"
            )

            print(
                f"    High   = "
                f"{terms['High']}"
            )

        # =================================================
        # STEP 4 - BUILD MAMDANI SYSTEM
        # =================================================

        print(
            "\n[STEP 4] Building Mamdani fuzzy system..."
        )

        FS = FuzzySystem(
            show_banner=False
        )

        # -------------------------------------------------
        # Input variables
        # -------------------------------------------------

        for feature, ling_var in (
            linguistic_variables.items()
        ):

            FS.add_linguistic_variable(
                feature,
                ling_var,
            )

        # -------------------------------------------------
        # Output variable
        # -------------------------------------------------

        risk_variable = (
            build_mamdani_output_variable()
        )

        FS.add_linguistic_variable(
            OUTPUT_VARIABLE,
            risk_variable,
        )

        print(
            f"✓ Input variables: "
            f"{len(feature_names)}"
        )

        print(
            f"✓ Output variable: "
            f"{OUTPUT_VARIABLE}"
        )

        print(
            f"✓ Output terms: "
            f"{OUTPUT_TERMS}"
        )

        # =================================================
        # STEP 5 - CART -> MAMDANI RULES
        # =================================================

        print(
            "\n[STEP 5] Extracting "
            "CART-guided Mamdani rules..."
        )

        raw_rules = (
            extract_cart_leaf_rules(
                tree_model,
                X_train,
                feature_names,
                mfs_params,
            )
        )

        print(
            f"✓ Raw CART leaf rules: "
            f"{len(raw_rules)}"
        )

        # -------------------------------------------------
        # Raw contradictions BEFORE MERGE
        # -------------------------------------------------

        raw_contradictions = (
            find_rule_contradictions(
                raw_rules
            )
        )

        raw_rule_analysis = (
            analyze_rule_duplicates_and_conflicts(
                raw_rules
            )
        )

        print(
            f"✓ Raw duplicate rules: "
            f"{raw_rule_analysis['duplicate_count']}"
        )

        print(
            f"✓ Raw conflicting rules: "
            f"{raw_rule_analysis['conflict_count']}"
        )

        print(
            f"✓ Raw duplicate groups: "
            f"{len(raw_rule_analysis['duplicate_groups'])}"
        )

        print(
            f"✓ Raw conflict groups: "
            f"{len(raw_rule_analysis['conflict_groups'])}"
        )

        print_conflict_details(
            raw_rules
        )

        calculate_conflict_scores(
            raw_rules
        )



        print(
            f"\n✓ Raw contradictory antecedents: "
            f"{len(raw_contradictions)}"
        )

        if raw_contradictions:

            print(
                "⚠ Raw contradictions detected "
                "before rule merging."
            )

        else:

            print(
                "✓ No raw contradictory antecedents."
            )

        # -------------------------------------------------
        # Raw output distribution
        # -------------------------------------------------

        raw_distribution = (
            get_rule_output_distribution(
                raw_rules
            )
        )

        print(
            "\nRaw Rule Output Distribution:"
        )

        print(
            f"  Low:    "
            f"{raw_distribution['Low']}"
        )

        print(
            f"  Medium: "
            f"{raw_distribution['Medium']}"
        )

        print(
            f"  High:   "
            f"{raw_distribution['High']}"
        )

        # -------------------------------------------------
        # Mapping statistics
        # -------------------------------------------------

        mapping_statistics = (
            get_rule_mapping_statistics(
                raw_rules
            )
        )

        print(
            "\nFuzzy Mapping Statistics:"
        )

        print(
            f"  Accepted mappings: "
            f"{mapping_statistics['accepted_mappings']}"
        )

        print(
            f"  Weak mappings: "
            f"{mapping_statistics['weak_mappings']}"
        )

        print(
            f"  Ambiguous mappings: "
            f"{mapping_statistics['ambiguous_mappings']}"
        )

        # -------------------------------------------------
        # Merge identical fuzzy antecedents
        # -------------------------------------------------

        (
            structured_rules,
            duplicate_count,
        ) = merge_fuzzy_rules(
            raw_rules
        )

        print(
            f"\n✓ Final fuzzy rules after "
            f"true-duplicate merging: "
            f"{len(structured_rules)}"
        )

        print(
            f"✓ True duplicate rules merged: "
            f"{duplicate_count}"
        )

        print(
            f"✓ Conflicting rules preserved: "
            f"{raw_rule_analysis['conflict_count']}"
        )

        # -------------------------------------------------
        # Remove empty antecedents
        # -------------------------------------------------

        before_empty_filter = len(
            structured_rules
        )

        structured_rules = [
            rule
            for rule in structured_rules
            if rule["conditions"]
        ]

        removed_empty_rules = (
            before_empty_filter
            -
            len(structured_rules)
        )

        print(
            f"✓ Empty antecedent rules removed: "
            f"{removed_empty_rules}"
        )

        # -------------------------------------------------
        # Re-number
        # -------------------------------------------------

        for index, rule in enumerate(
            structured_rules,
            start=1,
        ):

            rule[
                "rule_id"
            ] = int(index)

        # -------------------------------------------------
        # Contradictions AFTER MERGE
        # -------------------------------------------------

        contradictions = (
            find_rule_contradictions(
                structured_rules
            )
        )

        print(
            f"✓ Remaining contradictory "
            f"antecedents: "
            f"{len(contradictions)}"
        )

        if contradictions:

            print(
                "⚠ WARNING: contradictory "
                "antecedents remain."
            )

        else:

            print(
                "✓ No contradictory "
                "antecedents remain."
            )

        # -------------------------------------------------
        # Final output distribution
        # -------------------------------------------------

        final_distribution = (
            get_rule_output_distribution(
                structured_rules
            )
        )

        print(
            "\nFinal Rule Output Distribution:"
        )

        print(
            f"  Low:    "
            f"{final_distribution['Low']}"
        )

        print(
            f"  Medium: "
            f"{final_distribution['Medium']}"
        )

        print(
            f"  High:   "
            f"{final_distribution['High']}"
        )

        # -------------------------------------------------
        # Build rule strings
        # -------------------------------------------------

        fuzzy_rules = (
            build_mamdani_rule_strings(
                structured_rules
            )
        )

        if not fuzzy_rules:

            raise RuntimeError(
                "No valid Mamdani rules were generated."
            )

        FS.add_rules(
            fuzzy_rules
        )

        print(
            f"✓ Mamdani rules added: "
            f"{len(fuzzy_rules)}"
        )

        weights_list = [
            rule["rule_weight"]
            for rule in structured_rules
            if "rule_weight" in rule
        ]

        if weights_list:

            print(
                f"✓ Rule weights — "
                f"min: {min(weights_list):.3f}, "
                f"mean: {sum(weights_list) / len(weights_list):.3f}, "
                f"max: {max(weights_list):.3f}"
            )

        # =================================================
        # DISPLAY RULES
        # =================================================

        print(
            "\nFirst 10 Mamdani Rules:"
        )

        for rule in fuzzy_rules[:10]:

            print(
                f"  {rule}"
            )

        # =================================================
        # STEP 6 - MAMDANI INFERENCE
        # =================================================

        print(
            "\n[STEP 6] Running Mamdani inference..."
        )

        test_scores, zero_firing = (
            run_mamdani_inference(
                FS,
                X_test,
                feature_names,
            )
        )

        valid_predictions = int(
            np.sum(
                np.isfinite(
                    test_scores
                )
            )
        )

        print(
            f"✓ Test samples: "
            f"{len(test_scores)}"
        )

        print(
            f"✓ Zero-firing samples: "
            f"{zero_firing}"
        )

        print(
            f"✓ Valid fuzzy predictions: "
            f"{valid_predictions}"
        )

        # -------------------------------------------------
        # Score distribution
        # -------------------------------------------------

        score_distribution = (
            get_prediction_distribution(
                test_scores,
                CLASSIFICATION_THRESHOLD,
            )
        )

        print(
            "\nFuzzy Score Distribution:"
        )

        print(
            f"  Mean: "
            f"{score_distribution['mean']}"
        )

        print(
            f"  Std:  "
            f"{score_distribution['std']}"
        )

        print(
            f"  Min:  "
            f"{score_distribution['min']}"
        )

        print(
            f"  Max:  "
            f"{score_distribution['max']}"
        )

        print(
            f"  Predicted Class 0: "
            f"{score_distribution['predicted_class_0']}"
        )

        print(
            f"  Predicted Class 1: "
            f"{score_distribution['predicted_class_1']}"
        )

        # =================================================
        # STEP 7 - FINAL TEST EVALUATION
        # =================================================

        print(
            "\n[STEP 7] Final test evaluation..."
        )

        threshold = (
            CLASSIFICATION_THRESHOLD
        )

        (
            metrics,
            cm,
            y_pred,
            y_eval,
        ) = evaluate_predictions(
            y_test,
            test_scores,
            threshold,
        )

        (
            tn,
            fp,
            fn,
            tp,
        ) = cm.ravel()

        # =================================================
        # FINAL RESULTS
        # =================================================

        print(
            "\n" + "=" * 80
        )

        print(
            "PHASE 4 MAMDANI FUZZY SYSTEM - "
            "FINAL RESULTS"
        )

        print(
            "=" * 80
        )

        print(
            f"Threshold:           "
            f"{metrics['threshold']:.2f}"
        )

        print(
            f"Accuracy:            "
            f"{metrics['accuracy']:.4f}"
        )

        print(
            f"Balanced Accuracy:   "
            f"{metrics['balanced_accuracy']:.4f}"
        )

        print(
            f"Precision:           "
            f"{metrics['precision']:.4f}"
        )

        print(
            f"Recall/Sensitivity:  "
            f"{metrics['recall']:.4f}"
        )

        print(
            f"Specificity:         "
            f"{metrics['specificity']:.4f}"
        )

        print(
            f"F1-Score:            "
            f"{metrics['f1_score']:.4f}"
        )

        print(
            f"MCC:                 "
            f"{metrics['mcc']:.4f}"
        )

        print(
            f"ROC-AUC:             "
            f"{metrics['roc_auc']:.4f}"
        )

        print(
            f"PR-AUC:              "
            f"{metrics['pr_auc']:.4f}"
        )

        print(
            f"Valid predictions:   "
            f"{metrics['valid_samples']}"
        )

        print(
            f"Invalid predictions: "
            f"{metrics['invalid_samples']}"
        )

        print(
            "\nConfusion Matrix:"
        )

        print(
            f"TN={tn}, "
            f"FP={fp}, "
            f"FN={fn}, "
            f"TP={tp}"
        )

        print(
            "\nClassification Report:"
        )

        print(
            classification_report(
                y_eval,
                y_pred,
                zero_division=0,
            )
        )

        # =================================================
        # STEP 8 - SAVE RULES
        # =================================================

        print(
            "\n[STEP 8] Saving rules..."
        )

        rules_df = pd.DataFrame(
            structured_rules
        )

        if not rules_df.empty:

            rules_df[
                "conditions_text"
            ] = rules_df[
                "conditions"
            ].apply(
                lambda x:
                    " AND ".join(
                        f"{k} IS {v}"
                        for k, v
                        in sorted(
                            x.items()
                        )
                    )
            )

        rules_path = os.path.join(
            paths["models_dir"],
            "mamdani_rules_cart_v5.csv",
        )

        rules_df.to_csv(
            rules_path,
            index=False,
        )

        print(
            f"✓ Rule CSV: "
            f"{rules_path}"
        )

        rules_text_path = os.path.join(
            paths["models_dir"],
            "mamdani_rules_cart_v5.txt",
        )

        with open(
            rules_text_path,
            "w",
            encoding="utf-8",
        ) as file:

            for rule in fuzzy_rules:

                file.write(
                    rule
                    +
                    "\n"
                )

        print(
            f"✓ Rule text: "
            f"{rules_text_path}"
        )

        # =================================================
        # STEP 9 - SAVE CONFIGURATION
        # =================================================

        print(
            "\n[STEP 9] Saving Phase 4 configuration..."
        )

        results = {

            "phase":
                4,

            "version":
                "Mamdani-CART-V5",

            "feature_selection":
                "BPSO",

            "decision_tree":
                "CART",

            "fuzzy_inference":
                "Mamdani",

            "input_features":
                feature_names,

            "num_features":
                len(feature_names),

            "cart_tree_depth":
                int(
                    tree_model.get_depth()
                ),

            "cart_tree_leaves":
                int(
                    tree_model.get_n_leaves()
                ),

            "raw_cart_leaf_rules":
                int(
                    len(raw_rules)
                ),

            "raw_contradictory_antecedents":
                int(
                    len(raw_contradictions)
                ),

            "unique_fuzzy_rules":
                int(
                    len(structured_rules)
                ),

            "mamdani_rules_added":
                int(
                    len(fuzzy_rules)
                ),

            "merged_duplicate_rules":
                int(
                    duplicate_count
                ),

            "preserved_conflicting_rules":
                int(
                    raw_rule_analysis[
                        "conflict_count"
                    ]
                ),

            "empty_antecedent_rules_removed":
                int(
                    removed_empty_rules
                ),

            "remaining_contradictions":
                int(
                    len(contradictions)
                ),

            "mapping_statistics":
                mapping_statistics,

            "mapping_configuration":
                {
                    "minimum_membership":
                        float(
                            MIN_RULE_MEMBERSHIP
                        ),

                    "minimum_term_margin":
                        float(
                            MIN_TERM_MARGIN
                        ),

                    "minimum_term_ratio":
                        float(
                            MIN_TERM_RATIO
                        ),

                    "dominance_require_both":
                        bool(
                            DOMINANCE_REQUIRE_BOTH
                        ),
                },

            "rule_output_distribution_raw":
                raw_distribution,

            "rule_output_distribution_final":
                final_distribution,

            "output_variable":
                OUTPUT_VARIABLE,

            "output_terms":
                list(
                    OUTPUT_TERMS
                ),

            "output_universe":
                [
                    RISK_MIN,
                    RISK_MAX,
                ],

            "output_mapping":
                {
                    "low_limit":
                        LOW_RISK_LIMIT,

                    "high_limit":
                        HIGH_RISK_LIMIT,
                },

            "classification_threshold":
                float(
                    threshold
                ),

            "mamdani_subdivisions":
                int(
                    MAMDANI_SUBDIVISIONS
                ),

            "zero_firing_samples":
                int(
                    zero_firing
                ),

            "score_distribution":
                score_distribution,

            "membership_parameters":
                serialize_membership_parameters(
                    mfs_params
                ),

            "metrics":
                metrics,

            "confusion_matrix":
                {
                    "TN":
                        int(tn),

                    "FP":
                        int(fp),

                    "FN":
                        int(fn),

                    "TP":
                        int(tp),
                },
        }

        json_path = os.path.join(
            paths["report_dir"],
            f"phase4_mamdani_fuzzy_v4_"
            f"{timestamp}.json",
        )

        with open(
            json_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                results,
                file,
                indent=4,
                ensure_ascii=False,
            )

        print(
            f"✓ JSON report: "
            f"{json_path}"
        )

        # =================================================
        # STEP 10 - SAVE FUZZY SCORES
        # =================================================

        print(
            "\n[STEP 10] Saving fuzzy scores..."
        )

        score_path = os.path.join(
            paths["report_dir"],
            f"phase4_mamdani_scores_v5_"
            f"{timestamp}.csv",
        )

        score_df = X_test.copy()

        score_df[
            "fuzzy_risk_score"
        ] = test_scores

        score_df[
            "y_true"
        ] = y_test

        score_df[
            "prediction"
        ] = np.where(
            np.isfinite(
                test_scores
            ),
            (
                test_scores
                >= threshold
            ).astype(int),
            -1,
        )

        score_df.to_csv(
            score_path,
            index=False,
        )

        print(
            f"✓ Fuzzy scores: "
            f"{score_path}"
        )

        # =================================================
        # FINAL SUMMARY
        # =================================================

        print(
            "\n" + "=" * 80
        )

        print(
            "PHASE 4 MAMDANI V4 COMPLETED"
        )

        print(
            "=" * 80
        )

        print(
            "Feature Selection: BPSO"
        )

        print(
            "Decision Tree: CART"
        )

        print(
            "Fuzzy Inference: Mamdani"
        )

        print(
            f"Features: "
            f"{len(feature_names)}"
        )

        print(
            f"Raw CART rules: "
            f"{len(raw_rules)}"
        )

        print(
            f"Raw contradictory antecedents: "
            f"{len(raw_contradictions)}"
        )

        print(
            f"Final Mamdani rules: "
            f"{len(fuzzy_rules)}"
        )

        print(
            f"Accuracy: "
            f"{metrics['accuracy']:.4f}"
        )

        print(
            f"Balanced Accuracy: "
            f"{metrics['balanced_accuracy']:.4f}"
        )

        print(
            f"Precision: "
            f"{metrics['precision']:.4f}"
        )

        print(
            f"Recall: "
            f"{metrics['recall']:.4f}"
        )

        print(
            f"Specificity: "
            f"{metrics['specificity']:.4f}"
        )

        print(
            f"F1-Score: "
            f"{metrics['f1_score']:.4f}"
        )

        print(
            f"MCC: "
            f"{metrics['mcc']:.4f}"
        )

        print(
            f"ROC-AUC: "
            f"{metrics['roc_auc']:.4f}"
        )

        print(
            f"PR-AUC: "
            f"{metrics['pr_auc']:.4f}"
        )

        print(
            f"Zero-firing: "
            f"{zero_firing}"
        )

        print(
            f"Remaining contradictions: "
            f"{len(contradictions)}"
        )

        print(
            "\nMapping statistics:"
        )

        print(
            f"  Accepted: "
            f"{mapping_statistics['accepted_mappings']}"
        )

        print(
            f"  Weak: "
            f"{mapping_statistics['weak_mappings']}"
        )

        print(
            f"  Ambiguous: "
            f"{mapping_statistics['ambiguous_mappings']}"
        )

        print(
            "\nOutput distribution:"
        )

        print(
            f"  Rule Low: "
            f"{final_distribution['Low']}"
        )

        print(
            f"  Rule Medium: "
            f"{final_distribution['Medium']}"
        )

        print(
            f"  Rule High: "
            f"{final_distribution['High']}"
        )

        print(
            "\nPrediction distribution:"
        )

        print(
            f"  Class 0: "
            f"{score_distribution['predicted_class_0']}"
        )

        print(
            f"  Class 1: "
            f"{score_distribution['predicted_class_1']}"
        )

        print(
            f"\nReport: "
            f"{report_path}"
        )

        print(
            "=" * 80
        )

    finally:

        sys.stdout = original_stdout

        tee.close()


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    run_fuzzy_pipeline()
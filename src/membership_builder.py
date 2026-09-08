import numpy as np
from simpful import TriangleFuzzySet, TrapezoidFuzzySet, LinguisticVariable


# =========================================================
# CONFIGURATION
# =========================================================

FUZZY_TERMS = ("Low", "Medium", "High")

# حداقل فاصله عددی پایه
EPS_RATIO = 1e-6
MIN_EPS = 1e-8

# حداکثر threshold مورد استفاده از CART
MAX_CART_THRESHOLDS = 7

# حداقل تعداد نمونه برای معتبر بودن node
MIN_REGION_SAMPLES = 10

# حداقل فاصله نسبی بین anchorها
# این مقدار برای جلوگیری از MFهای بسیار باریک است.
MIN_ANCHOR_GAP_RATIO = 0.02

# سهم توزیع داده در initialization
# نسبت به CART
DATA_WEIGHT = 0.75
CART_WEIGHT = 0.25

# حداکثر فاصله threshold از quantile هدف
# برای اینکه threshold نامرتبط روی anchor اثر نگذارد.
MAX_THRESHOLD_DISTANCE_RATIO = 0.15


# =========================================================
# INTERNAL HELPERS
# =========================================================

def _safe_eps(vmin, vmax):
    """
    محاسبه فاصله حداقلی متناسب با دامنه feature.
    """

    data_range = float(vmax - vmin)

    if data_range <= 0:
        return MIN_EPS

    return max(
        data_range * EPS_RATIO,
        MIN_EPS
    )


def _finite_values(values):
    """
    حذف NaN و Inf.
    """

    values = np.asarray(
        values,
        dtype=float
    )

    return values[
        np.isfinite(values)
    ]


def _unique_sorted(values):
    """
    حذف مقادیر تکراری و مرتب‌سازی.
    """

    if len(values) == 0:
        return np.array(
            [],
            dtype=float
        )

    values = np.asarray(
        values,
        dtype=float
    )

    values = values[
        np.isfinite(values)
    ]

    return np.unique(values)


def _validate_anchor_order(points):
    """
    بررسی ترتیب پنج anchor.

    انتظار:

        x0 < x1 < x2 < x3 < x4
    """

    points = np.asarray(
        points,
        dtype=float
    )

    if len(points) != 5:
        return False

    if not np.all(
        np.isfinite(points)
    ):
        return False

    return bool(
        np.all(
            np.diff(points) > 0
        )
    )




def _validate_anchor_geometry(
    points,
    vmin,
    vmax,
    eps
):
    """
    اعتبارسنجی کامل هندسه anchorها.

    شروط:
        1. دقیقاً 5 anchor
        2. همه مقادیر finite باشند
        3. داخل [vmin, vmax] باشند
        4. به‌صورت strict صعودی باشند
        5. فاصله بین anchorهای مجاور کمتر از min_gap نباشد
    """

    points = np.asarray(
        points,
        dtype=float
    )

    # -----------------------------------------------------
    # Basic shape
    # -----------------------------------------------------

    if len(points) != 5:
        return False

    # -----------------------------------------------------
    # Finite values
    # -----------------------------------------------------

    if not np.all(np.isfinite(points)):
        return False

    # -----------------------------------------------------
    # Valid domain
    # -----------------------------------------------------

    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return False

    if vmax <= vmin:
        return False

    if np.any(points < vmin) or np.any(points > vmax):
        return False

    # -----------------------------------------------------
    # Strict ordering
    # -----------------------------------------------------

    if not np.all(np.diff(points) > 0):
        return False

    # -----------------------------------------------------
    # Minimum spacing
    # -----------------------------------------------------

    min_gap = _minimum_anchor_gap(
        vmin,
        vmax,
        eps
    )

    gaps = np.diff(points)

    if np.any(gaps + eps < min_gap):
        return False

    return True

def _minimum_anchor_gap(
    vmin,
    vmax,
    eps
):
    """
    محاسبه حداقل فاصله مجاز بین anchorها.

    هدف:
        جلوگیری از Triangleهای بسیار باریک یا
        تقریباً هم‌پوشان.
    """

    data_range = float(
        vmax - vmin
    )

    if data_range <= 0:
        return max(
            eps,
            MIN_EPS
        )

    return max(
        data_range * MIN_ANCHOR_GAP_RATIO,
        eps * 10.0
    )


def _repair_anchor_order(
    points,
    vmin,
    vmax,
    eps
):
    """
    اصلاح anchorها با حفظ فاصله حداقلی.

    برای initialization استفاده می‌شود.

    ترتیب:

        x0 < x1 < x2 < x3 < x4

    و:

        xi+1 - xi >= min_gap
    """

    points = np.asarray(
        points,
        dtype=float
    ).copy()

    if len(points) != 5:
        raise ValueError(
            "Exactly 5 anchors are required."
        )

    points = np.clip(
        points,
        vmin,
        vmax
    )

    points[0] = float(vmin)
    points[-1] = float(vmax)

    min_gap = _minimum_anchor_gap(
        vmin,
        vmax,
        eps
    )

    data_range = vmax - vmin

    # اگر بازه بسیار کوچک باشد،
    # fallback مساوی استفاده می‌شود.
    if data_range < (
        4.0 * min_gap
    ):

        return np.linspace(
            vmin,
            vmax,
            5
        )

    # -----------------------------------------------------
    # ابتدا نقاط داخلی را محدود می‌کنیم
    # -----------------------------------------------------

    points[1] = np.clip(
        points[1],
        vmin + min_gap,
        vmax - 3.0 * min_gap
    )

    points[2] = np.clip(
        points[2],
        points[1] + min_gap,
        vmax - 2.0 * min_gap
    )

    points[3] = np.clip(
        points[3],
        points[2] + min_gap,
        vmax - min_gap
    )

    # -----------------------------------------------------
    # Forward repair
    # -----------------------------------------------------

    for i in range(1, 4):

        minimum_value = (
            points[i - 1] + min_gap
        )

        if points[i] < minimum_value:
            points[i] = minimum_value

    # -----------------------------------------------------
    # Backward repair
    # -----------------------------------------------------

    for i in range(3, 0, -1):

        maximum_value = (
            points[i + 1] - min_gap
        )

        if points[i] > maximum_value:
            points[i] = maximum_value

    # -----------------------------------------------------
    # Final clipping
    # -----------------------------------------------------

    points[0] = vmin
    points[4] = vmax

    # -----------------------------------------------------
    # Final validation
    # -----------------------------------------------------

    if not _validate_anchor_order(
        points
    ):

        return np.linspace(
            vmin,
            vmax,
            5
        )

    gaps = np.diff(points)

    if np.any(
        gaps < min_gap
    ):

        return np.linspace(
            vmin,
            vmax,
            5
        )

    return points


# =========================================================
# CART THRESHOLD EXTRACTION
# =========================================================

def _get_tree_thresholds(
    tree_model,
    feature_names
):
    """
    استخراج thresholdهای واقعی Decision Tree.

    این thresholdها فقط برای initialization
    استفاده می‌شوند.

    thresholdهای CART مستقیماً
    Low / Medium / High نیستند.
    """

    thresholds_per_feature = {
        feature: []
        for feature in feature_names
    }

    if tree_model is None:
        return thresholds_per_feature

    if not hasattr(
        tree_model,
        "tree_"
    ):
        raise TypeError(
            "tree_model must be a fitted sklearn "
            "DecisionTreeClassifier exposing .tree_."
        )

    tree_ = tree_model.tree_

    for node in range(
        tree_.node_count
    ):

        feature_idx = int(
            tree_.feature[node]
        )

        # sklearn TREE_UNDEFINED = -2
        if feature_idx < 0:
            continue

        if feature_idx >= len(
            feature_names
        ):
            continue

        threshold = float(
            tree_.threshold[node]
        )

        if not np.isfinite(
            threshold
        ):
            continue

        feature_name = feature_names[
            feature_idx
        ]

        thresholds_per_feature[
            feature_name
        ].append(
            threshold
        )

    for feature in feature_names:

        thresholds = np.asarray(
            thresholds_per_feature[
                feature
            ],
            dtype=float
        )

        thresholds = thresholds[
            np.isfinite(thresholds)
        ]

        thresholds_per_feature[
            feature
        ] = _unique_sorted(
            thresholds
        )

    return thresholds_per_feature


# =========================================================
# NODE DEPTHS
# =========================================================

def _get_node_depths(tree_):
    """
    محاسبه depth تمام nodeها به صورت یک‌بار traversal.

    این روش نسبت به محاسبه parent برای هر node
    سریع‌تر و تمیزتر است.
    """

    depths = np.zeros(
        tree_.node_count,
        dtype=int
    )

    stack = [(0, 0)]

    while stack:

        node, depth = stack.pop()

        depths[node] = depth

        left = int(
            tree_.children_left[node]
        )

        right = int(
            tree_.children_right[node]
        )

        if left >= 0:
            stack.append(
                (left, depth + 1)
            )

        if right >= 0:
            stack.append(
                (right, depth + 1)
            )

    return depths


# =========================================================
# CART THRESHOLD STATISTICS
# =========================================================

def _get_threshold_statistics(
    tree_model,
    feature_names
):
    """
    استخراج اطلاعات thresholdهای CART.

    شامل:

        threshold
        samples
        weighted_samples
        impurity
        depth
        impurity_gain
    """

    statistics = {
        feature: []
        for feature in feature_names
    }

    if tree_model is None:
        return statistics

    if not hasattr(
        tree_model,
        "tree_"
    ):
        return statistics

    tree_ = tree_model.tree_

    depths = _get_node_depths(
        tree_
    )

    for node in range(
        tree_.node_count
    ):

        feature_idx = int(
            tree_.feature[node]
        )

        if feature_idx < 0:
            continue

        if feature_idx >= len(
            feature_names
        ):
            continue

        threshold = float(
            tree_.threshold[node]
        )

        if not np.isfinite(
            threshold
        ):
            continue

        left = int(
            tree_.children_left[node]
        )

        right = int(
            tree_.children_right[node]
        )

        if left < 0 or right < 0:
            continue

        parent_weight = float(
            tree_.weighted_n_node_samples[
                node
            ]
        )

        left_weight = float(
            tree_.weighted_n_node_samples[
                left
            ]
        )

        right_weight = float(
            tree_.weighted_n_node_samples[
                right
            ]
        )

        parent_impurity = float(
            tree_.impurity[node]
        )

        left_impurity = float(
            tree_.impurity[left]
        )

        right_impurity = float(
            tree_.impurity[right]
        )

        # -------------------------------------------------
        # Impurity decrease
        # -------------------------------------------------

        if parent_weight > 0:

            impurity_gain = (
                parent_weight
                * parent_impurity
                - left_weight
                * left_impurity
                - right_weight
                * right_impurity
            ) / parent_weight

        else:

            impurity_gain = 0.0

        feature_name = feature_names[
            feature_idx
        ]

        statistics[
            feature_name
        ].append(
            {
                "threshold": threshold,

                "samples": int(
                    tree_.n_node_samples[
                        node
                    ]
                ),

                "weighted_samples": (
                    parent_weight
                ),

                "impurity": (
                    parent_impurity
                ),

                "depth": int(
                    depths[node]
                ),

                "impurity_gain": float(
                    max(
                        impurity_gain,
                        0.0
                    )
                )
            }
        )

    return statistics


# =========================================================
# SELECT CART THRESHOLDS
# =========================================================

def _select_cart_thresholds(
    threshold_statistics,
    values,
    vmin,
    vmax
):
    """
    انتخاب thresholdهای معنادار CART.

    معیار اصلی:

        1. impurity gain
        2. تعداد نمونه
        3. depth

    thresholdهای بسیار نزدیک به هم حذف می‌شوند.
    """

    if not threshold_statistics:
        return np.array(
            [],
            dtype=float
        )

    candidates = []

    for item in threshold_statistics:

        threshold = float(
            item["threshold"]
        )

        samples = int(
            item["samples"]
        )

        impurity_gain = float(
            item.get(
                "impurity_gain",
                0.0
            )
        )

        if not np.isfinite(
            threshold
        ):
            continue

        if not (
            vmin < threshold < vmax
        ):
            continue

        if samples < MIN_REGION_SAMPLES:
            continue

        candidates.append(
            {
                **item,
                "impurity_gain": (
                    impurity_gain
                )
            }
        )

    if not candidates:
        return np.array(
            [],
            dtype=float
        )

    # -----------------------------------------------------
    # نرمال‌سازی معیارها
    # -----------------------------------------------------

    max_gain = max(
        item["impurity_gain"]
        for item in candidates
    )

    max_samples = max(
        item["samples"]
        for item in candidates
    )

    for item in candidates:

        gain_score = (
            item["impurity_gain"]
            / max_gain
            if max_gain > 0
            else 0.0
        )

        sample_score = (
            item["samples"]
            / max_samples
            if max_samples > 0
            else 0.0
        )

        # Gain کمی مهم‌تر از sample support
        item["score"] = (
            0.65 * gain_score
            + 0.35 * sample_score
        )

    candidates.sort(
        key=lambda x: (
            -x["score"],
            x["depth"]
        )
    )

    selected = []

    data_range = float(
        vmax - vmin
    )

    tolerance = max(
        data_range * 0.01,
        MIN_EPS
    )

    for item in candidates:

        threshold = float(
            item["threshold"]
        )

        if any(
            abs(
                threshold - existing
            ) <= tolerance
            for existing in selected
        ):
            continue

        selected.append(
            threshold
        )

        if len(selected) >= (
            MAX_CART_THRESHOLDS
        ):
            break

    return np.sort(
        np.asarray(
            selected,
            dtype=float
        )
    )


# =========================================================
# CART-ASSISTED PEAK
# =========================================================

def _cart_assisted_peak(
    quantile_value,
    cart_thresholds,
    vmin,
    vmax
):
    """
    اصلاح محدود quantile با کمک CART.

    CART فقط زمانی اثر می‌گذارد که threshold
    به ناحیه آماری موردنظر نزدیک باشد.

    خروجی:
        مقدار blended بین quantile و threshold
    """

    if len(cart_thresholds) == 0:
        return float(
            quantile_value
        )

    data_range = float(
        vmax - vmin
    )

    if data_range <= 0:
        return float(
            quantile_value
        )

    distances = np.abs(
        cart_thresholds
        - quantile_value
    )

    nearest_idx = int(
        np.argmin(distances)
    )

    nearest_threshold = float(
        cart_thresholds[nearest_idx]
    )

    nearest_distance = float(
        distances[nearest_idx]
    )

    max_distance = (
        data_range
        * MAX_THRESHOLD_DISTANCE_RATIO
    )

    # اگر threshold بیش از حد با quantile فاصله دارد،
    # اصلاً از آن استفاده نمی‌کنیم.
    if nearest_distance > max_distance:

        return float(
            quantile_value
        )

    blended = (
        DATA_WEIGHT * quantile_value
        + CART_WEIGHT * nearest_threshold
    )

    return float(
        blended
    )


# =========================================================
# CHOOSE INITIAL ANCHORS
# =========================================================

def _choose_initial_anchors(
    values,
    cart_thresholds,
    vmin,
    vmax,
    eps
):
    """
    ساخت پنج anchor اولیه.

        x0 = vmin
        x1 = Low peak
        x2 = Medium peak
        x3 = High peak
        x4 = vmax

    Initialization از:

        Data Distribution
        +
        CART Thresholds

    استفاده می‌کند.

    نکته مهم:
        CART فقط یک اصلاح‌کننده است و اجازه ندارد
        anchorها را به thresholdهای بسیار نزدیک
        فشرده کند.
    """

    values = np.asarray(
        values,
        dtype=float
    )

    values = _finite_values(
        values
    )

    # -----------------------------------------------------
    # Robust distribution initialization
    # -----------------------------------------------------

    p20, p50, p80 = np.percentile(
        values,
        [20, 50, 80]
    )

    # -----------------------------------------------------
    # CART-assisted peaks
    # -----------------------------------------------------

    low_peak = _cart_assisted_peak(
        p20,
        cart_thresholds,
        vmin,
        vmax
    )

    medium_peak = _cart_assisted_peak(
        p50,
        cart_thresholds,
        vmin,
        vmax
    )

    high_peak = _cart_assisted_peak(
        p80,
        cart_thresholds,
        vmin,
        vmax
    )

    # -----------------------------------------------------
    # Basic clipping
    # -----------------------------------------------------

    low_peak = float(
        np.clip(
            low_peak,
            vmin,
            vmax
        )
    )

    medium_peak = float(
        np.clip(
            medium_peak,
            vmin,
            vmax
        )
    )

    high_peak = float(
        np.clip(
            high_peak,
            vmin,
            vmax
        )
    )

    anchors = np.array(
        [
            vmin,
            low_peak,
            medium_peak,
            high_peak,
            vmax
        ],
        dtype=float
    )

    # -----------------------------------------------------
    # Repair ordering and spacing
    # -----------------------------------------------------

    anchors = _repair_anchor_order(
        anchors,
        vmin,
        vmax,
        eps
    )

    return anchors


# =========================================================
# MAIN MEMBERSHIP FUNCTION CALCULATION
# =========================================================

def calculate_mf_parameters(
    tree_model,
    X_train,
    feature_names
):
    """
    تولید Membership Functionهای اولیه
    برای سیستم Fuzzy Mamdani.

    معماری:

        BPSO
          ↓
        CART
          ↓
        CART thresholds
          +
        Data distribution
          ↓
        Five anchors
          ↓
        Triangular Membership Functions
          ↓
        Mamdani Fuzzy Inference
          ↓
        Phase 5 GA

    برای هر feature:

        [x0,x1,x2,x3,x4]

    و:

        Low    = (x0,x1,x2)
        Medium = (x1,x2,x3)
        High   = (x2,x3,x4)

    این تابع فقط initialization را انجام می‌دهد.
    """

    if X_train is None:
        raise ValueError(
            "X_train cannot be None."
        )

    if not feature_names:
        raise ValueError(
            "feature_names cannot be empty."
        )

    missing_features = [
        feature
        for feature in feature_names
        if feature not in X_train.columns
    ]

    if missing_features:
        raise ValueError(
            "Features not found in X_train: "
            f"{missing_features}"
        )

    threshold_statistics = (
        _get_threshold_statistics(
            tree_model,
            feature_names
        )
    )

    mfs_params = {}
    feature_bounds = {}

    for feature in feature_names:

        values = (
            X_train[feature]
            .astype(float)
            .values
        )

        values = _finite_values(
            values
        )

        if len(values) == 0:
            raise ValueError(
                f"Feature '{feature}' contains "
                f"no finite values."
            )

        vmin = float(
            np.min(values)
        )

        vmax = float(
            np.max(values)
        )

        feature_bounds[feature] = (
            vmin,
            vmax
        )



        # -------------------------------------------------
        # Constant feature
        # -------------------------------------------------

        if np.isclose(
            vmin,
            vmax
        ):

            eps = max(
                MIN_EPS,
                abs(vmin) * EPS_RATIO
            )

            # برای feature ثابت، هر سه MF تقریباً
            # روی همان نقطه متمرکز هستند.
            # این مورد در feature selection نباید
            # معمولاً رخ دهد.
            mfs_params[feature] = {

                "Low": (
                    vmin - eps,
                    vmin,
                    vmin + eps
                ),

                "Medium": (
                    vmin - eps,
                    vmin,
                    vmin + eps
                ),

                "High": (
                    vmin - eps,
                    vmin,
                    vmin + eps
                )
            }

            continue

        eps = _safe_eps(
            vmin,
            vmax
        )

        # -------------------------------------------------
        # CART thresholds
        # -------------------------------------------------

        cart_thresholds = (
            _select_cart_thresholds(
                threshold_statistics.get(
                    feature,
                    []
                ),
                values,
                vmin,
                vmax
            )
        )

        # -------------------------------------------------
        # Five anchors
        # -------------------------------------------------

        anchors = _choose_initial_anchors(
            values,
            cart_thresholds,
            vmin,
            vmax,
            eps
        )

        x0, x1, x2, x3, x4 = (
            anchors
        )

        # -------------------------------------------------
        # Triangular input MFs
        # -------------------------------------------------

        mfs_params[feature] = {

            "Low": (
                float(x0),
                float(x1),
                float(x2)
            ),

            "Medium": (
                float(x1),
                float(x2),
                float(x3)
            ),

            "High": (
                float(x2),
                float(x3),
                float(x4)
            )
        }

    validate_mf_parameters(
        mfs_params
    )

    return mfs_params


# =========================================================
# VALIDATION
# =========================================================

def validate_mf_parameters(
    mfs_params
):
    """
    اعتبارسنجی Membership Functionها.
    """

    if not isinstance(
        mfs_params,
        dict
    ):
        raise TypeError(
            "mfs_params must be a dictionary."
        )

    for feature, terms in (
        mfs_params.items()
    ):

        if not isinstance(
            terms,
            dict
        ):
            raise ValueError(
                f"Invalid MF structure for "
                f"'{feature}'."
            )

        if set(terms.keys()) != set(
            FUZZY_TERMS
        ):
            raise ValueError(
                f"Invalid fuzzy terms for "
                f"'{feature}'."
            )

        for term in FUZZY_TERMS:

            params = terms[term]

            if len(params) != 3:
                raise ValueError(
                    f"{feature}/{term} must "
                    f"contain 3 parameters."
                )

            a, b, c = map(
                float,
                params
            )

            if not np.all(
                np.isfinite(
                    [a, b, c]
                )
            ):
                raise ValueError(
                    f"Non-finite MF parameters "
                    f"for {feature}/{term}."
                )

            if not (
                a <= b <= c
            ):
                raise ValueError(
                    f"Invalid MF ordering "
                    f"for {feature}/{term}: "
                    f"{params}"
                )

    return True


# =========================================================
# BUILD SIMPFUL VARIABLES
# =========================================================


def build_simpful_variables(
    mfs_params
):
    """
    ساخت LinguisticVariableهای Simpful.

    این Membership Functionها برای
    ورودی‌های سیستم Mamdani استفاده می‌شوند.

    نکته:
        Low و High به‌صورت shoulder (ذوزنقه) ساخته می‌شوند
        تا در انتهای بازه (vmin و vmax) عضویت به‌جای صفر،
        برابر ۱ باشد و partition درستی (sum=1) حفظ شود.
        Medium همچنان مثلثی باقی می‌ماند.
    """

    validate_mf_parameters(
        mfs_params
    )

    linguistic_vars = {}

    for feature, terms in (
        mfs_params.items()
    ):

        low_a, low_b, low_c = (
            terms["Low"]
        )

        med_a, med_b, med_c = (
            terms["Medium"]
        )

        high_a, high_b, high_c = (
            terms["High"]
        )

        # Left shoulder: flat=1 از low_a تا low_b، سپس افت تا low_c
        fs_low = TrapezoidFuzzySet(
            low_a,
            low_a,
            low_b,
            low_c,
            term="Low"
        )

        fs_medium = TriangleFuzzySet(
            med_a,
            med_b,
            med_c,
            term="Medium"
        )

        # Right shoulder: افزایش تا high_b، سپس flat=1 تا high_c
        fs_high = TrapezoidFuzzySet(
            high_a,
            high_b,
            high_c,
            high_c,
            term="High"
        )

        linguistic_vars[feature] = (
            LinguisticVariable(
                [
                    fs_low,
                    fs_medium,
                    fs_high
                ],
                concept=feature
            )
        )

    return linguistic_vars

# =========================================================
# GET FIVE ANCHORS
# =========================================================

def get_mf_anchors(
    mfs_params,
    feature_names=None
):
    """
    استخراج پنج anchor مشترک.

        x0 = Low.a
        x1 = Low.b
        x2 = Medium.b
        x3 = High.b
        x4 = High.c
    """

    validate_mf_parameters(
        mfs_params
    )

    if feature_names is None:
        feature_names = list(
            mfs_params.keys()
        )

    anchors = {}

    for feature in feature_names:

        if feature not in mfs_params:
            raise KeyError(
                f"Feature '{feature}' not found."
            )

        low = mfs_params[
            feature
        ]["Low"]

        medium = mfs_params[
            feature
        ]["Medium"]

        high = mfs_params[
            feature
        ]["High"]

        anchors[feature] = np.array(
            [
                low[0],
                low[1],
                medium[1],
                high[1],
                high[2]
            ],
            dtype=float
        )

    return anchors


# =========================================================
# BUILD MFS FROM ANCHORS
# =========================================================

def build_mfs_from_anchors(
    anchors,
    feature_names=None,
    feature_bounds=None
):
    """
    ساخت Membership Functionها از پنج anchor.

    ساختار anchor:
        [x0, x1, x2, x3, x4]

    که:
        x0 = vmin
        x1 = Low peak
        x2 = Medium peak
        x3 = High peak
        x4 = vmax

    Membership Functions:

        Low    = (x0, x1, x2)
        Medium = (x1, x2, x3)
        High   = (x2, x3, x4)

    سازگاری با Phase 5 GA:
        اگر feature_bounds داده نشود،
        از x0 و x4 هر feature به‌عنوان
        [vmin, vmax] استفاده می‌شود.
    """

    if not isinstance(
        anchors,
        dict
    ):
        raise TypeError(
            "anchors must be a dictionary."
        )

    if feature_names is None:

        feature_names = list(
            anchors.keys()
        )

    mfs_params = {}

    for feature in feature_names:

        if feature not in anchors:

            raise KeyError(
                f"Feature '{feature}' not found in anchors."
            )

        points = np.asarray(
            anchors[feature],
            dtype=float
        )

        if len(points) != 5:

            raise ValueError(
                f"{feature} requires exactly 5 anchors."
            )

        if not np.all(
            np.isfinite(points)
        ):

            raise ValueError(
                f"Non-finite anchors for '{feature}'."
            )

        # -----------------------------------------------------
        # Determine feature bounds
        # -----------------------------------------------------

        if feature_bounds is not None:

            if feature not in feature_bounds:

                raise KeyError(
                    f"Bounds for feature "
                    f"'{feature}' not found."
                )

            vmin, vmax = map(
                float,
                feature_bounds[feature]
            )

        else:

            # -------------------------------------------------
            # GA compatibility
            #
            # x0 and x4 are fixed domain boundaries.
            # -------------------------------------------------

            vmin = float(
                points[0]
            )

            vmax = float(
                points[4]
            )

        # -----------------------------------------------------
        # Validate domain
        # -----------------------------------------------------

        if not np.isfinite(vmin) or not np.isfinite(vmax):

            raise ValueError(
                f"Invalid bounds for '{feature}'."
            )

        if vmax <= vmin:

            raise ValueError(
                f"Invalid bounds for '{feature}': "
                f"({vmin}, {vmax})"
            )

        # -----------------------------------------------------
        # Force boundary anchors to actual bounds
        # -----------------------------------------------------

        points[0] = vmin
        points[4] = vmax

        # -----------------------------------------------------
        # Anchor geometry validation
        # -----------------------------------------------------

        eps = _safe_eps(
            vmin,
            vmax
        )

        if not _validate_anchor_geometry(
            points,
            vmin,
            vmax,
            eps
        ):

            raise ValueError(
                f"Invalid anchor geometry "
                f"for '{feature}': {points}"
            )

        # -----------------------------------------------------
        # Unpack anchors
        # -----------------------------------------------------

        x0, x1, x2, x3, x4 = points

        # -----------------------------------------------------
        # Build Membership Functions
        # -----------------------------------------------------

        mfs_params[feature] = {

            "Low": (
                float(x0),
                float(x1),
                float(x2)
            ),

            "Medium": (
                float(x1),
                float(x2),
                float(x3)
            ),

            "High": (
                float(x2),
                float(x3),
                float(x4)
            ),
        }

    # ---------------------------------------------------------
    # Final validation
    # ---------------------------------------------------------

    validate_mf_parameters(
        mfs_params
    )

    return mfs_params

# =========================================================
# PHASE 5 - FLATTEN
# =========================================================

def flatten_mf_parameters(
    mfs_params,
    feature_names
):
    """
    تبدیل MFها به chromosome.

    برای 8 feature:

        8 × 5 = 40 genes
    """

    anchors = get_mf_anchors(
        mfs_params,
        feature_names
    )

    chromosome = []

    for feature in feature_names:

        chromosome.extend(
            anchors[feature].tolist()
        )

    return np.asarray(
        chromosome,
        dtype=float
    )


# =========================================================
# PHASE 5 - UNFLATTEN
# =========================================================

def unflatten_mf_parameters(
    chromosome,
    feature_names,
    feature_bounds
):
    """
    تبدیل chromosome به MF parameters.
    """

    chromosome = np.asarray(
        chromosome,
        dtype=float
    )

    expected_length = (
        len(feature_names) * 5
    )

    if len(chromosome) != (
        expected_length
    ):
        raise ValueError(
            f"Invalid chromosome length. "
            f"Expected {expected_length}, "
            f"got {len(chromosome)}."
        )

    if not np.all(
        np.isfinite(chromosome)
    ):
        raise ValueError(
            "Chromosome contains "
            "non-finite values."
        )

    anchors = {}

    index = 0

    for feature in feature_names:

        points = chromosome[
            index:index + 5
        ].copy()

        if feature not in feature_bounds:
            raise KeyError(
                f"Bounds for feature "
                f"'{feature}' not found."
            )

        vmin, vmax = map(
            float,
            feature_bounds[feature]
        )

        eps = _safe_eps(
            vmin,
            vmax
        )

        if not _validate_anchor_geometry(
                points,
                vmin,
                vmax,
                eps
        ):
            raise ValueError(
                f"Invalid anchor geometry "
                f"for '{feature}': {points}"
            )

        anchors[feature] = points

        index += 5

    return build_mfs_from_anchors(
        anchors,
        feature_names,
        feature_bounds
    )


# =========================================================
# PHASE 5 - CHROMOSOME VALIDATION
# =========================================================

def is_valid_chromosome(
    chromosome,
    feature_names,
    feature_bounds
):
    """
    بررسی سریع chromosome.

    شروط اعتبار:
        1. طول صحیح
        2. finite بودن
        3. قرارگیری anchorها داخل bounds
        4. ترتیب صحیح
        5. حداقل فاصله بین anchorها
    """

    try:

        chromosome = np.asarray(
            chromosome,
            dtype=float
        )

        expected_length = (
            len(feature_names) * 5
        )

        if len(chromosome) != expected_length:
            return False

        if not np.all(
            np.isfinite(chromosome)
        ):
            return False

        index = 0

        for feature in feature_names:

            if feature not in feature_bounds:
                return False

            vmin, vmax = map(
                float,
                feature_bounds[feature]
            )

            points = chromosome[
                index:index + 5
            ]

            eps = _safe_eps(
                vmin,
                vmax
            )

            if not _validate_anchor_geometry(
                points,
                vmin,
                vmax,
                eps
            ):
                return False

            index += 5

        return True

    except Exception:
        return False

# =========================================================
# GET MF SUMMARY
# =========================================================

def get_mf_summary(
    mfs_params,
    feature_names=None
):
    """
    تولید summary عددی برای گزارش.
    """

    if feature_names is None:
        feature_names = list(
            mfs_params.keys()
        )

    anchors = get_mf_anchors(
        mfs_params,
        feature_names
    )

    summary = {}

    for feature in feature_names:

        points = anchors[feature]

        summary[feature] = {

            "x0": float(points[0]),
            "x1": float(points[1]),
            "x2": float(points[2]),
            "x3": float(points[3]),
            "x4": float(points[4]),

            "Low": tuple(
                map(
                    float,
                    mfs_params[
                        feature
                    ]["Low"]
                )
            ),

            "Medium": tuple(
                map(
                    float,
                    mfs_params[
                        feature
                    ]["Medium"]
                )
            ),

            "High": tuple(
                map(
                    float,
                    mfs_params[
                        feature
                    ]["High"]
                )
            )
        }

    return summary


# =========================================================
# DISPLAY
# =========================================================

def print_mf_parameters(
    mfs_params
):
    """
    نمایش پارامترهای Membership Function.
    """

    validate_mf_parameters(
        mfs_params
    )

    print("\n" + "=" * 90)
    print(
        "MAMDANI FUZZY MEMBERSHIP "
        "FUNCTION PARAMETERS"
    )
    print("=" * 90)

    for feature, terms in (
        mfs_params.items()
    ):

        print(
            f"\nFeature: {feature}"
        )

        for term in FUZZY_TERMS:

            a, b, c = terms[term]

            print(
                f"  {term:<8}"
                f"a={a:.6f}, "
                f"b={b:.6f}, "
                f"c={c:.6f}"
            )

    print("=" * 90)


def print_mf_anchors(
    mfs_params,
    feature_names=None
):
    """
    نمایش پنج anchor هر feature.
    """

    anchors = get_mf_anchors(
        mfs_params,
        feature_names
    )

    print("\n" + "=" * 90)
    print(
        "MAMDANI FUZZY MEMBERSHIP "
        "FUNCTION ANCHORS"
    )
    print("=" * 90)

    for feature, points in (
        anchors.items()
    ):

        gaps = np.diff(points)

        print(
            f"{feature:<22}: "
            f"x0={points[0]:.6f}, "
            f"x1={points[1]:.6f}, "
            f"x2={points[2]:.6f}, "
            f"x3={points[3]:.6f}, "
            f"x4={points[4]:.6f} | "
            f"gaps="
            f"[{gaps[0]:.6f}, "
            f"{gaps[1]:.6f}, "
            f"{gaps[2]:.6f}, "
            f"{gaps[3]:.6f}]"
        )

    print("=" * 90)
import os
import joblib
import pandas as pd

from fuzzy_system import (
    get_project_paths,
    sanitize_feature_names,
    extract_cart_leaf_rules,
)

from membership_builder import calculate_mf_parameters


def main():

    paths = get_project_paths()

    X_train = pd.read_csv(
        os.path.join(paths["data_dir"], "X_train_selected.csv")
    )

    feature_names = sanitize_feature_names(
        X_train.columns.tolist()
    )

    X_train.columns = feature_names

    tree_model = joblib.load(
        os.path.join(
            paths["models_dir"],
            "decision_tree_model_optimized.pkl",
        )
    )

    mfs_params = calculate_mf_parameters(
        tree_model,
        X_train,
        feature_names,
    )

    raw_rules = extract_cart_leaf_rules(
        tree_model,
        X_train,
        feature_names,
        mfs_params,
    )

    print(f"Total leaves: {len(raw_rules)}\n")

    for rule_index, rule in enumerate(raw_rules):

        details = rule["feature_score_details"]

        for feature, info in details.items():

            status = info["mapping_status"]

            if status is not None:

                print(
                    f"Leaf {rule_index} | {feature} | "
                    f"status={status} | "
                    f"lower={info['lower']}, upper={info['upper']} | "
                    f"selected_membership={info['selected_membership']:.4f} | "
                    f"sample_count={info['sample_count']} | "
                    f"all_memberships={info['all_memberships']}"
                )


if __name__ == "__main__":
    main()
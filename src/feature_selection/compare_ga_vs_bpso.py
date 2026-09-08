import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


def compare_ga_vs_bpso():
    """
    مقایسه نتایج GA و BPSO برای Feature Selection
    """
    
    print("=" * 80)
    print("--- COMPARISON: GENETIC ALGORITHM vs BPSO (FEATURE SELECTION) ---")
    print("=" * 80)

    # ✅ خواندن نتایج BPSO
    print("\n[STEP 1] Reading BPSO Results...")
    print("-" * 80)

    try:
        X_train_bpso = pd.read_csv("../../data/processed/X_train_selected.csv")
        X_test_bpso = pd.read_csv("../../data/processed/X_test_selected.csv")
        
        with open("../../data/processed/selected_feature_names.txt", 'r', encoding='utf-8') as f:
            features_bpso = [line.strip().split('. ')[1] for line in f.readlines()]
        
        print(f"✓ BPSO Results Loaded:")
        print(f"  Features Selected: {len(features_bpso)}")
        print(f"  X_train shape: {X_train_bpso.shape}")
        print(f"  X_test shape: {X_test_bpso.shape}")
    except Exception as e:
        print(f"❌ Error loading BPSO results: {e}")
        return

    # ✅ خواندن نتایج GA
    print("\n[STEP 2] Reading GA Results...")
    print("-" * 80)

    try:
        X_train_ga = pd.read_csv("../../data/processed/X_train_selected_ga.csv")
        X_test_ga = pd.read_csv("../../data/processed/X_test_selected_ga.csv")
        
        with open("../../data/processed/selected_feature_names_ga.txt", 'r', encoding='utf-8') as f:
            features_ga = [line.strip().split('. ')[1] for line in f.readlines()]
        
        print(f"✓ GA Results Loaded:")
        print(f"  Features Selected: {len(features_ga)}")
        print(f"  X_train shape: {X_train_ga.shape}")
        print(f"  X_test shape: {X_test_ga.shape}")
    except Exception as e:
        print(f"❌ Error loading GA results: {e}")
        return

    # ✅ تجزیه و تحلیل
    print("\n" + "=" * 80)
    print("--- DETAILED COMPARISON ---")
    print("=" * 80)

    # جدول مقایسه
    print("\n[COMPARISON TABLE]")
    print("-" * 80)
    
    comparison_data = {
        'Metric': [
            'Number of Features Selected',
            'Feature Reduction %',
            'Total Train Samples',
            'Total Test Samples',
            'Algorithm Complexity',
            'Convergence Speed'
        ],
        'BPSO': [
            f"{len(features_bpso)}",
            f"{(1 - len(features_bpso)/21) * 100:.1f}%",
            f"{X_train_bpso.shape[0]}",
            f"{X_test_bpso.shape[0]}",
            'Medium (Particle-based)',
            'Fast (50 iterations)'
        ],
        'GA': [
            f"{len(features_ga)}",
            f"{(1 - len(features_ga)/21) * 100:.1f}%",
            f"{X_train_ga.shape[0]}",
            f"{X_test_ga.shape[0]}",
            'Medium (Population-based)',
            'Variable'
        ]
    }

    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))

    # ✅ مقایسه فیچرها
    print("\n" + "=" * 80)
    print("--- FEATURE OVERLAP ANALYSIS ---")
    print("=" * 80)

    features_bpso_set = set(features_bpso)
    features_ga_set = set(features_ga)

    common_features = features_bpso_set.intersection(features_ga_set)
    bpso_only = features_bpso_set - features_ga_set
    ga_only = features_ga_set - features_bpso_set

    print(f"\nCommon Features (both algorithms): {len(common_features)}")
    if common_features:
        for i, feat in enumerate(sorted(common_features), 1):
            print(f"  {i}. {feat}")

    print(f"\nBPSO Only: {len(bpso_only)}")
    if bpso_only:
        for i, feat in enumerate(sorted(bpso_only), 1):
            print(f"  {i}. {feat}")

    print(f"\nGA Only: {len(ga_only)}")
    if ga_only:
        for i, feat in enumerate(sorted(ga_only), 1):
            print(f"  {i}. {feat}")

    # ✅ نسبت تطابق
    overlap_ratio = len(common_features) / max(len(features_bpso), len(features_ga))
    print(f"\nFeature Overlap Ratio: {overlap_ratio:.1%}")

    # ✅ رسم مقایسه
    print("\n[STEP 3] Creating Visualization...")
    print("-" * 80)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Feature Count Comparison
    ax1 = axes[0, 0]
    algorithms = ['BPSO', 'GA']
    feature_counts = [len(features_bpso), len(features_ga)]
    colors = ['#2E86AB', '#A23B72']
    bars = ax1.bar(algorithms, feature_counts, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax1.set_ylabel('Number of Features', fontsize=11, fontweight='bold')
    ax1.set_title('Feature Count Comparison', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 12)
    for bar, count in zip(bars, feature_counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count)}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # 2. Feature Reduction Percentage
    ax2 = axes[0, 1]
    reductions = [(1 - len(features_bpso)/21) * 100, (1 - len(features_ga)/21) * 100]
    bars = ax2.bar(algorithms, reductions, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax2.set_ylabel('Reduction %', fontsize=11, fontweight='bold')
    ax2.set_title('Feature Reduction Rate', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 80)
    for bar, reduction in zip(bars, reductions):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{reduction:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # 3. Venn Diagram Style (Overlap)
    ax3 = axes[1, 0]
    categories = ['Common', 'BPSO Only', 'GA Only']
    values = [len(common_features), len(bpso_only), len(ga_only)]
    colors_overlap = ['#F18F01', '#2E86AB', '#A23B72']
    wedges, texts, autotexts = ax3.pie(values, labels=categories, autopct='%1.1f%%',
                                         colors=colors_overlap, startangle=90,
                                         textprops={'fontsize': 10, 'fontweight': 'bold'})
    ax3.set_title('Feature Selection Overlap', fontsize=12, fontweight='bold')

    # 4. Feature Comparison Summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = f"""
    ALGORITHM COMPARISON SUMMARY
    {'='*50}
    
    BPSO:
    • Features: {len(features_bpso)}
    • Reduction: {(1 - len(features_bpso)/21) * 100:.1f}%
    • Iterations: 50 (Early Stop)
    • Speed: Fast
    
    GA:
    • Features: {len(features_ga)}
    • Reduction: {(1 - len(features_ga)/21) * 100:.1f}%
    • Generations: Variable
    • Speed: Medium
    
    OVERLAP ANALYSIS:
    • Common Features: {len(common_features)} ({overlap_ratio:.1%})
    • BPSO Unique: {len(bpso_only)}
    • GA Unique: {len(ga_only)}
    
    RECOMMENDATION:
    {'Both algorithms agree well ✓' if overlap_ratio > 0.7 else 'Moderate agreement ⚠'}
    """
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    comparison_path = f"../../report/ga_vs_bpso_comparison_{timestamp}.png"
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Comparison visualization saved to: {comparison_path}")

    # ✅ ذخیره نتایج در CSV
    print("\n[STEP 4] Saving Comparison Report...")
    print("-" * 80)

    comparison_df.to_csv(f"../../report/ga_vs_bpso_comparison_{timestamp}.csv", index=False)
    print(f"✓ Comparison report saved to: ../../report/ga_vs_bpso_comparison_{timestamp}.csv")

    # ✅ خلاصه نهایی
    print("\n" + "=" * 80)
    print("--- FINAL SUMMARY ---")
    print("=" * 80)
    
    if len(features_bpso) < len(features_ga):
        print(f"\n✓ BPSO selected fewer features ({len(features_bpso)} vs {len(features_ga)})")
        print(f"  → BPSO might be better for dimensionality reduction")
    elif len(features_ga) < len(features_bpso):
        print(f"\n✓ GA selected fewer features ({len(features_ga)} vs {len(features_bpso)})")
        print(f"  → GA might be better for dimensionality reduction")
    else:
        print(f"\n✓ Both algorithms selected the same number of features: {len(features_bpso)}")
    
    if overlap_ratio >= 0.8:
        print(f"\n✓ High feature overlap ({overlap_ratio:.1%}) → Both algorithms are consistent")
    elif overlap_ratio >= 0.6:
        print(f"\n⚠ Moderate overlap ({overlap_ratio:.1%}) → Some differences in feature selection")
    else:
        print(f"\n⚠ Low overlap ({overlap_ratio:.1%}) → Significant differences")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    compare_ga_vs_bpso()

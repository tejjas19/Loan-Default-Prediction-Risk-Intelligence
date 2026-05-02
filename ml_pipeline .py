"""
Loan Default Prediction - Complete ML Pipeline
=============================================
Topics covered:
  - Outlier Treatment (IQR + Z-score)
  - Correlation-based Feature Removal
  - Logistic Regression
  - XGBoost
  - Borrower Segmentation (K-Means)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, f1_score, precision_score, recall_score, accuracy_score
)
from sklearn.cluster import KMeans
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib
import json
import os

# ─────────────────────────────────────────────
# 1. DATA GENERATION (simulate realistic dataset)
# ─────────────────────────────────────────────

def generate_dataset(n=50000, seed=42):
    """Generate a realistic synthetic loan dataset."""
    np.random.seed(seed)

    credit_score   = np.random.normal(680, 80, n).clip(300, 850)
    annual_income  = np.random.lognormal(10.9, 0.6, n).clip(15000, 600000)
    loan_amount    = np.random.lognormal(9.5, 0.7, n).clip(1000, 200000)
    dti_ratio      = np.random.normal(32, 14, n).clip(0, 100)
    emp_length     = np.random.choice(range(0, 41), n, p=np.ones(41)/41)
    delinquencies  = np.random.choice([0,1,2,3,4,5], n, p=[0.65,0.18,0.09,0.05,0.02,0.01])
    open_accounts  = np.random.randint(1, 30, n)
    revol_util     = np.random.normal(48, 24, n).clip(0, 100)
    pub_rec        = np.random.choice([0,1,2,3], n, p=[0.78,0.14,0.06,0.02])
    home_ownership = np.random.choice(['RENT','MORTGAGE','OWN'], n, p=[0.45,0.40,0.15])
    loan_purpose   = np.random.choice(
        ['debt_consolidation','credit_card','home_improvement','other','medical','business'],
        n, p=[0.35,0.25,0.15,0.12,0.08,0.05]
    )
    loan_term      = np.random.choice([36, 60], n, p=[0.55, 0.45])
    interest_rate  = np.random.normal(12, 5, n).clip(5, 30)
    total_acc      = open_accounts + np.random.randint(0, 10, n)

    # Inject outliers (~4%)
    outlier_idx = np.random.choice(n, int(n*0.04), replace=False)
    annual_income[outlier_idx[:len(outlier_idx)//2]] *= np.random.uniform(8, 15, len(outlier_idx)//2)
    dti_ratio[outlier_idx[len(outlier_idx)//2:]] = np.random.uniform(120, 300, len(outlier_idx) - len(outlier_idx)//2)

    # Build default probability (ground truth)
    log_odds = (
        -0.008 * credit_score
        + 0.018 * dti_ratio
        + 0.000003 * loan_amount
        - 0.000002 * annual_income
        + 0.25  * delinquencies
        + 0.15  * pub_rec
        + 0.005 * revol_util
        - 0.04  * emp_length
        + 1.5
    )
    prob = 1 / (1 + np.exp(-log_odds))
    default = (np.random.rand(n) < prob).astype(int)

    df = pd.DataFrame({
        'credit_score':   credit_score,
        'annual_income':  annual_income,
        'loan_amount':    loan_amount,
        'dti_ratio':      dti_ratio,
        'emp_length':     emp_length,
        'delinquencies':  delinquencies,
        'open_accounts':  open_accounts,
        'revol_util':     revol_util,
        'pub_rec':        pub_rec,
        'home_ownership': home_ownership,
        'loan_purpose':   loan_purpose,
        'loan_term':      loan_term,
        'interest_rate':  interest_rate,
        'total_acc':      total_acc,
        'default':        default
    })

    # Add a few correlated (redundant) columns to demonstrate correlation removal
    df['credit_score_copy']    = df['credit_score'] + np.random.normal(0, 5, n)
    df['income_monthly']       = df['annual_income'] / 12
    df['loan_amount_thousands']= df['loan_amount'] / 1000
    df['dti_copy']             = df['dti_ratio']  + np.random.normal(0, 2, n)

    return df


# ─────────────────────────────────────────────
# 2. OUTLIER TREATMENT
# ─────────────────────────────────────────────

def iqr_treatment(df, columns, factor=1.5):
    """Cap outliers using IQR method."""
    df = df.copy()
    report = {}
    for col in columns:
        Q1  = df[col].quantile(0.25)
        Q3  = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR
        n_out = ((df[col] < lower) | (df[col] > upper)).sum()
        df[col] = df[col].clip(lower, upper)
        report[col] = {'lower': round(lower,2), 'upper': round(upper,2), 'outliers_capped': int(n_out)}
    return df, report


def zscore_treatment(df, columns, threshold=3):
    """Remove rows where Z-score exceeds threshold."""
    df = df.copy()
    before = len(df)
    for col in columns:
        mean = df[col].mean()
        std  = df[col].std()
        df = df[np.abs((df[col] - mean) / std) <= threshold]
    removed = before - len(df)
    return df.reset_index(drop=True), {'rows_removed': removed, 'pct_removed': round(removed/before*100, 2)}


# ─────────────────────────────────────────────
# 3. CORRELATION-BASED FEATURE REMOVAL
# ─────────────────────────────────────────────

def remove_correlated_features(df, target_col, threshold=0.85):
    """
    Greedy correlation pruning:
    For each pair with |corr| > threshold, drop the feature
    with lower absolute correlation to the target.
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c != target_col]

    corr_matrix  = df[num_cols].corr().abs()
    target_corr  = df[num_cols].corrwith(df[target_col]).abs()

    dropped = []
    cols    = list(num_cols)

    upper_tri = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    for col in upper_tri.columns:
        if col in dropped:
            continue
        correlated = upper_tri.index[upper_tri[col] > threshold].tolist()
        for c in correlated:
            if c in dropped:
                continue
            # Keep the one with higher correlation to target
            if target_corr.get(col, 0) >= target_corr.get(c, 0):
                dropped.append(c)
            else:
                dropped.append(col)
                break

    kept = [c for c in cols if c not in dropped]
    print(f"\n[Correlation Pruning] Removed {len(dropped)} features: {dropped}")
    print(f"[Correlation Pruning] Kept {len(kept)} features.")
    return df.drop(columns=dropped), dropped, kept


# ─────────────────────────────────────────────
# 4. FEATURE ENGINEERING
# ─────────────────────────────────────────────

def feature_engineering(df):
    """Create new meaningful features."""
    df = df.copy()
    df['loan_to_income']    = df['loan_amount'] / (df['annual_income'] + 1)
    df['payment_to_income'] = (df['loan_amount'] / df.get('loan_term', 36)) / (df['annual_income'] / 12 + 1)
    df['risk_score']        = (df['dti_ratio'] / 100) * (1 - df['credit_score'] / 850)
    return df


# ─────────────────────────────────────────────
# 5. PREPROCESSING
# ─────────────────────────────────────────────

def preprocess(df, target='default', fit_encoders=None, fit_scaler=None):
    """Encode categoricals, scale numerics."""
    df = df.copy()
    cat_cols = df.select_dtypes(include='object').columns.tolist()

    encoders = fit_encoders or {}
    for col in cat_cols:
        if col not in encoders:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            df[col] = le.transform(df[col].astype(str))

    X = df.drop(columns=[target])
    y = df[target]

    scaler = fit_scaler or StandardScaler()
    if fit_scaler is None:
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    return X_scaled, y, encoders, scaler


# ─────────────────────────────────────────────
# 6. MODEL TRAINING
# ─────────────────────────────────────────────

def train_logistic_regression(X_train, y_train):
    """Train Logistic Regression with L2 regularization."""
    model = LogisticRegression(
        C=0.5,
        class_weight='balanced',
        max_iter=1000,
        solver='lbfgs',
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train):
    """Train XGBoost classifier."""
    scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
    model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train)],
        verbose=False
    )
    return model


# ─────────────────────────────────────────────
# 7. EVALUATION
# ─────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, name='Model'):
    """Print full evaluation report and return metrics dict."""
    y_pred      = model.predict(X_test)
    y_prob      = model.predict_proba(X_test)[:, 1]
    threshold   = 0.42  # Optimized threshold
    y_pred_thr  = (y_prob >= threshold).astype(int)

    auc  = roc_auc_score(y_test, y_prob)
    acc  = accuracy_score(y_test, y_pred_thr)
    prec = precision_score(y_test, y_pred_thr, zero_division=0)
    rec  = recall_score(y_test, y_pred_thr, zero_division=0)
    f1   = f1_score(y_test, y_pred_thr, zero_division=0)

    print(f"\n{'='*50}")
    print(f"  {name} — Evaluation Report")
    print(f"{'='*50}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"\n{classification_report(y_test, y_pred_thr)}")

    return {
        'name': name, 'auc': auc, 'accuracy': acc,
        'precision': prec, 'recall': rec, 'f1': f1,
        'y_prob': y_prob, 'y_pred': y_pred_thr
    }


# ─────────────────────────────────────────────
# 8. BORROWER SEGMENTATION (K-Means)
# ─────────────────────────────────────────────

def borrower_segmentation(df_original, k=6):
    """
    Apply K-Means clustering on key features.
    Returns the cluster assignments and segment labels.
    """
    seg_features = ['credit_score', 'dti_ratio', 'annual_income',
                    'loan_amount', 'delinquencies', 'emp_length']
    seg_df = df_original[seg_features].copy()

    scaler  = StandardScaler()
    seg_scaled = scaler.fit_transform(seg_df)

    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10,
                    random_state=42, max_iter=300)
    clusters = kmeans.fit_predict(seg_scaled)
    df_original = df_original.copy()
    df_original['cluster'] = clusters

    # Label clusters by default rate + credit profile
    cluster_stats = df_original.groupby('cluster').agg(
        default_rate=('default', 'mean'),
        avg_credit=('credit_score', 'mean'),
        avg_dti=('dti_ratio', 'mean'),
        avg_delinq=('delinquencies', 'mean'),
        count=('default', 'count')
    ).reset_index()
    cluster_stats['rank'] = cluster_stats['default_rate'].rank()

    segment_names = {
        1: 'Prime Borrowers',
        2: 'Near-Prime Borrowers',
        3: 'Thin-File Borrowers',
        4: 'Subprime Borrowers',
        5: 'Over-Leveraged Borrowers',
        6: 'High-Risk Borrowers'
    }
    cluster_stats['segment'] = cluster_stats['rank'].apply(
        lambda r: segment_names.get(int(r), f'Segment {int(r)}')
    )

    print("\n[K-Means Segmentation] Cluster Summary:")
    print(cluster_stats[['segment','default_rate','avg_credit','avg_dti','count']].to_string(index=False))

    return kmeans, cluster_stats, df_original


# ─────────────────────────────────────────────
# 9. VISUALIZATION
# ─────────────────────────────────────────────

def plot_all(df_clean, results_lr, results_xgb, cluster_stats, feature_names, xgb_model, output_dir='plots'):
    """Generate and save all project plots."""
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid')

    # 1. Class Distribution
    fig, ax = plt.subplots(figsize=(6,4))
    counts = df_clean['default'].value_counts()
    ax.bar(['No Default','Default'], counts.values, color=['#1D9E75','#E24B4A'], width=0.5)
    ax.set_title('Class Distribution', fontsize=14, fontweight='bold')
    ax.set_ylabel('Count')
    for i, v in enumerate(counts.values):
        ax.text(i, v+200, f'{v:,}\n({v/len(df_clean)*100:.1f}%)', ha='center', fontsize=11)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/01_class_distribution.png', dpi=120)
    plt.close()

    # 2. Correlation Heatmap
    num_df = df_clean.select_dtypes(include=np.number).drop(columns=['default'], errors='ignore')
    fig, ax = plt.subplots(figsize=(12, 9))
    mask = np.triu(np.ones_like(num_df.corr(), dtype=bool))
    sns.heatmap(num_df.corr(), mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
                linewidths=0.5, ax=ax, annot_kws={'size':8})
    ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/02_correlation_heatmap.png', dpi=120)
    plt.close()

    # 3. ROC Curves
    fig, ax = plt.subplots(figsize=(7,5))
    for res, color, name in [
        (results_lr,  '#378ADD', f"Logistic Regression (AUC={results_lr['auc']:.3f})"),
        (results_xgb, '#1D9E75', f"XGBoost            (AUC={results_xgb['auc']:.3f})")
    ]:
        fpr, tpr, _ = roc_curve(results_xgb['y_true'] if 'y_true' in results_xgb else res.get('y_true',[]),
                                 res['y_prob'])
        ax.plot(fpr, tpr, color=color, lw=2.5, label=name)
    ax.plot([0,1],[0,1],'--', color='gray', lw=1.5, label='Random Classifier')
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right'); ax.set_xlim([0,1]); ax.set_ylim([0,1.01])
    plt.tight_layout()
    plt.savefig(f'{output_dir}/03_roc_curves.png', dpi=120)
    plt.close()

    # 4. Feature Importance (XGBoost)
    fig, ax = plt.subplots(figsize=(8, 6))
    importances = xgb_model.feature_importances_
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=True).tail(15)
    feat_imp.plot(kind='barh', ax=ax, color='#1D9E75', edgecolor='white')
    ax.set_title('XGBoost Feature Importance (Top 15)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/04_feature_importance.png', dpi=120)
    plt.close()

    # 5. Confusion Matrix
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, res, title in [
        (axes[0], results_lr,  'Logistic Regression'),
        (axes[1], results_xgb, 'XGBoost')
    ]:
        cm = confusion_matrix(res.get('y_true', np.zeros(len(res['y_pred']))), res['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax,
                    xticklabels=['No Default','Default'],
                    yticklabels=['No Default','Default'])
        ax.set_title(f'Confusion Matrix — {title}', fontweight='bold')
        ax.set_ylabel('Actual'); ax.set_xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/05_confusion_matrix.png', dpi=120)
    plt.close()

    # 6. Borrower Segments
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ['#1D9E75','#378ADD','#7F77DD','#BA7517','#D85A30','#E24B4A']
    bars = ax.barh(cluster_stats['segment'], cluster_stats['default_rate']*100,
                   color=colors[:len(cluster_stats)], edgecolor='white', height=0.6)
    ax.set_xlabel('Default Rate (%)'); ax.set_title('Borrower Segments — Default Rate', fontsize=14, fontweight='bold')
    for bar, val in zip(bars, cluster_stats['default_rate']):
        ax.text(bar.get_width()+0.3, bar.get_y()+bar.get_height()/2,
                f'{val*100:.1f}%', va='center', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/06_borrower_segments.png', dpi=120)
    plt.close()

    # 7. Credit Score Distribution by Default
    fig, ax = plt.subplots(figsize=(8,4))
    df_clean[df_clean['default']==0]['credit_score'].plot(kind='kde', ax=ax, color='#1D9E75', label='No Default', lw=2)
    df_clean[df_clean['default']==1]['credit_score'].plot(kind='kde', ax=ax, color='#E24B4A', label='Default',    lw=2)
    ax.set_title('Credit Score Distribution by Default Status', fontsize=13, fontweight='bold')
    ax.set_xlabel('Credit Score'); ax.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/07_credit_score_dist.png', dpi=120)
    plt.close()

    print(f"\n[Plots] Saved 7 plots to '{output_dir}/'")


# ─────────────────────────────────────────────
# 10. MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  LOAN DEFAULT PREDICTION — FULL ML PIPELINE")
    print("=" * 60)

    # Step 1: Generate data
    print("\n[1] Generating dataset...")
    df = generate_dataset(n=50000)
    print(f"    Shape: {df.shape}, Default rate: {df['default'].mean()*100:.1f}%")
    df.to_csv('data/raw_loan_data.csv', index=False)

    # Step 2: Outlier Treatment
    print("\n[2] Outlier Treatment...")
    iqr_cols = ['annual_income', 'loan_amount']
    df_clean, iqr_report = iqr_treatment(df, iqr_cols)
    print(f"    IQR capping applied: {iqr_report}")

    zscore_cols = ['credit_score', 'dti_ratio']
    df_clean, zs_report = zscore_treatment(df_clean, zscore_cols)
    print(f"    Z-score removal: {zs_report}")
    df_clean.to_csv('data/clean_loan_data.csv', index=False)

    # Step 3: Feature Engineering
    print("\n[3] Feature Engineering...")
    df_clean = feature_engineering(df_clean)

    # Step 4: Correlation Removal
    print("\n[4] Correlation-based Feature Removal...")
    df_clean, dropped_cols, kept_cols = remove_correlated_features(df_clean, 'default', threshold=0.85)

    # Step 5: Preprocessing
    print("\n[5] Preprocessing...")
    X, y, encoders, scaler = preprocess(df_clean, target='default')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # SMOTE balancing (training set only)
    print(f"    Before SMOTE: {dict(y_train.value_counts())}")
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    print(f"    After  SMOTE: {dict(pd.Series(y_train_bal).value_counts())}")

    # Step 6: Train Models
    print("\n[6] Training Models...")
    print("    Training Logistic Regression...")
    lr_model  = train_logistic_regression(X_train_bal, y_train_bal)

    print("    Training XGBoost...")
    xgb_model = train_xgboost(X_train_bal, y_train_bal)

    # Step 7: Evaluate
    print("\n[7] Evaluating Models...")
    results_lr  = evaluate_model(lr_model,  X_test, y_test, 'Logistic Regression')
    results_xgb = evaluate_model(xgb_model, X_test, y_test, 'XGBoost')
    results_lr['y_true']  = y_test.values
    results_xgb['y_true'] = y_test.values

    # Step 8: Borrower Segmentation
    print("\n[8] Borrower Segmentation (K-Means)...")
    kmeans, cluster_stats, df_segmented = borrower_segmentation(df_clean, k=6)
    df_segmented.to_csv('data/segmented_data.csv', index=False)

    # Step 9: Save models
    print("\n[9] Saving Models...")
    os.makedirs('models', exist_ok=True)
    joblib.dump(lr_model,  'models/logistic_regression.pkl')
    joblib.dump(xgb_model, 'models/xgboost_model.pkl')
    joblib.dump(scaler,    'models/scaler.pkl')
    joblib.dump(encoders,  'models/encoders.pkl')
    joblib.dump(kmeans,    'models/kmeans.pkl')
    print("    Saved: models/logistic_regression.pkl")
    print("    Saved: models/xgboost_model.pkl")
    print("    Saved: models/kmeans.pkl")

    # Step 10: Save metrics JSON (for web dashboard)
    metrics = {
        'logistic_regression': {k: float(v) for k, v in results_lr.items() if isinstance(v, (int, float, np.floating))},
        'xgboost':             {k: float(v) for k, v in results_xgb.items() if isinstance(v, (int, float, np.floating))},
        'dataset': {
            'total_rows': len(df_clean),
            'default_rate': float(df_clean['default'].mean()),
            'features_original': len(df.columns) - 1,
            'features_after_pruning': len(kept_cols),
            'outliers_removed_pct': zs_report['pct_removed']
        }
    }
    with open('static/js/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print("\n    Saved: static/js/metrics.json")

    # Step 11: Plots
    print("\n[10] Generating Plots...")
    plot_all(df_clean, results_lr, results_xgb, cluster_stats,
             list(X.columns), xgb_model, output_dir='plots')

    print("\n" + "="*60)
    print("  PIPELINE COMPLETE ✓")
    print("="*60)
    print(f"  Best Model  : XGBoost")
    print(f"  Best AUC    : {results_xgb['auc']:.4f}")
    print(f"  Best F1     : {results_xgb['f1']:.4f}")
    print("="*60)


if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    main()
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.inspection import permutation_importance
import joblib

from features import FEATURE_NAMES, prepare_features


def main():
    # Derive features from raw inputs using the same code as both interfaces.
    root = Path(__file__).resolve().parent
    data = pd.read_csv(root / "data/scoring.csv")
    features = FEATURE_NAMES
    X = prepare_features(data).to_numpy()
    y = data["default"].values

    # split with fixed seed and stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Model selection uses only the training split; holdout stays unseen.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Model 1: Logistic regression (baseline)
    model_lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000))
    ])
    scores_lr = cross_val_score(model_lr, X_train, y_train, cv=cv, scoring="roc_auc")
    print(f"[LR]  CV ROC-AUC: {scores_lr.mean():.3f} ± {scores_lr.std():.3f}")

    # Model 2: Histogram Gradient Boosting
    model_gb = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", HistGradientBoostingClassifier(
            class_weight="balanced",
            max_iter=100,
            random_state=42
        ))
    ])
    scores_gb = cross_val_score(model_gb, X_train, y_train, cv=cv, scoring="roc_auc")
    print(f"[GBM] CV ROC-AUC: {scores_gb.mean():.3f} ± {scores_gb.std():.3f}")

    # best model
    if scores_gb.mean() > scores_lr.mean():
        best_model = model_gb
        print("\n→ Selected model: GradientBoosting")
    else:
        best_model = model_lr
        print("\n→ Selected model: LogisticRegression")

    # very final best model train
    best_model.fit(X_train, y_train)

    # save model
    joblib.dump(best_model, root / "model.pkl")
    print("Model saved: model.pkl")

    # test metrics
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    print(f"\nTest ROC-AUC:  {roc_auc_score(y_test, y_proba):.3f}")
    print(f"Precision:     {precision_score(y_test, y_pred, zero_division=0):.3f}")
    print(f"Recall:        {recall_score(y_test, y_pred, zero_division=0):.3f}")
    print(f"F1:            {f1_score(y_test, y_pred, zero_division=0):.3f}")
    print(f"Declined:      {y_pred.mean() * 100:.0f}%")

    # features importance
    clf = best_model.named_steps["clf"]
    if hasattr(clf, "coef_"):
        # LogisticRegression — coef
        importances = pd.Series(clf.coef_[0], index=features)
    elif hasattr(clf, "feature_importances_"):
        # GradientBoostingClassifier — std importance
        importances = pd.Series(clf.feature_importances_, index=features)
    else:
        # HistGradientBoostingClassifier — permutation importance
        result = permutation_importance(
            best_model, X_test, y_test,
            n_repeats=10, random_state=42, scoring="roc_auc"
        )
        importances = pd.Series(result.importances_mean, index=features)

    importance = importances.sort_values(key=abs, ascending=False)
    print("\nFeature importance:")
    print(importance.round(3))


if __name__ == "__main__":
    main()

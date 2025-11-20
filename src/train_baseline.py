import pandas as pd
import numpy as np
import json
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from feature_selection import FeatureSelectionConfig, make_feature_pipeline, global_feature_engineering

def build_models() -> dict[str, object]:
    models: dict[str, object] = {}
    
    # 1. Try to load optimized LGBM params
    lgbm_params_path = "spaceship-titanic/lgbm_params.json"
    if os.path.exists(lgbm_params_path):
        print(f"Loading tuned LGBM parameters from {lgbm_params_path}...")
        with open(lgbm_params_path, "r") as f:
            lgbm_params = json.load(f)
        # Ensure required non-tunable params are present
        lgbm_params["objective"] = "binary"
        lgbm_params["random_state"] = 42
        lgbm_params["verbose"] = -1
        models["lightgbm"] = LGBMClassifier(**lgbm_params)
    else:
        print("No tuned parameters found. Using defaults for LGBM.")
        models["lightgbm"] = LGBMClassifier(
            n_estimators = 1000,
            learning_rate = 0.05,
            num_leaves = 31,
            objective = 'binary',
            random_state = 42,
            verbose = -1,
        )
    
    models["cat"] = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        random_seed=42,
        verbose=False,
    )
    
    models["xgb"] = XGBClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        eval_metric='logloss',
    )
    
    return models

def main() -> None:
    train = pd.read_csv("spaceship-titanic/train.csv")
    test = pd.read_csv("spaceship-titanic/test.csv")
    
    # 1. Apply Global Engineering FIRST
    print("Applying global feature engineering...")
    train, test = global_feature_engineering(train, test)
    
    y = train["Transported"].astype(int)
    
    # 2. Update Config with NEW features
    config = FeatureSelectionConfig(
        numerical_features=[
            "Age",
            "RoomService",
            "FoodCourt",
            "ShoppingMall",
            "Spa",
            "VRDeck",
            "TotalSpending",
            "GroupSize",
            "FamilySize",
            "Num",
            "CabinRegion",
        ],
        categorical_features=[
            "HomePlanet",
            "CryoSleep",
            "Destination",
            "VIP",
            "Deck",
            "Side",
            "IsSolo",
        ],
        poly_degree=1,
        k_best=0, 
    )
    
    pipe = make_feature_pipeline(config, columns_to_drop=["PassengerId", "Name", "Transported"])
    
    from sklearn.pipeline import Pipeline

    models = build_models()

    cv_results: dict[str, float] = {}

    best_name: str | None = None
    best_score: float = -1.0

    for name, model in models.items():
        clf = Pipeline(
            steps=[
                ("features", pipe),
                ("classifier", model),
            ]
        )
        scores = cross_val_score(clf, train, y, cv=5, scoring="accuracy", n_jobs=-1)
        mean_score = scores.mean()
        std_score = scores.std()
        print(f"{name:6s} | CV accuracy: {mean_score:.4f} ± {std_score:.4f}")

        cv_results[name] = mean_score

        if mean_score > best_score:
            best_score = mean_score
            best_name = name

    if best_name is None:
        raise RuntimeError("No models were built; cannot train final model.")
    
    model_names = list(models.keys())
    weights = np.array([cv_results[name] for name in model_names], dtype=float)
    weights = weights / weights.sum()
    print("\nEnsemble weights (from CV scores):")
    for name, w in zip(model_names, weights):
        print(f"  {name:6s}: {w:.3f}")

    from sklearn.pipeline import Pipeline

    probs_list = []
    for name, model in models.items():
        clf = Pipeline(
            steps=[
                ("features", pipe),
                ("classifier", model),
            ]
        )
        clf.fit(train, y)
        prob = clf.predict_proba(test)[:, 1]
        probs_list.append(prob)

    probs_array = np.stack(probs_list, axis=1)

    model_names = list(models.keys())
    weights = np.array([cv_results[name] for name in model_names], dtype=float)
    weights = weights / weights.sum()
    weighted_probs = np.average(probs_array, axis=1, weights=weights)
    weighted_preds = (weighted_probs >= 0.5).astype(int)

    submission_weighted = pd.DataFrame(
        {"PassengerId": test["PassengerId"], "Transported": weighted_preds.astype(bool)}
    )
    submission_weighted.to_csv("spaceship-titanic/submission_baseline.csv", index=False)
    print("Wrote weighted-ensemble submission to spaceship-titanic/submission_baseline.csv")

    stacking_base_models = build_models()
    stack_estimators = [(name, stacking_base_models[name]) for name in model_names]

    stacking_clf = Pipeline(
        steps=[
            ("features", pipe),
            ("classifier", StackingClassifier(
                estimators=stack_estimators,
                final_estimator=LogisticRegression(max_iter=5000, n_jobs=-1),
                n_jobs=-1,
            )),
        ]
    )

    stack_scores = cross_val_score(
        stacking_clf, train, y, cv=5, scoring="accuracy", n_jobs=-1
    )
    print(f"\nstack  | CV accuracy: {stack_scores.mean():.4f} ± {stack_scores.std():.4f}")

    stacking_clf.fit(train, y)
    stack_probs = stacking_clf.predict_proba(test)[:, 1]
    stack_preds = (stack_probs >= 0.5).astype(int)

    submission_stacking = pd.DataFrame(
        {"PassengerId": test["PassengerId"], "Transported": stack_preds.astype(bool)}
    )
    submission_stacking.to_csv("spaceship-titanic/submission_stacking.csv", index=False)
    print("Wrote stacking submission to spaceship-titanic/submission_stacking.csv")
    
if __name__ == "__main__":
    main()
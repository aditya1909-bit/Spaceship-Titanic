import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

from feature_selection import(
    FeatureSelectionConfig,
    make_feature_pipeline,
)

def build_models() -> dict[str, object]:
    models: dict[str, object] = {}

    models["logreg"] = LogisticRegression(
        max_iter=5000,
        solver="saga",
        n_jobs=-1,
    )

    models["rf"] = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=42,
    )

    models["gb"] = GradientBoostingClassifier(random_state=42)

    try:
        from xgboost import XGBClassifier

        models["xgb"] = XGBClassifier(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=-1,
            random_state=42,
        )
    except ImportError:
        pass

    return models

def main() -> None:
    train = pd.read_csv("spaceship-titanic/train.csv")
    test = pd.read_csv("spaceship-titanic/test.csv")
    
    y = train["Transported"].astype(int)
    
    config = FeatureSelectionConfig(
        numerical_features=[
            "Age",
            "RoomService",
            "FoodCourt",
            "ShoppingMall",
            "Spa",
            "VRDeck",
        ],
        categorical_features=[
            "HomePlanet",
            "CryoSleep",
            "Cabin",
            "Destination",
            "VIP",
        ],
        k_best=80,
    )
    
    pipe = make_feature_pipeline(config, columns_to_drop=["PassengerId", "Name", "Transported"])
    
    from sklearn.pipeline import Pipeline

    models = build_models()

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

        if mean_score > best_score:
            best_score = mean_score
            best_name = name

    if best_name is None:
        raise RuntimeError("No models were built; cannot train final model.")

    print(f"\nUsing best model: {best_name} (CV accuracy={best_score:.4f})")

    best_model = models[best_name]
    best_clf = Pipeline(
        steps=[
            ("features", pipe),
            ("classifier", best_model),
        ]
    )

    best_clf.fit(train, y)
    preds = best_clf.predict(test)
    
    submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Transported": preds.astype(bool)})
    submission.to_csv("spaceship-titanic/submission_baseline.csv", index=False)
    print("Wrote submission to spaceship-titanic/submission_baseline.csv")
    
if __name__ == "__main__":
    main()
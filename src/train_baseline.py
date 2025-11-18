import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from feature_selection import(
    FeatureSelectionConfig,
    make_feature_pipeline,
)

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
    
    clf = Pipeline(
        steps=[
            ("features", pipe),
            ("classifier", LogisticRegression(max_iter=10000, solver = "saga", n_jobs=-1)),
        ]
    )
    
    scores = cross_val_score(clf, train, y, cv=5, scoring="accuracy", n_jobs=-1)
    print(f"Cross-validated accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
    
    clf.fit(train, y)
    preds = clf.predict(test)
    
    submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Transported": preds.astype(bool)})
    submission.to_csv("spaceship-titanic/submission_baseline.csv", index=False)
    print("Wrote submission to data/processed/submission_baseline.csv")
    
if __name__ == "__main__":
    main()
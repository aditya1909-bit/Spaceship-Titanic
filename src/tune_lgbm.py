import optuna
import pandas as pd
import json
import os
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from feature_selection import FeatureSelectionConfig, make_feature_pipeline, global_feature_engineering

def objective(trial):
    # Suggest parameters
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 500, 2000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
        "num_leaves": trial.suggest_int("num_leaves", 20, 100),
        "max_depth": trial.suggest_int("max_depth", 4, 15),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "random_state": 42,
        "verbose": -1,
        "n_jobs": 1
    }

    train = pd.read_csv("spaceship-titanic/train.csv")
    test = pd.read_csv("spaceship-titanic/test.csv")
    train, test = global_feature_engineering(train, test)
    y = train["Transported"].astype(int)

    config = FeatureSelectionConfig(
        numerical_features=["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "TotalSpending", "GroupSize", "FamilySize", "Num", "CabinRegion"],
        categorical_features=["HomePlanet", "CryoSleep", "Destination", "VIP", "Deck", "Side", "IsSolo"],
        poly_degree=1, k_best=0
    )
    pipe = make_feature_pipeline(config, columns_to_drop=["PassengerId", "Name", "Transported"])

    clf = Pipeline(steps=[("features", pipe), ("model", LGBMClassifier(**params))])
    
    scores = cross_val_score(clf, train, y, cv=5, scoring="accuracy", n_jobs=1)
    return scores.mean()

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50, n_jobs=-1)
    
    print("Best params:", study.best_params)
    
    out_path = "spaceship-titanic/lgbm_params.json"
    with open(out_path, "w") as f:
        json.dump(study.best_params, f, indent=4)
    print(f"Saved best parameters to {out_path}")
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer

@dataclass
class FeatureSelectionConfig:
    numerical_features: Sequence[str]
    categorical_features: Sequence[str]
    poly_degree: Optional[int] = 2
    k_best: Optional[int] = 80

def global_feature_engineering(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df["dataset"] = "train"
    test_df["dataset"] = "test"
    full = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    
    full["GroupId"] = full["PassengerId"].str.split("_").str[0]
    full["GroupSize"] = full.groupby("GroupId")["PassengerId"].transform("count")
    
    full["Surname"] = full["Name"].str.split().str[-1]
    full["FamilySize"] = full.groupby("Surname")["Surname"].transform("count")
    full["FamilySize"] = full["FamilySize"].fillna(0)
    
    full[["Deck", "Num", "Side"]] = full["Cabin"].str.split("/", expand=True)
    full["Num"] = pd.to_numeric(full["Num"], errors="coerce")
    
    full["CabinRegion"] = (full["Num"] // 300).fillna(-1).astype(int)
    
    full["IsSolo"] = (full["GroupSize"] == 1).astype(int)

    train_eng = full[full["dataset"] == "train"].drop(columns=["dataset", "Surname", "Deck", "Num", "Side"]).reset_index(drop=True)
    test_eng = full[full["dataset"] == "test"].drop(columns=["dataset", "Surname", "Deck", "Num", "Side"]).reset_index(drop=True)
    
    return train_eng, test_eng

class SpaceShipFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        if "CryoSleep" in X.columns:
            expenses = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
            valid_expenses = [c for c in expenses if c in X.columns]
            if valid_expenses:
                X.loc[X["CryoSleep"] == True, valid_expenses] = 0.0
                
        if "Cabin" in X.columns:
            X[["Deck", "Num", "Side"]] = X["Cabin"].str.split("/", expand=True)
            X["Num"] = pd.to_numeric(X["Num"], errors="coerce")
            X.drop(columns=["Cabin"], inplace=True)

        expenses = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
        valid_expenses = [c for c in expenses if c in X.columns]
        if valid_expenses:
            X["TotalSpending"] = X[valid_expenses].sum(axis=1)
            X["NoSpending"] = (X["TotalSpending"] == 0).astype(int)
    
        return X

class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop=None):
        self.columns_to_drop = columns_to_drop
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if not self.columns_to_drop:
            return X
        return X.drop(columns=[c for c in self.columns_to_drop if c in X.columns], errors="ignore")

def build_preprocessor(config: FeatureSelectionConfig) -> ColumnTransformer:
    
    numeric_pipeline_steps =[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]
    
    if config.poly_degree and config.poly_degree > 1:
        numeric_pipeline_steps.append(
            (
                "poly",
                PolynomialFeatures(
                    degree=config.poly_degree,
                    include_bias=False,
                    interaction_only=False,
                ),
            )
        )
    
    numeric_pipeline = Pipeline(steps=numeric_pipeline_steps)
    
    categorical_pipeline = Pipeline(
        steps = [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )
    
    preprocessor = ColumnTransformer(
        transformers = [
            ("num", numeric_pipeline, list(config.numerical_features)),
            ("cat", categorical_pipeline, list(config.categorical_features)),
        ],
        remainder = "drop",
        verbose_feature_names_out=False,
    )
    
    preprocessor.set_output(transform="pandas")
    
    return preprocessor

def build_selector(k_best: Optional[int]) -> Optional[SelectKBest]:
    if k_best is  None or k_best <= 0:
        return None
    return SelectKBest(score_func=mutual_info_classif, k=k_best)

def make_feature_pipeline(config: FeatureSelectionConfig, columns_to_drop: Optional[Sequence[str]] = None) -> Pipeline:
    steps = []
    
    steps.append(("engineer", SpaceShipFeatureEngineer()))
    
    if columns_to_drop:
        steps.append(("dropper", ColumnDropper(columns_to_drop)))
    
    preprocessor = build_preprocessor(config)
    steps.append(("preprocessor", preprocessor))
    
    selector = build_selector(config.k_best)
    if selector is not None:
        steps.append(("selector", selector))
    
    return Pipeline(steps=steps)

def get_feature_names(pipeline: Pipeline, config: FeatureSelectionConfig, fitted: bool = True) -> List[str]:
    
    preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]
    
    numeric_feature_names = list(config.numerical_features)
    categorical_feature_names = list(config.categorical_features)
    
    num_pipeline: pipeline = preprocessor.named_transformers_["num"]
    
    num_steps = dict(num_pipeline.steps)
    
    if "poly" in num_steps:
        poly: PolynomialFeatures = num_steps["poly"]
        num_feature_names = poly.get_feature_names_out(numeric_feature_names).tolist()
    else:
        num_feature_names = numeric_feature_names
    
    cat_pipeline: Pipeline = preprocessor.named_transformers_["cat"]
    one_hot: OneHotEncoder = cat_pipeline.named_steps["onehot"]
    
    cat_feature_names = one_hot.get_feature_names_out(categorical_feature_names).tolist()
    all_feature_names = num_feature_names + cat_feature_names
    
    if fitted and "selector" in pipeline.named_steps:
        selector: SelectKBest = pipeline.named_steps["selector"]
        mask = selector.get_support()
        all_feature_names = [name for name, keep in zip(all_feature_names, mask) if keep]
    
    return all_feature_names
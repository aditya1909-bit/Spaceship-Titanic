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
    poly_degree = 5
    k_best: Optional[int] = 50
    
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
    )
    
    return preprocessor

def build_selector(k_best: Optional[int]) -> Optional[SelectKBest]:
    if k_best is  None or k_best <= 0:
        return None
    return SelectKBest(score_func=mutual_info_classif, k=k_best)

def make_feature_pipeline(config: FeatureSelectionConfig, columns_to_drop: Optional[Sequence[str]] = None) -> Pipeline:
    steps = []
    
    if columns_to_drop:
        steps.append(("dropper", ColumnDropper(columns_to_drop)))
    
    preprocessor = build_preprocessor(config)
    
    steps.append(("preprocessor", preprocessor))
    
    selector = build_selector(config.k_best)
    if selector is not None:
        selector = build_selector(config.k_best)
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
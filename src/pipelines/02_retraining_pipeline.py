import os
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from xgboost import plot_importance
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
import hopsworks
from hopsworks.hsfs.builtin_transformations import label_encoder
from hopsworks import udf
from hsml.schema import Schema
from hsml.model_schema import ModelSchema

import warnings
warnings.filterwarnings("ignore")

from ..app.utils.functions import (
    get_mock_data,
    connect_to_hopsworks,
    get_feature_view,
    get_model,
    get_model_deployment,
    save_prediction_to_feature_store,
    prepare_inference_data,
    predict_house_price,
)

# Connect to Hopsworks and retrieve the project and feature store objects
proj, fs = connect_to_hopsworks()
model_artifact = get_model(proj, "house_price_xgboost_model", 16)
feature_view = get_feature_view(fs, "house_price_fv", 5)
deployment = get_model_deployment(proj, "house")

# Download the saved model artifacts to a local directory
saved_model_dir = model_artifact.download()

# Load the saved model
retrieved_xgboost_model = XGBRegressor()
retrieved_xgboost_model.load_model(saved_model_dir + "/model.json")

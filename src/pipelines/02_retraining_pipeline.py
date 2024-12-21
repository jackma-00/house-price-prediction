import os
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor, plot_importance
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
import hopsworks
from hsml.schema import Schema
from hsml.model_schema import ModelSchema

# Suppress warnings
warnings.filterwarnings("ignore")


# Initialize Hopsworks project and feature store
def init_hopsworks():
    proj = hopsworks.login()
    fs = proj.get_feature_store()
    mr = proj.get_model_registry()
    return fs, mr


# Create directories for model artifacts if they don't exist
def create_dirs(model_name):
    model_dir = f"{model_name}_model"
    images_dir = f"{model_dir}/images"

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    return model_dir, images_dir


# Perform hyperparameter tuning using RandomizedSearchCV
def tune_model(X_train, y_train):
    param_distributions = {
        "n_estimators": [100, 200, 300, 500],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [4, 6, 8, 10],
        "min_child_weight": [1, 5, 10],
        "gamma": [0, 0.1, 0.2, 0.3],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_lambda": [0, 1, 5],
        "reg_alpha": [0, 1, 5],
    }

    random_search = RandomizedSearchCV(
        estimator=XGBRegressor(),
        param_distributions=param_distributions,
        n_iter=50,
        scoring="r2",
        cv=3,
        verbose=2,
        random_state=42,
        n_jobs=-1,
    )

    random_search.fit(X_train, y_train)
    return random_search.best_estimator_


# Evaluate model performance
def evaluate_model(best_model, X_test, y_test):
    y_pred = best_model.predict(X_test)

    mse = mean_squared_error(y_test.iloc[:, 0], y_pred)
    r2 = r2_score(y_test.iloc[:, 0], y_pred)

    return mse, r2, y_pred


# Save model plot
def save_model_plot(df, images_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df["price"], label="Actual Price", color="blue", linewidth=2)
    plt.plot(
        df.index,
        df["predicted_price"],
        label="Predicted Price",
        color="orange",
        linestyle="--",
        linewidth=2,
    )

    plt.xlabel("Data Point Index (Sorted by Actual Price)")
    plt.ylabel("Price")
    plt.title("Actual Price vs Predicted Price (Ordered by Actual Price)")
    plt.legend()
    plt.grid(True)

    file_path = f"{images_dir}/price_hindcast.png"
    plt.savefig(file_path, format="png", dpi=300, bbox_inches="tight")
    # plt.show()


# Save feature importance plot
def save_feature_importance_plot(best_model, images_dir):
    plot_importance(best_model, importance_type="weight")
    feature_importance_path = f"{images_dir}/feature_importance.png"
    plt.savefig(feature_importance_path)
    # plt.show()


# Save model schema and model
def save_model_schema_and_artifacts(
    mr, model_dir, best_model, X_train, y_train, X_test, res_dict, feature_view
):
    input_schema = Schema(X_train)
    output_schema = Schema(y_train)
    model_schema = ModelSchema(input_schema=input_schema, output_schema=output_schema)

    best_model.save_model(f"{model_dir}/model.json")

    hp_model = mr.python.create_model(
        name="house_price_xgboost_model",
        metrics=res_dict,
        feature_view=feature_view,
        model_schema=model_schema,
        input_example=X_test.sample().values,
        description="Italian house price predictor",
    )

    hp_model.save(model_dir)


def main():
    fs, mr = init_hopsworks()

    feature_view = fs.get_feature_view("house_price_fv", 5)

    # Train-test split
    TEST_SIZE = 0.2
    X_train, X_test, y_train, y_test = feature_view.train_test_split(
        test_size=TEST_SIZE,
        description="house price training dataset",
    )

    model_dir, images_dir = create_dirs("house_price")

    # Model training and hyperparameter tuning
    best_model = tune_model(X_train, y_train)

    # Model evaluation
    mse, r2, y_pred = evaluate_model(best_model, X_test, y_test)
    print(f"MSE: {mse}, R squared: {r2}")

    # Prepare results DataFrame
    df = y_test.copy()
    df["predicted_price"] = y_pred
    df = df.sort_values(by="price").reset_index(drop=True)

    # Save plots
    save_model_plot(df, images_dir)
    save_feature_importance_plot(best_model, images_dir)

    # Prepare results for the model registry
    res_dict = {"MSE": str(mse), "R squared": str(r2)}

    # Save model and artifacts
    save_model_schema_and_artifacts(
        mr, model_dir, best_model, X_train, y_train, X_test, res_dict, feature_view
    )


if __name__ == "__main__":
    main()

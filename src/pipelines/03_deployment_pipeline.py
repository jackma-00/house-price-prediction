import os
import warnings
import hopsworks
from hsml.model_schema import ModelSchema

# Suppress warnings
warnings.filterwarnings("ignore")


# Initialize Hopsworks project and model registry
def init_hopsworks():
    proj = hopsworks.login()
    mr = proj.get_model_registry()
    ms = proj.get_model_serving()
    return proj, mr, ms


# Retrieve the latest model from the model registry
def get_latest_model(mr, model_name):
    retrieved_models = mr.get_models(model_name)
    return max(retrieved_models, key=lambda x: x.version)


# Delete the old deployment
def delete_old_deployment(ms, deployment_name):
    deployment = ms.get_deployment(deployment_name)
    deployment.delete(force=True)


# Deploy a new model
def deploy_new_model(latest_model, predictor_script_path, deployment_name):
    deployment = latest_model.deploy(
        name=deployment_name,
        script_file=predictor_script_path,
    )
    # Start the deployment and wait for it to be running, with a maximum waiting time of 180 seconds
    deployment.start(await_running=180)


def main():
    # Initialize Hopsworks components
    proj, mr, ms = init_hopsworks()

    # Get the latest model
    latest_model = get_latest_model(mr, "house_price_xgboost_model")

    # Delete the old deployment
    delete_old_deployment(ms, "house")

    # Define the script path for prediction
    predictor_script_path = os.path.join(
        "/Projects", proj.name, "Models", "predict_house_price.py"
    )

    # Deploy the new model
    deploy_new_model(latest_model, predictor_script_path, "house")
    print("New model deployed and running.")


if __name__ == "__main__":
    main()

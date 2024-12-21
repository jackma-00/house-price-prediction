from datetime import datetime
import pandas as pd
import numpy as np
import hopsworks
import uuid


def connect_to_hopsworks():
    """Connect to Hopsworks and return the project and feature store objects."""
    proj = hopsworks.login()
    fs = proj.get_feature_store()
    return proj, fs


def get_feature_view(fs, name, version):
    """Retrieve a feature view from the Hopsworks Feature Store."""
    return fs.get_feature_view(name=name, version=version)


def get_model_deployment(proj, deployment_name):
    """Retrieve the model deployment object."""
    ms = proj.get_model_serving()
    return ms.get_deployment(deployment_name)


def generate_numeric_uuid():
    """Generate a numeric UUID for unique identification."""
    raw_uuid = uuid.uuid4()
    return int(str(raw_uuid.int)[:9])


def save_prediction_to_feature_store(fs, inference_data, prediction):
    """Save prediction data to the Hopsworks Feature Store."""
    inference_data["predicted_price"] = prediction
    inference_data["id"] = generate_numeric_uuid()
    inference_data["timestamp"] = datetime.today().date()

    inference_data = pd.DataFrame(inference_data, index=[0])

    # Reorder columns to match properties feature group schema
    column_order = [
        "id",
        "timestamp",
        "agencyid",
        "bedroomsnumber",
        "buildingyear",
        "codcom",
        "gsm",
        "surface",
        "latitude",
        "longitude",
        "isluxury",
        "isnew",
        "on_the_market",
        "zeroenergybuilding",
        "airconditioning",
        "bathrooms",
        "city",
        "condition",
        "energyclass",
        "ga4heating",
        "garage",
        "heatingtype",
        "pricerange",
        "rooms",
        "id_zona_omi",
        "predicted_price",
    ]
    inference_data = inference_data[column_order]

    property_preds = fs.get_or_create_feature_group(
        name="property_preds",
        version=4,
        description="Property predicted prices",
        primary_key=["id"],
        event_time="timestamp",
    )

    property_preds.insert(inference_data)


def prepare_inference_data(**kwargs):
    """Prepare the inference data dictionary."""
    return {
        key: int(value) if isinstance(value, bool) else value
        for key, value in kwargs.items()
    }


def predict_house_price(deployment, feature_view, inference_data):
    """Make predictions using the deployed model."""
    transformed_data = feature_view.get_feature_vector(
        entry={"id_zona_omi": inference_data["id_zona_omi"]},
        passed_features=inference_data,
        return_type="list",
    )

    transformed_data = [
        int(x) if isinstance(x, np.int64) else x for x in transformed_data
    ]

    predictions = deployment.predict({"instances": [transformed_data]})
    return predictions["predictions"][0]


def get_mock_data():
    return [example_1, example_2, example_3]


# Mock data
example_1 = [
    169110.0,
    3.0,
    2023.0,
    26086.0,
    181.0,
    253,
    45.6674,
    12.244,
    True,  # Changed to int
    False,  # Changed to int
    False,  # Changed to int
    False,  # Changed to int
    "autonomo, freddo",
    "3",
    "Treviso",
    "Nuovo / In costruzione",
    "A2",
    "Autonomo",
    "1 in box privato/box in garage",
    "autonomo, a pavimento",
    "oltre 500.000 €",
    "F704-B11",
    "4",
]

# Price 149000.0
example_2 = [
    75834.0,
    3.0,
    1960.0,
    40012.0,
    96.0,
    150,
    44.1684,
    12.0798,
    False,
    False,
    False,
    False,
    "autonomo, freddo/caldo",
    "2",
    "Forlì",
    "Da ristrutturare",
    "G",
    "Autonomo",
    "1 in box privato/box in garage",
    "autonomo, a radiatori, alimentato a gas",
    "100.001 - 150.000 &euro;",
    "D704-E2",
    "5",
]

# Price 3100000.0
example_3 = [
    104966.0,
    4.0,
    1890.0,
    48017.0,
    230.0,
    300,
    43.7656,
    11.2348,
    True,
    False,
    False,
    False,
    "autonomo, freddo/caldo",
    "3+",
    "Firenze",
    "Ottimo / Ristrutturato",
    "D",
    "Autonomo",
    "1 in box privato/box in garage, 1 in parcheggio/garage comune",
    "autonomo, a radiatori, alimentato a gas",
    "oltre 500.000 &euro;",
    "D612-C11",
    "5+",
]

from xgboost import XGBRegressor
from datetime import datetime
import gradio as gr
import pandas as pd
import numpy as np
import hopsworks
import uuid
import os


# Connect to Hopsworks Feature Store
with open("src/app/data/hopsworks-api-key.txt", "r") as file:
    os.environ["HOPSWORKS_API_KEY"] = file.read().rstrip()
proj = hopsworks.login()

# Get Feature View
fs = proj.get_feature_store()
feature_view = fs.get_feature_view(
    name="house_price_fv",
    version=5,
)

# Get model deployment
ms = proj.get_model_serving()
deployment = ms.get_deployment("house")


def predict_price(
    agencyid,
    bedroomsnumber,
    buildingyear,
    codcom,
    gsm,
    surface,
    latitude,
    longitude,
    isluxury,
    isnew,
    on_the_market,
    zeroenergybuilding,
    airconditioning,
    bathrooms,
    city,
    condition,
    energyclass,
    ga4heating,
    garage,
    heatingtype,
    pricerange,
    id_zona_omi,
    rooms,
):
    # Prepare inference data dictionary
    inference_data = {
        "agencyid": agencyid,
        "bedroomsnumber": bedroomsnumber,
        "buildingyear": buildingyear,
        "codcom": codcom,
        "gsm": gsm,
        "surface": surface,
        "latitude": latitude,
        "longitude": longitude,
        "isluxury": int(isluxury),
        "isnew": int(isnew),
        "on_the_market": int(on_the_market),
        "zeroenergybuilding": int(zeroenergybuilding),
        "airconditioning": airconditioning,
        "bathrooms": bathrooms,
        "city": city,
        "condition": condition,
        "energyclass": energyclass,
        "ga4heating": ga4heating,
        "garage": garage,
        "heatingtype": heatingtype,
        "pricerange": pricerange,
        "id_zona_omi": id_zona_omi,
        "rooms": rooms,
    }

    # Apply Model-dependent transformations to the inference data
    transformed_data = feature_view.get_feature_vector(
        entry={"id_zona_omi": inference_data["id_zona_omi"]},
        passed_features=inference_data,
        return_type="list",
    )

    # Convert NumPy int64 to native Python int
    transformed_data_python = [
        int(x) if isinstance(x, np.int64) else x for x in transformed_data
    ]

    # Make predictions using the deployed model
    predictions = deployment.predict(
        {"instances": [transformed_data_python]},
    )

    return predictions["predictions"][0]


demo = gr.Interface(
    fn=predict_price,
    inputs=[
        "number",
        "number",
        "number",
        "number",
        "number",
        "number",
        "number",
        "number",
        "checkbox",
        "checkbox",
        "checkbox",
        "checkbox",
        "text",
        "text",
        "text",
        "text",
        "text",
        "text",
        "text",
        "text",
        "text",
        "text",
        "text",
    ],
    outputs=[gr.Number(label="price")],
    examples=[
        [
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
        ],
    ],
    title="Italian House Price Predictor",
    description="Enter house details.",
)


demo.launch()

import gradio as gr
import threading

from app.utils.functions import (
    get_mock_data,
    connect_to_hopsworks,
    get_feature_view,
    get_model_deployment,
    save_prediction_to_feature_store,
    prepare_inference_data,
    predict_house_price,
)


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
    """Main prediction function invoked by the Gradio interface."""
    proj, fs = connect_to_hopsworks()
    feature_view = get_feature_view(fs, "house_price_fv", 5)
    deployment = get_model_deployment(proj, "house")

    inference_data = prepare_inference_data(
        agencyid=agencyid,
        bedroomsnumber=bedroomsnumber,
        buildingyear=buildingyear,
        codcom=codcom,
        gsm=gsm,
        surface=surface,
        latitude=latitude,
        longitude=longitude,
        isluxury=isluxury,
        isnew=isnew,
        on_the_market=on_the_market,
        zeroenergybuilding=zeroenergybuilding,
        airconditioning=airconditioning,
        bathrooms=bathrooms,
        city=city,
        condition=condition,
        energyclass=energyclass,
        ga4heating=ga4heating,
        garage=garage,
        heatingtype=heatingtype,
        pricerange=pricerange,
        id_zona_omi=id_zona_omi,
        rooms=rooms,
    )

    prediction = predict_house_price(deployment, feature_view, inference_data)

    # Save prediction asynchronously
    threading.Thread(
        target=save_prediction_to_feature_store, args=(fs, inference_data, prediction)
    ).start()

    return prediction


def create_gradio_interface():
    """Create and launch the Gradio interface."""
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
        examples=get_mock_data(),
        title="Italian House Price Predictor",
        description="Enter house details.",
        theme="soft",
    )
    demo.launch()


if __name__ == "__main__":
    create_gradio_interface()

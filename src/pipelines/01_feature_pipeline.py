import pandas as pd
import hopsworks
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")


def init_hopsworks():
    """Initialize and return the Hopsworks project and feature store."""
    proj = hopsworks.login()
    return proj.get_feature_store()


def fetch_new_data(filepath="data/new_data.csv"):
    """Fetch new data from a specified source.

    Args:
        filepath (str): Path to the CSV file containing new data.

    Returns:
        pd.DataFrame: DataFrame containing the new data.
    """
    return pd.read_csv(filepath)


def preprocess_properties_data(properties_df):
    """Preprocess the properties DataFrame.

    Args:
        properties_df (pd.DataFrame): Raw properties data.

    Returns:
        pd.DataFrame: Preprocessed properties DataFrame.
    """
    # Drop missing values
    properties_df.dropna(inplace=True)

    # Convert column names to lowercase
    properties_df.columns = properties_df.columns.str.lower()

    # Select relevant columns
    selected_columns = [
        "id",
        "scraping_date",
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
        "price",
    ]
    properties_df = properties_df[selected_columns]

    # Convert boolean features to integers
    boolean_columns = ["isluxury", "isnew", "on_the_market", "zeroenergybuilding"]
    properties_df[boolean_columns] = properties_df[boolean_columns].astype(int)

    # Parse and sort by scraping_date
    properties_df["scraping_date"] = pd.to_datetime(properties_df["scraping_date"])
    properties_df.sort_values(by="scraping_date", inplace=True)

    return properties_df


def save_to_feature_store(fs, properties_df):
    """Save the properties DataFrame to the feature store.

    Args:
        fs: Feature store object.
        properties_df (pd.DataFrame): Preprocessed properties data.
    """
    properties_fg = fs.get_or_create_feature_group(
        name="properties",
        version=4,
        description="Property Features and Corresponding Prices",
        online_enabled=True,
        primary_key=["id"],
        event_time="scraping_date",
    )
    properties_fg.insert(
        properties_df, wait=True
    )  # Wait for data materialization to finish before returning


def main():
    """Main function to execute the data pipeline."""
    # Initialize feature store
    fs = init_hopsworks()

    # Fetch and preprocess new data
    raw_data = fetch_new_data()
    properties_data = preprocess_properties_data(raw_data)

    # Save processed data to the feature store
    save_to_feature_store(fs, properties_data)


if __name__ == "__main__":
    main()

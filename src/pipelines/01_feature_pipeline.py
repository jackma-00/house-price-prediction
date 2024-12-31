import os
import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd
import hopsworks
import warnings
from datetime import datetime, timedelta

# Suppress warnings
warnings.filterwarnings("ignore")


def fetch_sales_data(query_date):
    """
    Fetch data from the sale table in the scraping schema.

    Args:
        query_date (str): The date to filter the results by scaping_date.

    Returns:
        list: A list of rows from the table.
    """
    # Load database credentials from environment variables
    DB_NAME = os.getenv("DB_NAME")
    DB_USER = os.getenv("DB_USER")
    DB_PASS = os.getenv("DB_PASS")
    DB_HOST = os.getenv("DB_HOST")
    DB_PORT = os.getenv("DB_PORT")

    connection = None
    try:
        # Establish database connection
        connection = psycopg2.connect(
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASS,
            host=DB_HOST,
            port=DB_PORT
        )
        print("Database connected successfully")

        # Create a cursor
        cursor = connection.cursor(cursor_factory=RealDictCursor)

        # Define and execute the query
        query = """
        SELECT s.*, i.*
        from scraping.sale s
        INNER JOIN omi."Interpolated" i
        ON ST_Contains(i.geometry , ST_SetSRID(ST_Point(s.longitude, s.latitude), 4326))
        WHERE s.scraping_date > %s
          AND i.condition_omi = 'scadente'
          AND i.typology_omi = 'Ville e Villini'
        """

        cursor.execute(query, (query_date,))

        # Fetch and return results
        results = cursor.fetchall()
        return pd.DataFrame(results)

    except psycopg2.Error as e:
        print(f"Database error: {e}")
        return []

    finally:
        if connection:
            connection.close()
            print("Database connection closed")


def init_hopsworks():
    """Initialize and return the Hopsworks project and feature store."""
    proj = hopsworks.login()
    return proj.get_feature_store()


def preprocess_properties_data(properties_df):
    """
    Preprocess the properties DataFrame.

    Args:
        properties_df (pd.DataFrame): Raw properties data.

    Returns:
        pd.DataFrame: Preprocessed properties DataFrame.
    """

    # Convert column names to lowercase
    properties_df.columns = properties_df.columns.str.lower()

    # Rename column 'comuneid' to 'codcom'
    properties_df.rename(columns={"comuneid": "codcom"}, inplace=True)

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

    # Drop missing values
    properties_df.dropna(inplace=True)

    # Convert boolean features to integers
    boolean_columns = ["isluxury", "isnew", "on_the_market", "zeroenergybuilding"]
    properties_df[boolean_columns] = properties_df[boolean_columns].astype(int)

    # Convert codcom from bigint to double
    properties_df["codcom"] = properties_df["codcom"].astype(float)

    # Parse and sort by scraping_date
    properties_df["scraping_date"] = pd.to_datetime(properties_df["scraping_date"])
    properties_df.sort_values(by="scraping_date", inplace=True)

    return properties_df


def save_to_feature_store(fs, properties_df):
    """
    Save the properties DataFrame to the feature store.

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

    # Fetch data from PostgreSQL
    query_date = (datetime.now().date() - timedelta(days=7)).strftime("%Y-%m-%d")
    raw_data = fetch_sales_data(query_date)

    if raw_data.empty:
        print("No data fetched from the database.")
        return

    # Preprocess fetched data
    properties_data = preprocess_properties_data(raw_data)

    # Save processed data to the feature store
    save_to_feature_store(fs, properties_data)


if __name__ == "__main__":
    main()

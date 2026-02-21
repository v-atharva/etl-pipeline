from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime, timedelta
import os
import sys

# Default arguments for the DAG
default_args = {
    "owner": "candy_store",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def _setup_java_and_env():
    """
    Set JAVA_HOME to Java 17 and load .env.
    Must be called inside every task that uses PySpark,
    because each Airflow task runs in its own subprocess.
    """
    from dotenv import load_dotenv
    load_dotenv()

    java17_home = "/opt/homebrew/opt/openjdk@17"
    os.environ["JAVA_HOME"] = java17_home

    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable


def _create_spark_session(app_name="CandyStoreAirflow"):
    """Create a Spark session. Must be called after _setup_java_and_env()."""
    from pyspark.sql import SparkSession

    return (
        SparkSession.builder.appName(app_name)
        .config(
            "spark.jars.packages",
            "org.mongodb.spark:mongo-spark-connector_2.12:3.0.1",
        )
        .config("spark.jars", os.getenv("MYSQL_CONNECTOR_PATH"))
        .config("spark.mongodb.input.uri", os.getenv("MONGODB_URI"))
        .getOrCreate()
    )


def _get_config():
    """Build the config dict from env vars (after load_dotenv)."""
    project_root = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
    output_path = os.path.join(
        project_root, os.getenv("OUTPUT_PATH", "data/output")
    )
    return {
        "mongodb_uri": os.getenv("MONGODB_URI"),
        "mongodb_db": os.getenv("MONGO_DB"),
        "mongodb_collection_prefix": os.getenv("MONGO_COLLECTION_PREFIX"),
        "mysql_url": os.getenv("MYSQL_URL"),
        "mysql_user": os.getenv("MYSQL_USER"),
        "mysql_password": os.getenv("MYSQL_PASSWORD"),
        "mysql_db": os.getenv("MYSQL_DB"),
        "customers_table": os.getenv("CUSTOMERS_TABLE"),
        "products_table": os.getenv("PRODUCTS_TABLE"),
        "output_path": output_path,
        "reload_inventory_daily": os.getenv(
            "RELOAD_INVENTORY_DAILY", "false"
        ).lower()
        == "true",
    }


# ──────────────────────────────────────────────
# Validating configuration
# ──────────────────────────────────────────────
def load_configuration(**kwargs):
    """Task 1: Load and validate .env configuration."""
    from dotenv import load_dotenv
    load_dotenv()

    start_date = os.getenv("MONGO_START_DATE")
    end_date = os.getenv("MONGO_END_DATE")

    required = [
        "MONGODB_URI", "MONGO_DB", "MYSQL_URL",
        "MYSQL_USER", "MYSQL_DB", "MYSQL_CONNECTOR_PATH",
        "MONGO_START_DATE", "MONGO_END_DATE",
    ]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        raise ValueError(f"Missing required env vars: {missing}")

    kwargs["ti"].xcom_push(key="start_date", value=start_date)
    kwargs["ti"].xcom_push(key="end_date", value=end_date)
    print(f"Configuration loaded. Date range: {start_date} → {end_date}")


# ──────────────────────────────────────────────
# Loading MySQL data (self-contained)
# ──────────────────────────────────────────────
def import_mysql_data(**kwargs):
    """Task 2: Create Spark session and load MySQL customer/product data."""
    from data_processor import DataProcessor

    _setup_java_and_env()
    config = _get_config()

    spark = _create_spark_session()
    try:
        processor = DataProcessor(spark)
        processor.configure(config)
        processor.load_mysql_data()
        print(f"MySQL data loaded: {len(processor.products_data)} products, "
              f"{processor.customers_df.count()} customers")
    finally:
        spark.stop()

    print("MySQL data import task complete.")


# ──────────────────────────────────────────────
# Verifying MongoDB connectivity
# ──────────────────────────────────────────────
def import_mongodb_data(**kwargs):
    """Task 3: Verify MongoDB connection and data availability."""
    from data_processor import DataProcessor
    from datetime import datetime, timedelta

    _setup_java_and_env()
    config = _get_config()

    ti = kwargs["ti"]
    start_date = ti.xcom_pull(key="start_date", task_ids="load_config")
    end_date = ti.xcom_pull(key="end_date", task_ids="load_config")

    # Generate date range
    start = datetime.strptime(start_date, "%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")
    date_range = []
    current = start
    while current <= end:
        date_range.append(current.strftime("%Y%m%d"))
        current += timedelta(days=1)

    spark = _create_spark_session()
    try:
        processor = DataProcessor(spark)
        processor.configure(config)
        # Test loading first day to verify connection
        test_df = processor.load_mongodb_transactions(date_range[0])
        count = test_df.count()
        print(f"MongoDB verified. First day ({date_range[0]}): {count} transactions. "
              f"Date range: {date_range[0]} → {date_range[-1]}")
    finally:
        spark.stop()


# ──────────────────────────────────────────────
# Processing all order batches and save CSVs
# ──────────────────────────────────────────────
def process_orders_batch(**kwargs):
    """Task 4: Run the full ETL — loads MySQL + MongoDB, processes 10 days, saves CSVs."""
    from data_processor import DataProcessor
    from datetime import datetime, timedelta

    _setup_java_and_env()
    config = _get_config()

    ti = kwargs["ti"]
    start_date = ti.xcom_pull(key="start_date", task_ids="load_config")
    end_date = ti.xcom_pull(key="end_date", task_ids="load_config")

    # Build date range
    start = datetime.strptime(start_date, "%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")
    date_range = []
    current = start
    while current <= end:
        date_range.append(current.strftime("%Y%m%d"))
        current += timedelta(days=1)

    spark = _create_spark_session()
    try:
        processor = DataProcessor(spark)
        processor.configure(config)
        processor.load_mysql_data()
        processor.initialize_inventory()

        all_orders_rows = []
        all_items_rows = []
        total_cancelled = 0

        for date_str in date_range:
            transactions_df = processor.load_mongodb_transactions(date_str)
            orders_rows, items_rows, cancelled = processor.process_daily_batch(
                transactions_df, date_str
            )
            all_orders_rows.extend(orders_rows)
            all_items_rows.extend(items_rows)
            total_cancelled += cancelled
            print(f"  {date_str}: {len(orders_rows)} orders, {cancelled} cancelled items")

        processor.total_cancelled_items = total_cancelled
        processor.finalize_processing()

        output_path = config["output_path"]
        os.makedirs(output_path, exist_ok=True)

        orders_df = processor.build_orders_dataframe(all_orders_rows)
        processor.save_to_csv(orders_df, output_path, "orders.csv")

        order_items_df = processor.build_order_line_items_dataframe(all_items_rows)
        processor.save_to_csv(order_items_df, output_path, "order_line_items.csv")

        daily_summary_df = processor.create_daily_summary(all_orders_rows, all_items_rows)
        processor.save_to_csv(daily_summary_df, output_path, "daily_summary.csv")

        products_updated_df = processor.create_products_updated()
        processor.save_to_csv(products_updated_df, output_path, "products_updated.csv")

        print(f"All output files saved to: {output_path}")
        print(f"Total cancelled items across all days: {total_cancelled}")

    finally:
        spark.stop()


# ──────────────────────────────────────────────
# Generating time-series forecast
# ──────────────────────────────────────────────
def forecast_sales_profits(**kwargs):
    """Load daily_summary.csv and run Prophet forecasting."""
    from data_processor import DataProcessor

    _setup_java_and_env()
    config = _get_config()

    output_path = config["output_path"]
    daily_summary_path = os.path.join(output_path, "daily_summary.csv")

    if not os.path.exists(daily_summary_path):
        raise FileNotFoundError(
            f"daily_summary.csv not found at {daily_summary_path}. "
            "process_orders_batch must run first."
        )

    spark = _create_spark_session()
    try:
        processor = DataProcessor(spark)
        processor.configure(config)

        # Read the saved daily summary back as a Spark DataFrame
        daily_summary_df = spark.read.csv(
            daily_summary_path, header=True, inferSchema=True
        )

        forecast_df = processor.forecast_sales_and_profits(daily_summary_df)
        if forecast_df is not None:
            processor.save_to_csv(forecast_df, output_path, "sales_profit_forecast.csv")
            print("Forecasting complete. sales_profit_forecast.csv saved.")
        else:
            print("Warning: Forecast returned None — check Prophet input data.")

    finally:
        spark.stop()


# ──────────────────────────────────────────────
# DAG definition
# ──────────────────────────────────────────────
with DAG(
    dag_id="candy_store_etl_pipeline",
    default_args=default_args,
    description="Batch Processing ETL Pipeline for Candy Store",
    schedule=None,
    start_date=datetime(2024, 2, 1),
    catchup=False,
    tags=["candy_store", "etl", "batch_processing"],
) as dag:

    load_config_task = PythonOperator(
        task_id="load_config",
        python_callable=load_configuration,
    )

    import_mysql_task = PythonOperator(
        task_id="import_mysql_data",
        python_callable=import_mysql_data,
    )

    import_mongodb_task = PythonOperator(
        task_id="import_mongodb_data",
        python_callable=import_mongodb_data,
    )

    process_orders_task = PythonOperator(
        task_id="process_orders_batch",
        python_callable=process_orders_batch,
    )

    forecast_task = PythonOperator(
        task_id="forecast_sales_profits",
        python_callable=forecast_sales_profits,
    )

    # Task dependency chain
    (
        load_config_task
        >> import_mysql_task
        >> import_mongodb_task
        >> process_orders_task
        >> forecast_task
    )

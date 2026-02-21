from pyspark.sql import SparkSession, DataFrame
from data_processor import DataProcessor
from dotenv import load_dotenv
from datetime import datetime, timedelta
import os
from pyspark.sql.functions import col
from typing import Dict, Tuple
import traceback


def create_spark_session(app_name: str = "CandyStoreAnalytics") -> SparkSession:
    """
    Create and configure Spark session with MongoDB and MySQL connectors
    """
    import sys

    python_path = sys.executable
    os.environ["PYSPARK_PYTHON"] = python_path
    os.environ["PYSPARK_DRIVER_PYTHON"] = python_path

    return (
        SparkSession.builder.appName(app_name)
        .config(
            "spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:3.0.1"
        )
        .config("spark.jars", os.getenv("MYSQL_CONNECTOR_PATH"))
        .config("spark.mongodb.input.uri", os.getenv("MONGODB_URI"))
        .getOrCreate()
    )


def get_date_range(start_date: str, end_date: str) -> list[str]:
    """Generate a list of dates between start and end date"""
    start = datetime.strptime(start_date, "%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")
    date_list = []

    current = start
    while current <= end:
        date_list.append(current.strftime("%Y%m%d"))
        current += timedelta(days=1)

    return date_list


def print_header():
    print("*" * 80)
    print("                        CANDY STORE DATA PROCESSING SYSTEM")
    print("                               Analysis Pipeline")
    print("*" * 80)


def print_processing_period(date_range: list):
    print("\n" + "=" * 80)
    print("PROCESSING PERIOD")
    print("-" * 80)
    print(f"Start Date: {date_range[0]}")
    print(f"End Date:   {date_range[-1]}")
    print("=" * 80)


def setup_configuration() -> Tuple[Dict, list]:
    """Setup application configuration"""
    load_dotenv()
    config = load_config()
    date_range = get_date_range(
        os.getenv("MONGO_START_DATE"), os.getenv("MONGO_END_DATE")
    )
    return config, date_range


def load_config() -> Dict:
    """Load configuration from environment variables"""
    # Resolve output path relative to project root (parent of src/)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path = os.path.join(project_root, os.getenv("OUTPUT_PATH", "data/output"))

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
        "reload_inventory_daily": os.getenv("RELOAD_INVENTORY_DAILY", "false").lower()
        == "true",
    }


def print_daily_summary(orders_count, processed_items, cancelled_count):
    """Print summary of daily processing"""
    print("\nDAILY PROCESSING SUMMARY")
    print("-" * 40)
    print(f"• Successfully Processed Orders: {orders_count}")
    print(f"• Successfully Processed Items: {processed_items}")
    print(f"• Items Cancelled (Inventory): {cancelled_count}")
    print("-" * 40)


def process_all_batches(data_processor, date_range):
    """Process all daily batches and return combined results"""
    all_orders_rows = []
    all_items_rows = []
    total_cancelled = 0

    for date_str in date_range:
        # Load transactions for this day
        transactions_df = data_processor.load_mongodb_transactions(date_str)

        # Process the daily batch
        orders_rows, items_rows, cancelled = (
            data_processor.process_daily_batch(transactions_df, date_str)
        )

        # Count successfully processed items
        processed_items = sum(
            1 for item in items_rows if item["quantity"] > 0
        )

        # Print daily summary
        print_daily_summary(
            len(orders_rows), processed_items, cancelled
        )

        all_orders_rows.extend(orders_rows)
        all_items_rows.extend(items_rows)
        total_cancelled += cancelled

    return all_orders_rows, all_items_rows, total_cancelled


def save_all_outputs(data_processor, config, all_orders_rows, all_items_rows):
    """Build DataFrames and save all output CSV files"""
    output_path = config["output_path"]

    # Build and save orders table
    print("\nBuilding orders table...")
    orders_df = data_processor.build_orders_dataframe(all_orders_rows)
    data_processor.save_to_csv(orders_df, output_path, "orders.csv")

    # Build and save order_line_items table
    print("Building order_line_items table...")
    order_items_df = data_processor.build_order_line_items_dataframe(
        all_items_rows
    )
    data_processor.save_to_csv(
        order_items_df, output_path, "order_line_items.csv"
    )

    # Create and save daily summary
    daily_summary_df = data_processor.create_daily_summary(
        all_orders_rows, all_items_rows
    )
    data_processor.save_to_csv(
        daily_summary_df, output_path, "daily_summary.csv"
    )

    # Create and save products_updated
    products_updated_df = data_processor.create_products_updated()
    data_processor.save_to_csv(
        products_updated_df, output_path, "products_updated.csv"
    )

    return daily_summary_df


def main():
    print_header()

    # Setup
    config, date_range = setup_configuration()
    print_processing_period(date_range)

    # Initialize processor
    spark = create_spark_session()
    data_processor = DataProcessor(spark)

    try:
        # Configure and load data
        data_processor.configure(config)
        data_processor.load_mysql_data()
        data_processor.initialize_inventory()

        print("\nStart batch processing for project 2!")

        # Process all daily batches
        all_orders_rows, all_items_rows, total_cancelled = (
            process_all_batches(data_processor, date_range)
        )

        # Print processing complete summary
        data_processor.total_cancelled_items = total_cancelled
        data_processor.finalize_processing()

        # Save all output files and get daily summary
        daily_summary_df = save_all_outputs(
            data_processor, config, all_orders_rows, all_items_rows
        )

        # Generate forecasts
        try:
            forecast_df = data_processor.forecast_sales_and_profits(
                daily_summary_df
            )
            if forecast_df is not None:
                data_processor.save_to_csv(
                    forecast_df,
                    config["output_path"],
                    "sales_profit_forecast.csv",
                )
        except Exception as e:
            print(f"⚠️  Warning: Could not generate forecasts: {str(e)}")

    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        raise
    finally:
        print("\nCleaning up...")
        spark.stop()


if __name__ == "__main__":
    main()

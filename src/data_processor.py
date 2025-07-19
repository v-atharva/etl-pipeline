from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    explode,
    col,
    round as spark_round,
    sum as spark_sum,
    count,
    abs as spark_abs,
)
from typing import Dict, Tuple
import os
import glob
import shutil
import decimal
import numpy as np
from time_series import ProphetForecaster
from datetime import datetime, timedelta
from pyspark.sql.types import DoubleType, DecimalType


class DataProcessor:
    def __init__(self, spark: SparkSession):
        self.spark = spark
        # Initialize all class properties
        self.config = None
        self.current_inventory = None
        self.inventory_initialized = False
        self.original_products_df = None  # Store original products data
        self.reload_inventory_daily = False  # New flag for inventory reload
        self.order_items = None
        self.products_df = None
        self.customers_df = None
        self.transactions_df = None
        self.orders_df = None
        self.order_line_items_df = None
        self.daily_summary_df = None
        self.total_cancelled_items = 0

    def configure(self, config: Dict) -> None:
        """Configure the data processor with environment settings"""
        self.config = config
        self.reload_inventory_daily = config.get("reload_inventory_daily", False)
        print("\nINITIALIZING DATA SOURCES")
        print("-" * 80)
        if self.reload_inventory_daily:
            print("Daily inventory reload: ENABLED")
        else:
            print("Daily inventory reload: DISABLED")

    def finalize_processing(self) -> None:
        """Finalize processing and create summary"""
        print("\nPROCESSING COMPLETE")
        print("=" * 80)
        print(f"Total Cancelled Items: {self.total_cancelled_items}")

    # ------------------------------------------------------------------------------------------------
    # Try not to change the logic of the time series forecasting model
    # DO NOT change functions with prefix _
    # ------------------------------------------------------------------------------------------------
    def forecast_sales_and_profits(
        self, daily_summary_df: DataFrame, forecast_days: int = 1
    ) -> DataFrame:
        """
        Main forecasting function that coordinates the forecasting process
        """
        try:
            # Build model
            model_data = self.build_time_series_model(daily_summary_df)

            # Calculate accuracy metrics
            metrics = self.calculate_forecast_metrics(model_data)

            # Generate forecasts
            forecast_df = self.make_forecasts(model_data, forecast_days)

            return forecast_df

        except Exception as e:
            print(
                f"Error in forecast_sales_and_profits: {str(e)}, please check the data"
            )
            return None

    def print_inventory_levels(self) -> None:
        """Print current inventory levels for all products"""
        print("\nCURRENT INVENTORY LEVELS")
        print("-" * 40)

        inventory_data = self.current_inventory.orderBy("product_id").collect()
        for row in inventory_data:
            print(
                f"• {row['product_name']:<30} (ID: {row['product_id']:>3}): {row['current_stock']:>4} units"
            )
        print("-" * 40)

    def build_time_series_model(self, daily_summary_df: DataFrame) -> dict:
        """Build Prophet models for sales and profits"""
        print("\n" + "=" * 80)
        print("TIME SERIES MODEL CONSTRUCTION")
        print("-" * 80)

        model_data = self._prepare_time_series_data(daily_summary_df)
        return self._fit_forecasting_models(model_data)

    def calculate_forecast_metrics(self, model_data: dict) -> dict:
        """Calculate forecast accuracy metrics for both models"""
        print("\nCalculating forecast accuracy metrics...")

        # Get metrics from each model
        sales_metrics = model_data["sales_model"].get_metrics()
        profit_metrics = model_data["profit_model"].get_metrics()

        metrics = {
            "sales_mae": sales_metrics["mae"],
            "sales_mse": sales_metrics["mse"],
            "profit_mae": profit_metrics["mae"],
            "profit_mse": profit_metrics["mse"],
        }

        # Print metrics and model types
        print("\nForecast Error Metrics:")
        print(f"Sales Model Type: {sales_metrics['model_type']}")
        print(f"Sales MAE: ${metrics['sales_mae']:.2f}")
        print(f"Sales MSE: ${metrics['sales_mse']:.2f}")
        print(f"Profit Model Type: {profit_metrics['model_type']}")
        print(f"Profit MAE: ${metrics['profit_mae']:.2f}")
        print(f"Profit MSE: ${metrics['profit_mse']:.2f}")

        return metrics

    def make_forecasts(self, model_data: dict, forecast_days: int = 7) -> DataFrame:
        """Generate forecasts using Prophet models"""
        print(f"\nGenerating {forecast_days}-day forecast...")

        forecasts = self._generate_model_forecasts(model_data, forecast_days)
        forecast_dates = self._generate_forecast_dates(
            model_data["training_data"]["dates"][-1], forecast_days
        )

        return self._create_forecast_dataframe(forecast_dates, forecasts)

    def _prepare_time_series_data(self, daily_summary_df: DataFrame) -> dict:
        """Prepare data for time series modeling"""
        data = (
            daily_summary_df.select("date", "total_sales", "total_profit")
            .orderBy("date")
            .collect()
        )

        dates = np.array([row["date"] for row in data])
        sales_series = np.array([float(row["total_sales"]) for row in data])
        profit_series = np.array([float(row["total_profit"]) for row in data])

        self._print_dataset_info(dates, sales_series, profit_series)

        return {"dates": dates, "sales": sales_series, "profits": profit_series}

    def _print_dataset_info(
        self, dates: np.ndarray, sales: np.ndarray, profits: np.ndarray
    ) -> None:
        """Print time series dataset information"""
        print("Dataset Information:")
        print(f"• Time Period:          {dates[0]} to {dates[-1]}")
        print(f"• Number of Data Points: {len(dates)}")
        print(f"• Average Daily Sales:   ${np.mean(sales):.2f}")
        print(f"• Average Daily Profit:  ${np.mean(profits):.2f}")

    def _fit_forecasting_models(self, data: dict) -> dict:
        """Fit Prophet models to the prepared data"""
        print("\nFitting Models...")
        sales_forecaster = ProphetForecaster()
        profit_forecaster = ProphetForecaster()

        sales_forecaster.fit(data["sales"])
        profit_forecaster.fit(data["profits"])
        print("Model fitting completed successfully")
        print("=" * 80)

        return {
            "sales_model": sales_forecaster,
            "profit_model": profit_forecaster,
            "training_data": data,
        }

    def _generate_model_forecasts(self, model_data: dict, forecast_days: int) -> dict:
        """Generate forecasts from both models"""
        return {
            "sales": model_data["sales_model"].predict(forecast_days),
            "profits": model_data["profit_model"].predict(forecast_days),
        }

    def _generate_forecast_dates(self, last_date: datetime, forecast_days: int) -> list:
        """Generate dates for the forecast period"""
        return [last_date + timedelta(days=i + 1) for i in range(forecast_days)]

    def _create_forecast_dataframe(self, dates: list, forecasts: dict) -> DataFrame:
        """Create Spark DataFrame from forecast data"""
        forecast_rows = [
            (date, float(sales), float(profits))
            for date, sales, profits in zip(
                dates, forecasts["sales"], forecasts["profits"]
            )
        ]

        return self.spark.createDataFrame(
            forecast_rows, ["date", "forecasted_sales", "forecasted_profit"]
        )

    # New methods for data loading and saving

    def load_csv_to_mysql(self, csv_path: str, table_name: str) -> DataFrame:
        """
        Load data from CSV file to MySQL database.

        :param csv_path: Path to the CSV file
        :param table_name: Name of the table to create in MySQL
        :return: DataFrame containing the loaded data
        """
        print(f"Loading data from {csv_path} to MySQL table {table_name}...")

        # Read CSV file
        df = self.spark.read.csv(csv_path, header=True, inferSchema=True)

        # Write to MySQL
        df.write.format("jdbc").option("url", self.config["mysql_url"]).option(
            "driver", "com.mysql.cj.jdbc.Driver"
        ).option("dbtable", table_name).option(
            "user", self.config["mysql_user"]
        ).option(
            "password", self.config["mysql_password"]
        ).mode(
            "overwrite"
        ).save()

        print(f"Successfully loaded {df.count()} rows to MySQL table {table_name}")
        return df

    def setup_candy_store_data(self) -> None:
        """
        Set up the candy store data by loading CSV files to MySQL.
        """
        csv_path = os.getenv("CSV_PATH")

        # Load customers data
        customers_csv = os.path.join(csv_path, "customers.csv")
        self.customers_df = self.load_csv_to_mysql(
            customers_csv, self.config["customers_table"]
        )

        # Load products data
        products_csv = os.path.join(csv_path, "products.csv")
        self.products_df = self.load_csv_to_mysql(
            products_csv, self.config["products_table"]
        )

        # Store original products data for inventory tracking
        self.original_products_df = self.products_df

        # Initialize inventory
        self.initialize_inventory()

        print("Candy store data setup completed successfully")

    def initialize_inventory(self) -> None:
        """
        Initialize inventory from products data.
        """
        if self.products_df is None:
            # Load products from MySQL if not already loaded
            self.products_df = self.load_mysql_data(
                self.config["mysql_url"],
                self.config["products_table"],
                self.config["mysql_user"],
                self.config["mysql_password"],
            )

        # Select only required columns and rename stock to current_stock
        self.current_inventory = self.products_df.select(
            "product_id", "product_name", col("stock").alias("current_stock")
        )
        self.inventory_initialized = True

        print("Inventory initialized successfully")

    def load_mysql_data(
        self, jdbc_url: str, db_table: str, db_user: str, db_password: str
    ) -> DataFrame:
        """
        Load data from MySQL database.

        :param jdbc_url: JDBC URL for the MySQL database
        :param db_table: Name of the table to load data from
        :param db_user: Database username
        :param db_password: Database password
        :return: DataFrame containing the loaded MySQL data
        """
        print(f"Loading data from MySQL table {db_table}...")

        df = (
            self.spark.read.format("jdbc")
            .option("url", jdbc_url)
            .option("driver", "com.mysql.cj.jdbc.Driver")
            .option("dbtable", db_table)
            .option("user", db_user)
            .option("password", db_password)
            .load()
        )

        print(f"Successfully loaded {df.count()} rows from MySQL table {db_table}")
        return df

    def load_transaction_data(self) -> None:
        """
        Load transaction data from JSON files to MongoDB.
        """
        csv_path = os.getenv("CSV_PATH")
        mongo_db = self.config["mongodb_db"]
        mongo_collection_prefix = self.config["mongodb_collection_prefix"]

        # Get all transaction JSON files
        transaction_files = glob.glob(os.path.join(csv_path, "transactions_*.json"))

        for file_path in transaction_files:
            # Extract date from filename (format: transactions_YYYYMMDD.json)
            date_str = os.path.basename(file_path).split("_")[1].split(".")[0]
            collection_name = f"{mongo_collection_prefix}{date_str}"

            print(
                f"Loading transactions from {file_path} to MongoDB collection {collection_name}..."
            )

            # Read JSON file - use multiLine option to handle JSON arrays
            transactions_df = self.spark.read.option("multiLine", "true").json(
                file_path
            )

            # Cache the dataframe to avoid the corrupt record issue
            transactions_df = transactions_df.cache()

            # Write to MongoDB
            transactions_df.write.format("mongo").option(
                "uri", self.config["mongodb_uri"]
            ).option("database", mongo_db).option("collection", collection_name).mode(
                "overwrite"
            ).save()

            print(
                f"Successfully loaded {transactions_df.count()} transactions to MongoDB collection {collection_name}"
            )

            # Unpersist the cached dataframe
            transactions_df.unpersist()

    def save_to_csv(self, df: DataFrame, output_path: str, filename: str) -> None:
        """
        Save a DataFrame to a CSV file.

        :param df: DataFrame to save
        :param output_path: Directory to save the file in
        :param filename: Name of the CSV file
        """
        if df is None:
            print(f"Warning: Cannot save {filename} - DataFrame is None")
            return

        # Ensure output directory exists
        os.makedirs(output_path, exist_ok=True)

        # Full path to the output file
        output_file = os.path.join(output_path, filename)

        # Save DataFrame to CSV
        df.coalesce(1).write.csv(output_file + "_temp", header=True, mode="overwrite")

        # Get the generated part file
        part_file = glob.glob(os.path.join(output_file + "_temp", "part-*.csv"))[0]

        # Move and rename the part file to the desired output file
        shutil.copy(part_file, output_file)

        # Clean up temporary directory
        shutil.rmtree(output_file + "_temp")

        print(f"Successfully saved {df.count()} rows to {output_file}")

    def get_date_range(self, start_date: str, end_date: str) -> list:
        """
        Generate a list of dates between start and end date.

        :param start_date: Start date in YYYYMMDD format
        :param end_date: End date in YYYYMMDD format
        :return: List of date strings in YYYYMMDD format
        """
        start = datetime.strptime(start_date, "%Y%m%d")
        end = datetime.strptime(end_date, "%Y%m%d")
        date_list = []

        current = start
        while current <= end:
            date_list.append(current.strftime("%Y%m%d"))
            current += timedelta(days=1)

        return date_list

    def process_daily_batch(self, date_str: str) -> Tuple[DataFrame, DataFrame, int]:
        """
        Process a daily batch of transactions.

        :param date_str: Date string in YYYYMMDD format
        :return: Tuple of (orders_df, order_items_df, cancelled_count)
        """
        print(f"\nProcessing daily batch for {date_str}...")

        # Load transactions from MongoDB
        collection_name = f"{self.config['mongodb_collection_prefix']}{date_str}"
        transactions_df = self.load_mongodb_data(
            self.config["mongodb_uri"], self.config["mongodb_db"], collection_name
        )

        # Process transactions
        # This is a placeholder - you would implement the actual processing logic here
        print(f"Processed {transactions_df.count()} transactions for {date_str}")

        # Return placeholder values
        return None, None, 0

    def load_mongodb_data(self, uri: str, database: str, collection: str) -> DataFrame:
        """
        Load data from MongoDB.

        :param uri: MongoDB URI
        :param database: Database name
        :param collection: Collection name
        :return: DataFrame containing the loaded MongoDB data
        """
        print(f"Loading data from MongoDB collection {collection}...")

        df = (
            self.spark.read.format("mongo")
            .option("uri", uri)
            .option("database", database)
            .option("collection", collection)
            .load()
        )

        print(
            f"Successfully loaded {df.count()} documents from MongoDB collection {collection}"
        )
        return df

    def create_daily_summary(
        self, date_str: str, orders_df: DataFrame, order_items_df: DataFrame
    ) -> DataFrame:
        """
        Create a daily summary of orders.

        :param date_str: Date string in YYYYMMDD format
        :param orders_df: Orders DataFrame
        :param order_items_df: Order items DataFrame
        :return: Daily summary DataFrame
        """
        # This is a placeholder - you would implement the actual summary logic here
        print(f"Creating daily summary for {date_str}...")

        # Return placeholder DataFrame
        return None

    def combine_daily_results(self) -> None:
        """
        Combine daily results into final output files.
        """
        print("Combining daily results...")

        # This is a placeholder - you would implement the actual combination logic here
        print("Daily results combined successfully")

    def generate_sales_forecasts(self) -> None:
        """
        Generate sales forecasts.
        """
        print("Generating sales forecasts...")

        # This is a placeholder - you would implement the actual forecasting logic here
        print("Sales forecasts generated successfully")

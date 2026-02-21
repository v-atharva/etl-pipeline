from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    explode,
    col,
    round as spark_round,
    sum as spark_sum,
    count,
    abs as spark_abs,
    lit,
    to_date,
    countDistinct,
)
from pyspark.sql.types import (
    StructType,
    StructField,
    IntegerType,
    StringType,
    DecimalType,
    DateType,
    DoubleType,
    LongType,
)
from typing import Dict, Tuple, List
import os
import glob
import shutil
import decimal
import numpy as np
from time_series import ProphetForecaster
from datetime import datetime, timedelta


class DataProcessor:
    def __init__(self, spark: SparkSession):
        self.spark = spark
        # Initialize all class properties
        self.config = None
        self.current_inventory = {}  # dict: product_id -> current_stock
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
        self.products_data = {}  # dict: product_id -> {sales_price, cost_to_make, ...}

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

    # ------------------------------------------------------------------------------------------------
    # Data Loading Methods
    # ------------------------------------------------------------------------------------------------

    def load_mysql_data(self) -> None:
        """Load customers and products tables from MySQL into Spark DataFrames"""
        config = self.config
        mysql_properties = {
            "user": config["mysql_user"],
            "password": config["mysql_password"],
            "driver": "com.mysql.cj.jdbc.Driver",
        }

        # Load customers table
        print("\nLoading customers data from MySQL...")
        self.customers_df = (
            self.spark.read.jdbc(
                url=config["mysql_url"],
                table=config["customers_table"],
                properties=mysql_properties,
            )
        )
        self._print_data_preview(self.customers_df, "Customers")

        # Load products table
        print("\nLoading products data from MySQL...")
        self.products_df = (
            self.spark.read.jdbc(
                url=config["mysql_url"],
                table=config["products_table"],
                properties=mysql_properties,
            )
        )
        self.original_products_df = self.products_df
        self._print_data_preview(self.products_df, "Products")

        # Cache products data as dict for fast lookups during processing
        self._cache_products_data()

    def load_mongodb_transactions(self, date_str: str) -> DataFrame:
        """Load a single day's transactions from MongoDB"""
        collection_name = (
            f"{self.config['mongodb_collection_prefix']}{date_str}"
        )
        print(f"\nLoading transactions from MongoDB: {collection_name}")

        transactions_df = (
            self.spark.read.format("mongo")
            .option("uri", self.config["mongodb_uri"])
            .option("database", self.config["mongodb_db"])
            .option("collection", collection_name)
            .load()
        )

        self._print_data_preview(transactions_df, f"Transactions ({date_str})")
        return transactions_df

    def _print_data_preview(self, df: DataFrame, name: str) -> None:
        """Print a data preview with dimensions"""
        row_count = df.count()
        col_count = len(df.columns)
        print(f"\n{name} Data Preview:")
        print(f"  Dimensions: {row_count} rows x {col_count} columns")
        df.show(5, truncate=False)

    def _cache_products_data(self) -> None:
        """Cache product data as dict for fast lookups"""
        products_rows = self.products_df.collect()
        for row in products_rows:
            self.products_data[row["product_id"]] = {
                "product_name": row["product_name"],
                "sales_price": float(row["sales_price"]),
                "cost_to_make": float(row["cost_to_make"]),
                "stock": int(row["stock"]),
            }

    # ------------------------------------------------------------------------------------------------
    # Inventory Management
    # ------------------------------------------------------------------------------------------------

    def initialize_inventory(self) -> None:
        """Initialize inventory tracker from products data"""
        print("\nInitializing inventory...")
        for pid, pdata in self.products_data.items():
            self.current_inventory[pid] = pdata["stock"]
        self.inventory_initialized = True
        print(f"  Inventory initialized for {len(self.current_inventory)} products")

    def check_and_update_inventory(
        self, product_id: int, qty: int, order_id: int
    ) -> bool:
        """
        Check if there is sufficient stock for the order item.
        If yes, deduct and return True.
        If no, print cancellation and return False.
        """
        current_stock = self.current_inventory.get(product_id, 0)
        if current_stock >= qty:
            self.current_inventory[product_id] = current_stock - qty
            return True
        else:
            product_name = self.products_data.get(product_id, {}).get(
                "product_name", "Unknown"
            )
            print(
                f"  ⚠ CANCELLED: Order {order_id}, Product '{product_name}' "
                f"(ID: {product_id}) - Requested: {qty}, Available: {current_stock}"
            )
            self.total_cancelled_items += 1
            return False

    def print_inventory_levels(self) -> None:
        """Print current inventory levels for all products"""
        print("\nCURRENT INVENTORY LEVELS")
        print("-" * 40)

        for pid in sorted(self.current_inventory.keys()):
            pname = self.products_data.get(pid, {}).get("product_name", "Unknown")
            stock = self.current_inventory[pid]
            print(f"• {pname:<30} (ID: {pid:>3}): {stock:>4} units")
        print("-" * 40)

    # ------------------------------------------------------------------------------------------------
    # Batch Processing ETL
    # ------------------------------------------------------------------------------------------------

    def process_daily_batch(
        self, transactions_df: DataFrame, date_str: str
    ) -> Tuple[List, List, int]:
        """
        Process one day's transactions. Returns:
          - orders_rows: list of order dicts
          - order_items_rows: list of order line item dicts
          - cancelled_count: number of items cancelled this day
        """
        formatted_date = (
            f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        )
        print(f"\n{'='*80}")
        print(f"PROCESSING BATCH: {formatted_date}")
        print(f"{'='*80}")

        # Collect transactions as Python objects for processing
        transactions = transactions_df.collect()
        print(f"  Transactions to process: {len(transactions)}")

        orders_rows = []
        order_items_rows = []
        cancelled_count = 0

        for txn in transactions:
            order_id = txn["transaction_id"]
            customer_id = txn["customer_id"]
            items = txn["items"]

            order_total = decimal.Decimal("0.00")
            num_items = 0
            has_non_null_item = False

            for item in items:
                product_id = item["product_id"]
                product_name = item["product_name"]
                qty = item["qty"]

                # Skip items with null quantity
                if qty is None:
                    continue

                has_non_null_item = True
                qty = int(qty)
                sales_price = self.products_data.get(product_id, {}).get(
                    "sales_price", 0.0
                )
                unit_price = decimal.Decimal(str(sales_price))

                # Check inventory
                if self.check_and_update_inventory(product_id, qty, order_id):
                    line_total = unit_price * qty
                    order_items_rows.append({
                        "order_id": order_id,
                        "product_id": product_id,
                        "product_name": product_name,
                        "quantity": qty,
                        "unit_price": unit_price,
                        "line_total": line_total,
                    })
                    order_total += line_total
                    num_items += 1
                else:
                    # Cancelled item - keep in table with qty=0, line_total=0
                    cancelled_count += 1
                    order_items_rows.append({
                        "order_id": order_id,
                        "product_id": product_id,
                        "product_name": product_name,
                        "quantity": 0,
                        "unit_price": unit_price,
                        "line_total": decimal.Decimal("0.00"),
                    })

            orders_rows.append({
                "order_id": order_id,
                "customer_id": customer_id,
                "order_date": formatted_date,
                "order_total": order_total,
                "num_items": num_items,
                "has_non_null_item": has_non_null_item,
            })

        print(f"  Orders processed: {len(orders_rows)}")
        print(f"  Items cancelled: {cancelled_count}")
        return orders_rows, order_items_rows, cancelled_count

    def build_orders_dataframe(self, all_orders_rows: List) -> DataFrame:
        """Build the orders DataFrame sorted by order_id"""
        schema = StructType([
            StructField("order_id", LongType(), False),
            StructField("customer_id", IntegerType(), False),
            StructField("order_date", StringType(), False),
            StructField("order_total", DecimalType(10, 2), False),
            StructField("num_items", IntegerType(), False),
        ])

        rows = [
            (
                int(o["order_id"]),
                int(o["customer_id"]),
                o["order_date"],
                o["order_total"],
                int(o.get("num_items", 0)),
            )
            for o in all_orders_rows
        ]

        orders_df = self.spark.createDataFrame(rows, schema)
        orders_df = orders_df.orderBy("order_id")
        return orders_df

    def build_order_line_items_dataframe(
        self, all_items_rows: List
    ) -> DataFrame:
        """Build the order_line_items DataFrame sorted by order_id, product_id"""
        schema = StructType([
            StructField("order_id", LongType(), False),
            StructField("product_id", IntegerType(), False),
            StructField("product_name", StringType(), False),
            StructField("quantity", IntegerType(), False),
            StructField("unit_price", DecimalType(10, 2), False),
            StructField("line_total", DecimalType(10, 2), False),
        ])

        rows = [
            (
                int(i["order_id"]),
                int(i["product_id"]),
                i["product_name"],
                int(i["quantity"]),
                i["unit_price"],
                i["line_total"],
            )
            for i in all_items_rows
        ]

        items_df = self.spark.createDataFrame(rows, schema)
        items_df = items_df.orderBy("order_id", "product_id")
        return items_df

    # ------------------------------------------------------------------------------------------------
    # Data Analytics
    # ------------------------------------------------------------------------------------------------

    def create_daily_summary(
        self, all_orders_rows: List, all_items_rows: List
    ) -> DataFrame:
        """
        Create daily_summary table with:
          date, num_orders, total_sales (decimal(10,2)), total_profit (decimal(10,2))
        """
        print("\nCreating daily summary...")

        # Group orders by date, only counting orders that had at least one
        # item pass the inventory check (tracked via has_fulfilled_item flag)
        daily_data = {}
        for order in all_orders_rows:
            date = order["order_date"]
            if date not in daily_data:
                daily_data[date] = {
                    "num_orders": 0,
                    "total_sales": decimal.Decimal("0.00"),
                    "total_cost": decimal.Decimal("0.00"),
                }
            # Only count orders that have at least one item with non-null qty.
            # Orders where ALL items have null qty are not valid orders.
            if order["has_non_null_item"]:
                daily_data[date]["num_orders"] += 1

        # Sum sales and cost from line items
        # Build a lookup from order_id to date
        order_date_map = {}
        for order in all_orders_rows:
            order_date_map[order["order_id"]] = order["order_date"]

        for item in all_items_rows:
            if item["quantity"] > 0:
                date = order_date_map.get(item["order_id"])
                if date and date in daily_data:
                    daily_data[date]["total_sales"] += item["line_total"]
                    cost_per_unit = decimal.Decimal(
                        str(
                            self.products_data.get(
                                item["product_id"], {}
                            ).get("cost_to_make", 0.0)
                        )
                    )
                    daily_data[date]["total_cost"] += (
                        cost_per_unit * item["quantity"]
                    )

        # Build DataFrame
        schema = StructType([
            StructField("date", DateType(), False),
            StructField("num_orders", IntegerType(), False),
            StructField("total_sales", DecimalType(10, 2), False),
            StructField("total_profit", DecimalType(10, 2), False),
        ])

        rows = []
        for date_str in sorted(daily_data.keys()):
            d = daily_data[date_str]
            profit = d["total_sales"] - d["total_cost"]
            date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
            rows.append((
                date_obj,
                d["num_orders"],
                d["total_sales"],
                profit,
            ))

        daily_summary_df = self.spark.createDataFrame(rows, schema)
        self.daily_summary_df = daily_summary_df

        print("\nDAILY SUMMARY:")
        daily_summary_df.show(truncate=False)
        return daily_summary_df

    def create_products_updated(self) -> DataFrame:
        """Create products_updated table with final inventory levels, sorted by product_id"""
        print("\nCreating products_updated table...")

        products_rows = self.original_products_df.collect()
        updated_rows = []
        for row in products_rows:
            pid = row["product_id"]
            updated_stock = self.current_inventory.get(pid, 0)
            updated_rows.append((
                row["product_id"],
                row["product_name"],
                row["product_category"],
                row["product_subcategory"],
                row["product_shape"],
                row["sales_price"],
                row["cost_to_make"],
                updated_stock,
            ))

        schema = StructType([
            StructField("product_id", IntegerType(), False),
            StructField("product_name", StringType(), False),
            StructField("product_category", StringType(), False),
            StructField("product_subcategory", StringType(), False),
            StructField("product_shape", StringType(), False),
            StructField("sales_price", DecimalType(10, 2), False),
            StructField("cost_to_make", DecimalType(10, 2), False),
            StructField("stock", IntegerType(), False),
        ])

        products_updated_df = self.spark.createDataFrame(updated_rows, schema)
        products_updated_df = products_updated_df.orderBy("product_id")

        print("\nUPDATED INVENTORY LEVELS:")
        products_updated_df.show(truncate=False)
        return products_updated_df

    # ------------------------------------------------------------------------------------------------
    # CSV Output
    # ------------------------------------------------------------------------------------------------

    def save_to_csv(
        self, df: DataFrame, output_path: str, filename: str
    ) -> None:
        """Save a Spark DataFrame to a single CSV file"""
        # Ensure output directory exists
        os.makedirs(output_path, exist_ok=True)

        temp_path = os.path.join(output_path, f"_temp_{filename}")
        final_path = os.path.join(output_path, filename)

        # Remove temp directory if it exists
        if os.path.exists(temp_path):
            shutil.rmtree(temp_path)

        # Write as CSV with header
        df.coalesce(1).write.option("header", "true").mode("overwrite").csv(
            temp_path
        )

        # Find the part file and move it to the final location
        part_files = glob.glob(os.path.join(temp_path, "part-*.csv"))
        if part_files:
            if os.path.exists(final_path):
                os.remove(final_path)
            shutil.move(part_files[0], final_path)

        # Clean up temp directory
        if os.path.exists(temp_path):
            shutil.rmtree(temp_path)

        print(f"  ✓ Saved: {final_path}")

    def finalize_processing(self) -> None:
        """Finalize processing and create summary"""
        print("\nPROCESSING COMPLETE")
        print("=" * 80)
        print(f"Total Cancelled Items: {self.total_cancelled_items}")

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

    def make_forecasts(
        self, model_data: dict, forecast_days: int = 7
    ) -> DataFrame:
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

    def _generate_model_forecasts(
        self, model_data: dict, forecast_days: int
    ) -> dict:
        """Generate forecasts from both models"""
        return {
            "sales": model_data["sales_model"].predict(forecast_days),
            "profits": model_data["profit_model"].predict(forecast_days),
        }

    def _generate_forecast_dates(
        self, last_date: datetime, forecast_days: int
    ) -> list:
        """Generate dates for the forecast period"""
        return [
            last_date + timedelta(days=i + 1) for i in range(forecast_days)
        ]

    def _create_forecast_dataframe(
        self, dates: list, forecasts: dict
    ) -> DataFrame:
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

# Batch Processing ETL Pipeline

This project implements a **batch processing ETL pipeline** for Tiger's Candy, a candy store that originated on the RIT campus. The system processes 10 days of raw order transactions (February 1–10, 2024) to:

1. **Load** customer and product data from MySQL, and transaction data from MongoDB
2. **Process** daily order batches with inventory validation
3. **Generate** output tables: orders, order line items, daily summary, and updated inventory
4. **Forecast** future sales and profits using a Prophet time series model

### Dataset

- `customers.csv` — 30 customers with contact information
- `products.csv` — Candy products with pricing, production costs, and stock levels
- `transactions_*.json` — 10 daily JSON files containing order transactions with items and quantities
 
---

## Required Packages

| Package | Purpose |
|---------|---------|
| `pyspark` | Data processing and Spark session |
| `python-dotenv` | Environment variable management |
| `numpy` | Numerical computations |
| `prophet` | Time series forecasting |
| `scikit-learn` | Forecast accuracy metrics |
| `pandas` | Required by Prophet internally |
| `mysql-connector-java` | MySQL JDBC connector (JAR file) |

---

## Setup Instructions

### 1. Database Setup

**MySQL:**
- Create a database named `candy_store`
- Load `customers.csv` into the `customers` table
- Load `products.csv` into the `products` table

**MongoDB:**
- Create a database named `candy_store`
- Load each `transactions_*.json` file into a separate collection named `transactions_YYYYMMDD`

### 2. Environment Configuration
Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env 
```

Update the following fields:
- `MYSQL_CONNECTOR_PATH` — Path to your MySQL JDBC JAR
- `MYSQL_USER` / `MYSQL_PASSWORD` — MySQL credentials
- `MONGODB_URI` — MongoDB connection string (default: `mongodb://localhost:27017`)

### 3. Running the Pipeline

#### Orchestration with Apache Airflow

1. Install Apache Airflow:
```bash
pip install apache-airflow
```

2. Initialize the Airflow database:
```bash
airflow db init
```

3. Copy the DAG file to the Airflow DAGs folder:
```bash
cp src/candy_store_dag.py ~/airflow/dags/
```

4. Start the Airflow webserver and scheduler:
```bash
airflow webserver --port 8080 &
airflow scheduler &
```

5. Open the Airflow UI at `http://localhost:8080` and trigger the `candy_store_etl_pipeline` DAG.



---

<p align="center">
  <b>— OR —</b>
</p>

---

#### Direct Script Execution
```bash
python main.py
```


### Expected Output

The pipeline will:
- Print data previews with dimensions for customers, products, and transactions
- Process 10 daily batches, printing cancellation messages for items with insufficient inventory
- Print a daily processing summary after each batch
- Save output CSV files to `data/output/`:
   - `orders.csv` — All orders sorted by `order_id`
   - `order_line_items.csv` — All line items sorted by `order_id`, `product_id`
   - `daily_summary.csv` — Daily statistics (date, num_orders, total_sales, total_profit)
   - `products_updated.csv` — Final inventory levels sorted by `product_id`
   - `sales_profit_forecast.csv` — 1-day sales and profit forecast
- Print forecast MAE and MSE metrics

---

## Processing Logic

### Overview

The pipeline follows a sequential batch-processing approach: transactions are loaded one day at a time, validated against current inventory, and accumulated into output tables. Inventory state is carried across days — stock levels depleted on Day 1 affect availability on Day 2.

### Daily Batch Processing (`process_daily_batch`)

For each day's transactions, every transaction (order) is processed item by item:

1. **Null Quantity Filtering** — Items with `qty = None` are skipped entirely. They are not counted as cancelled; they simply do not participate in the order.

2. **Inventory Validation** — For each item with a non-null quantity, the system checks for availability of the product (`current_stock >= requested_qty`)
   - **If true**: Inventory is deducted by the requested quantity. The line item is recorded with its original quantity and calculated `line_total = unit_price × quantity`.
   - **If false**: The line item is still recorded but with `quantity = 0` and `line_total = 0`. A cancellation message is printed.

3. **Order Total** — Each order's total is the sum of `line_total` values from its fulfilled items only.

4. **Validity Tracking** — Each order is flagged with `has_non_null_item` (`True` if at least one item had a non-null quantity). This flag is used later in the daily summary to determine valid orders.

## Time Series Forecasting

### Model: Facebook Prophet

The pipeline uses [Facebook Prophet](https://facebook.github.io/prophet/) to forecast future sales and profits based on the 10-day historical data from the daily summary.

### Forecasting Process

Two independent Prophet models are trained — one for `total_sales` and one for `total_profit`. Each model uses default hyperparameters, including additive seasonality and automatic changepoint detection, and is fitted on 10 daily data points.

Once trained, the models generate in-sample predictions for the training period, which are then compared against the actual values to measure accuracy. 
   - **MAE (Mean Absolute Error)** — Average absolute difference between predicted and actual values
   - **MSE (Mean Squared Error)** — Average squared difference, penalizing large errors more heavily

Finally, each model produces a **1-day ahead forecast**, generating predicted values for both `forecasted_sales` and `forecasted_profit`.

> **Note:** Prophet typically performs best with longer historical periods (months/years) and may not capture weekly seasonality reliably from such a short window. With only 10 data points, the forecast should be considered a rough estimate. 

---

```
root/
├── .env.example              # Environment variable template
├── data/
│   └── dataset/
│       ├── customers.csv
│       ├── products.csv
│       └── transactions_*.json (10 files)
├── src/
│   ├── main.py               # Main pipeline orchestration
│   ├── data_processor.py     # Core ETL processing logic
│   ├── time_series.py        # Prophet forecasting model
│   └── candy_store_dag.py    # Apache Airflow DAG
└── README.md
```

---

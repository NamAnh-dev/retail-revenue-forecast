import pandas as pd
from pathlib import Path


def load_raw_tables(data_dir = "data/") -> dict[str, pd.DataFrame]:
    
    data_dir = Path(data_dir)

    tables = {
        "orders": pd.read_csv(
            data_dir / "olist_orders_dataset.csv",
            parse_dates=[
                "order_purchase_timestamp",
                "order_approved_at",
                "order_delivered_customer_date",
                "order_estimated_delivery_date",
            ],
        ),
        "order_items": pd.read_csv(data_dir / "olist_order_items_dataset.csv"),
        "products": pd.read_csv(data_dir / "olist_products_dataset.csv"),
        "payments": pd.read_csv(data_dir / "olist_order_payments_dataset.csv"),
        "customers": pd.read_csv(data_dir / "olist_customers_dataset.csv"),
        "category_translation": pd.read_csv(
            data_dir / "product_category_name_translation.csv"
        ),
    }


    return tables


def build_master_df(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:

    orders      = tables["orders"]
    order_items = tables["order_items"]
    products    = tables["products"]
    payments    = tables["payments"]
    customers   = tables["customers"]
    cat_trans   = tables["category_translation"]

    products = products.merge(cat_trans, on="product_category_name", how="left")
    products["category_en"] = products["product_category_name_english"].fillna(
        products["product_category_name"]
    )

    payment_agg = (
        payments.groupby("order_id")["payment_value"].sum().reset_index()
    )

    df = (
        orders
        .merge(order_items[["order_id", "product_id", "price", "freight_value"]], on="order_id")
        .merge(products[["product_id", "product_category_name", "category_en"]], on="product_id")
        .merge(payment_agg, on="order_id")
        .merge(customers[["customer_id", "customer_state", "customer_city"]], on="customer_id")
    )

    df = df[df["order_status"] == "delivered"].copy()

    df["date"]    = df["order_purchase_timestamp"].dt.normalize()
    df["year"]    = df["order_purchase_timestamp"].dt.year
    df["month"]   = df["order_purchase_timestamp"].dt.month
    df["quarter"] = df["order_purchase_timestamp"].dt.quarter
    df["hour"]    = df["order_purchase_timestamp"].dt.hour
    df["dayofweek"] = df["order_purchase_timestamp"].dt.dayofweek 

    df["delivery_days"] = (
        df["order_delivered_customer_date"] - df["order_purchase_timestamp"]
    ).dt.days

    print(f"\n=== Master DataFrame ===")
    print(f"  Shape      : {df.shape}")
    print(f"  Date range : {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"  Orders     : {df['order_id'].nunique():,}")
    print(f"  Customers  : {df['customer_id'].nunique():,}")
    print(f"  Categories : {df['category_en'].nunique()}")
    print(f"  Missing %  :\n{(df.isnull().mean() * 100).round(2).to_string()}")

    return df


def build_daily_series(df: pd.DataFrame) -> pd.DataFrame:

    daily = (
        df.groupby("date")["payment_value"]
        .sum()
        .reset_index()
        .rename(columns={"payment_value": "revenue"})
    )

    full_range = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    daily = (
        daily.set_index("date")
        .reindex(full_range, fill_value=0)
        .reset_index()
        .rename(columns={"index": "date"})
    )

    print(f"\n=== Daily Series ===")
    print(f"  Days  : {len(daily)}")
    print(f"  Mean  : {daily['revenue'].mean():,.0f} BRL/day")
    print(f"  Max   : {daily['revenue'].max():,.0f} BRL")
    print(f"  Zeros : {(daily['revenue'] == 0).sum()} days")

    return daily


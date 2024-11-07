# %%
import os
import logging
import threading
import polars as pl
import numpy as np
from datetime import time
pl.Config.set_tbl_cols(100)

# %%
# Logging configuration
outfile = "./log"
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(outfile),
        logging.StreamHandler()
    ]
)

logging.info(f"Starting {os.path.abspath(__file__)}")

# %%
# Data configuration
root = os.path.abspath(os.path.abspath(''))
raw_data_folder = os.path.join(root, "raw_data")
final_data_folder = os.path.join(root, "dataset", "lob")
book_filename = os.path.join(raw_data_folder, "book.csv.gz")


bid_columns_to_check_all_nans = [
    'L2-BidPrice',  'L2-BidSize',  'L2-BuyNo',
    'L3-BidPrice',  'L3-BidSize',  'L3-BuyNo',
    'L4-BidPrice',  'L4-BidSize',  'L4-BuyNo',
    'L5-BidPrice',  'L5-BidSize',  'L5-BuyNo',
    'L6-BidPrice',  'L6-BidSize',  'L6-BuyNo',
    'L7-BidPrice',  'L7-BidSize',  'L7-BuyNo',
    'L8-BidPrice',  'L8-BidSize',  'L8-BuyNo',
    'L9-BidPrice',  'L9-BidSize',  'L9-BuyNo',
    'L10-BidPrice', 'L10-BidSize', 'L10-BuyNo',
]
ask_columns_to_check_all_nans = [
    'L2-AskPrice',  'L2-AskSize',  'L2-SellNo',
    'L3-AskPrice',  'L3-AskSize',  'L3-SellNo',
    'L4-AskPrice',  'L4-AskSize',  'L4-SellNo',
    'L5-AskPrice',  'L5-AskSize',  'L5-SellNo',
    'L6-AskPrice',  'L6-AskSize',  'L6-SellNo',
    'L7-AskPrice',  'L7-AskSize',  'L7-SellNo',
    'L8-AskPrice',  'L8-AskSize',  'L8-SellNo',
    'L9-AskPrice',  'L9-AskSize',  'L9-SellNo',
    'L10-AskPrice', 'L10-AskSize', 'L10-SellNo'
]
fill_null_with_zero = [
    'L2-BidSize',  'L2-BuyNo',  'L2-AskSize',  'L2-SellNo',
    'L3-BidSize',  'L3-BuyNo',  'L3-AskSize',  'L3-SellNo',
    'L4-BidSize',  'L4-BuyNo',  'L4-AskSize',  'L4-SellNo',
    'L5-BidSize',  'L5-BuyNo',  'L5-AskSize',  'L5-SellNo',
    'L6-BidSize',  'L6-BuyNo',  'L6-AskSize',  'L6-SellNo',
    'L7-BidSize',  'L7-BuyNo',  'L7-AskSize',  'L7-SellNo',
    'L8-BidSize',  'L8-BuyNo',  'L8-AskSize',  'L8-SellNo',
    'L9-BidSize',  'L9-BuyNo',  'L9-AskSize',  'L9-SellNo',
    'L10-BidSize', 'L10-BuyNo', 'L10-AskSize', 'L10-SellNo'
]

bid_price_columns = [f"L{level}-BidPrice" for level in range(1, 11)]
ask_price_columns = [f"L{level}-AskPrice" for level in range(1, 11)]

ordered_columns = [
    "#RIC", "Date-Time", "Date",
    # Bid price
    "L10-BidPrice", "L9-BidPrice", "L8-BidPrice", "L7-BidPrice", "L6-BidPrice", "L5-BidPrice", "L4-BidPrice", "L3-BidPrice", "L2-BidPrice", "L1-BidPrice",
    # Ask price
    "L1-AskPrice", "L2-AskPrice", "L3-AskPrice", "L4-AskPrice", "L5-AskPrice", "L6-AskPrice", "L7-AskPrice", "L8-AskPrice", "L9-AskPrice", "L10-AskPrice",
    # Bid size
    "L10-BidSize", "L9-BidSize", "L8-BidSize", "L7-BidSize", "L6-BidSize", "L5-BidSize", "L4-BidSize", "L3-BidSize", "L2-BidSize", "L1-BidSize",
    # Ask size
    "L1-AskSize", "L2-AskSize", "L3-AskSize", "L4-AskSize", "L5-AskSize", "L6-AskSize", "L7-AskSize", "L8-AskSize", "L9-AskSize", "L10-AskSize",
    # Bid No
    "L10-BuyNo", "L9-BuyNo", "L8-BuyNo", "L7-BuyNo", "L6-BuyNo", "L5-BuyNo", "L4-BuyNo", "L3-BuyNo", "L2-BuyNo", "L1-BuyNo",
    # Ask No
    "L1-SellNo", "L2-SellNo", "L3-SellNo", "L4-SellNo", "L5-SellNo", "L6-SellNo", "L7-SellNo", "L8-SellNo", "L9-SellNo", "L10-SellNo"
]


# %%
def __write_csv(df: pl.DataFrame, ric, date):
    logging.info(f"Going to save csv for {ric} on {date}")
    filename = os.path.join(final_data_folder, f"{ric}_{date.strftime('%y_%m_%d')}.csv")
    df.filter((pl.col("#RIC") == ric) & (pl.col("Date") == date)).drop(["#RIC", "Date"]).rename({"Date-Time": "date"}).write_csv(filename)

# %%
book_df = pl.scan_csv(book_filename, infer_schema_length=100000, try_parse_dates=True)
logging.info(f"Going to lazy load {book_filename}")
# %%

nanoseconds_in_a_microsecond = 1000
book_df = (
    book_df
    .drop(["Domain", "GMT Offset", "Type", "Exch Time"])
    .filter(
        (pl.col("Date-Time").dt.time() >= time(7, 0)) &
        (pl.col("Date-Time").dt.time() <= time(15, 30))
    )
    .filter(~pl.all_horizontal(pl.col(bid_columns_to_check_all_nans).is_null()))
    .filter(~pl.all_horizontal(pl.col(ask_columns_to_check_all_nans).is_null()))
    .with_columns(
        [pl.col(col).fill_null(0) for col in fill_null_with_zero]
    )  # Fill null size and number of orders with zero 
    .with_columns(
        [pl.col(col).fill_null(strategy='forward') for col in [*bid_price_columns, *ask_price_columns]]
    )  # Fill 
    .with_columns(
        pl.col("Date-Time").dt.date().alias("Date")
    )
    .with_columns(pl.col("Date-Time").dt.round("500ms").alias("Date-Time"))
    .drop_nulls()
    # .with_columns(
    #     pl.len().over("Date-Time").alias('duplicate_datetime')
    # )
    # .with_columns(
    #     pl.col("Date-Time").cum_count().over("Date-Time").alias('counter')
    # )
    # .with_columns(
    #     (pl.col("Date-Time").dt.cast_time_unit('ns') + pl.duration(nanoseconds=(nanoseconds_in_a_microsecond/pl.col("duplicate_datetime"))*(pl.col("counter")-1))).alias("Date-Time")
    # )
    .select(ordered_columns)
).collect()
from datetime import timedelta

# This snippet checks for the minimum time between events. 
# For the 3 symbols is 1 microsecond (AENA)
min_time_diff = 10e8
for ric in book_df.select(["#RIC"]).unique().to_numpy():
    df_with_diffs = book_df.filter(
        pl.col("#RIC") == ric
    ).with_columns([
        (pl.col("Date-Time") - pl.col("Date-Time").shift(1)).alias("time_diff")
    ]).with_columns(
        pl.col("time_diff").cast(pl.Int64).alias("time_diff")
    )
    min_diff_ric = df_with_diffs.select(pl.col("time_diff").min()).item()
    if min_diff_ric < min_time_diff:
        min_time_diff = min_diff_ric
    logging.info(f"The minimal time diff for {ric} is {min_diff_ric}")
logging.info(f"The minimal time diff for all symbols is {min_time_diff}, {type(min_time_diff)}")

# Upsample then downsample to have equally-spaced measures
import pandas as pd
new_book_df = pl.DataFrame()
for (ric, date) in book_df.select(["#RIC", "Date"]).unique().to_numpy():
    logging.info(f"Going to save csv for {ric} on {date}")
    df = book_df.filter(
        (pl.col("#RIC") == ric) & (pl.col("Date") == date)
    )
    
    first_row = df.head(1).with_columns(
        (pl.col("Date-Time").dt.truncate("1s")).alias("Date-Time")
    )
    df = pl.concat([first_row, df])
        
    filename = os.path.join(final_data_folder, f"{ric}_{date.strftime('%y_%m_%d')}.csv")

    df = (df.unique(subset=["Date-Time"], keep="first", maintain_order=True)
            .upsample(time_column="Date-Time", every="500ms", maintain_order=True)
            .with_columns(pl.col(df.columns).fill_null(strategy='forward'))
            # .select(pl.all().last())
            # .group_by_dynamic('Date-Time', every='500us')
            # .agg(pl.all().mean())
            )
    df.drop(["#RIC", "Date"]).rename({"Date-Time": "date"}).write_csv(filename)


# %%
logging.info(f"Finished {os.path.abspath(__file__)}")

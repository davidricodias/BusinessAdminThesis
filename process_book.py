# %%
import os
import logging
import threading
import pickle
import polars as pl

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
final_data_folder = os.path.join(root, "data")
book_filename = os.path.join(raw_data_folder, "book.csv.gz")


columns_to_check_all_nans = [
    'L2-BidPrice',  'L2-BidSize',  'L2-BuyNo',  'L2-AskPrice',  'L2-AskSize',  'L2-SellNo',
    'L3-BidPrice',  'L3-BidSize',  'L3-BuyNo',  'L3-AskPrice',  'L3-AskSize',  'L3-SellNo',
    'L4-BidPrice',  'L4-BidSize',  'L4-BuyNo',  'L4-AskPrice',  'L4-AskSize',  'L4-SellNo',
    'L5-BidPrice',  'L5-BidSize',  'L5-BuyNo',  'L5-AskPrice',  'L5-AskSize',  'L5-SellNo',
    'L6-BidPrice',  'L6-BidSize',  'L6-BuyNo',  'L6-AskPrice',  'L6-AskSize',  'L6-SellNo',
    'L7-BidPrice',  'L7-BidSize',  'L7-BuyNo',  'L7-AskPrice',  'L7-AskSize',  'L7-SellNo',
    'L8-BidPrice',  'L8-BidSize',  'L8-BuyNo',  'L8-AskPrice',  'L8-AskSize',  'L8-SellNo',
    'L9-BidPrice',  'L9-BidSize',  'L9-BuyNo',  'L9-AskPrice',  'L9-AskSize',  'L9-SellNo',
    'L10-BidPrice', 'L10-BidSize', 'L10-BuyNo', 'L10-AskPrice', 'L10-AskSize', 'L10-SellNo'
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

ordered_columns = [
    "#RIC", "Date-Time",
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
def __write_parquet(df: pl.DataFrame, ric, date):
    filename = os.path.join(final_data_folder, f"{ric}_{date.strftime('%y_%m_%d')}.parquet")
    df.filter((pl.col("#RIC") == ric) & (pl.col("Date") == date)).drop(["#RIC", "Date", "Date-Time"]).write_parquet(filename)

# %%
book_df = pl.scan_csv(book_filename, infer_schema_length=100000, try_parse_dates=True, low_memory=True)
logging.info(f"Going to lazy load {book_filename}")
# %%
book_df = (
    book_df
    .drop(["Domain", "GMT Offset", "Type", "Exch Time"])
    .filter(pl.col("#RIC") == "AENA.MC")
    .filter((7 <= pl.col("Date-Time").dt.hour()) & (pl.col("Date-Time").dt.hour() <= 16))
    .filter(~pl.all_horizontal(pl.col(columns_to_check_all_nans).is_null()))
    .fill_null(0)
    .with_columns(pl.col("Date-Time").dt.date().alias("Date"))
    # .with_columns(pl.col("Date-Time").dt.month().alias("Month"))
    # .with_columns(pl.col("Date-Time").dt.weekday().alias("Weekday"))
    # .with_columns(pl.col("Date-Time").dt.day().alias("Day"))
    # .with_columns(pl.col("Date-Time").dt.hour().alias("Hour"))
    # .with_columns(pl.col("Date-Time").dt.minute().alias("Minute"))
    # .with_columns(pl.col("Date-Time").dt.second().alias("Second"))
    # .with_columns(pl.col("Date-Time").dt.microsecond().alias("Microsecond"))
    .select(ordered_columns)
    .slice(500000, 5000)
).collect()
logging.info(f"Filtered {book_filename}")
logging.info(book_df.describe())
logging.info(book_df.schema)
logging.info(book_df.select(pl.col("#RIC").value_counts()))
book_df.drop(["#RIC"]).rename({"Date-Time": "date"}).write_csv("./dataset/lob/lob.csv")


# %%
# Decomopose time index

# %%
# scaler = MinMaxScaler(feature_range=(-0.5, 0.5))
# scaler_filename = os.path.join(root, "model", "scaler.pkl")
# columns_to_scale = book_df.columns[3:]  # Exclude "#RIC", "Date", "Date-Time"
# scaler.fit(book_df.select(columns_to_scale).to_numpy())
# scaled_data = scaler.transform(book_df.select(columns_to_scale).to_numpy())

# # Create a new DataFrame with the scaled data
# scaled_df = pl.DataFrame(scaled_data, schema=columns_to_scale)

# # Replace the original columns with the scaled columns
# book_df = book_df.with_columns([scaled_df[col].alias(col) for col in scaled_df.columns])

# with open(scaler_filename, 'wb') as f:
#     pickle.dump(scaler, f)
# %%
threads = []
for (ric, date) in book_df.select(["#RIC", "Date"]).unique().to_numpy():
    logging.info(f"Going to save parquet for {ric} on {date}")
    th = threading.Thread(target=__write_parquet, args=(book_df, ric, date))
    th.start()
    threads.append(th)
[ t.join() for t in threads ]

# %%
logging.info(f"Finished {os.path.abspath(__file__)}")

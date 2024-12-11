
import os
import logging
import argparse
import random
import torch
import polars as pl
import numpy as np
import polars_xdt  # noqa: F401
from datetime import time


# Data configuration
root = os.path.abspath(os.path.abspath(''))
raw_data_folder = os.path.join(root, "raw_data")
final_data_folder = os.path.join(root, "dataset", "lob")
book_filename = os.path.join(raw_data_folder, "book.csv.gz")
trades_filename = os.path.join(raw_data_folder, "trades.csv.gz")

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

trades_drop_columns = ["Trade Price Currency", "Ex/Cntrb.ID", "Seq. No.", "Domain", "Qualifiers", "Type", "Exch Time"]

def write_book_one_day(book_df, trades_df, ric, date):
    logging.info(f"Going to save csv for {ric} on {date}")
    df = book_df.filter(
            (pl.col("#RIC") == ric) & (pl.col("Date") == date)
        )
        
    first_row = df.head(1).with_columns(
            (pl.col("Date-Time").dt.truncate("1s")).alias("Date-Time")
        )
    df = pl.concat([first_row, df])
            

    df = (
        df.unique(subset=["Date-Time"], keep="last", maintain_order=True)
        .upsample(time_column="Date-Time", every="500ms", maintain_order=True)
        .with_columns(pl.col(df.columns).fill_null(strategy='forward'))
        # .select(pl.all().last())
        # .group_by_dynamic('Date-Time', every='500us')
        # .agg(pl.all().mean())
    )
    
    ticker_df = trades_df.filter(
            (pl.col("#RIC") == f"{ric.split('.')[0]}.xt") & (pl.col("Date") == date)
        )

    ticker_agg_df = ticker_df.group_by("Date-Time-Rounded", maintain_order=True).agg([
        (pl.col("Price") * pl.col("Volume")).sum().alias("VWAP_Numerator"),
        pl.col("Volume").sum().alias("Total_Volume")
        ])

    ticker_agg_df = ticker_agg_df.with_columns(
                    (pl.col("VWAP_Numerator") / pl.col("Total_Volume")).alias("VWAP")
                ).with_columns(pl.col("Total_Volume").alias("Volume")
                ).drop(["VWAP_Numerator", "Total_Volume"]
                ).rename({"Date-Time-Rounded": "Date-Time"})
                
                
    df = (
        df
        .join(ticker_agg_df, on="Date-Time", how="left")
        .with_columns(pl.col("Volume").fill_null(0))
        .with_columns(pl.col("VWAP").fill_null(strategy="forward").round(3))
        .with_columns(pl.col("VWAP").fill_null(strategy="backward").round(3))
        )
    df = df.drop(["#RIC", "Date"]).rename({"Date-Time": "date"}).drop_nulls()
    return df

def main(args):

    logging.info(f"Starting {os.path.abspath(__file__)}")

    logging.info(f"Going to lazy load {book_filename}")
    book_df = pl.scan_csv(book_filename, infer_schema_length=100000, try_parse_dates=True)


    logging.info(f"Going to eagerly load {trades_filename}")
    trades_df = pl.read_csv(trades_filename, infer_schema_length=100000, try_parse_dates=True)
    trades_df = (
            trades_df
            .filter(pl.col("#RIC")==f"{args.ric_ticker}.xt")
            .filter(
                (pl.col("Date-Time").dt.time() >= time(7, 0)) &
                (pl.col("Date-Time").dt.time() <= time(15, 30)) &
                (pl.col("Volume") != 0) &
                (~pl.col("Qualifiers").is_in([i for i in trades_df["Qualifiers"].unique().to_list() if "Previous Day" in i]))
            )
            .with_columns(
                pl.col("Date-Time").dt.date().alias("Date")
            )
            .with_columns(pl.col("Date-Time").xdt.ceil("500ms").alias("Date-Time-Rounded"))
            .drop(trades_drop_columns)
        )

    book_df = (
        book_df
        .filter(pl.col("#RIC")==f"{args.ric_ticker}.MC")
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
        .with_columns(pl.col("Date-Time").xdt.ceil("500ms").alias("Date-Time"))
        .drop_nulls()
        # .drop_nulls()
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
    final_df = pl.DataFrame()
    for (ric, date) in book_df.select(["#RIC", "Date"]).unique().sort(by="Date", maintain_order=True).to_numpy():
        df = write_book_one_day(book_df, trades_df, ric, date)
        final_df = pl.concat([final_df, df], how="vertical")
    final_df.write_csv(os.path.join(f"{args.final_data_folder}", f"{args.ric_ticker}_lob.csv"))
    logging.info(f"Null count: {final_df.null_count().sum_horizontal().sum()}")
    logging.info(f"Finished {os.path.abspath(__file__)}")

if __name__ == "__main__": 
    fix_seed = 42
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    # Logging configuration
    outfile = "./log_experiment.log"
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(outfile),
            logging.StreamHandler()
        ]
    )
    parser = argparse.ArgumentParser(description='L1-L10 Limit Order Book data processing')
    parser.add_argument('--final_data_folder', type=str, default='./dataset/lob', help='Path to output the data')
    parser.add_argument('--ric_ticker', type=str, default='GRLS', help='Ticker to process')
    args = parser.parse_args()
    logging.info(f"Arguments for script: {args}")
    main(args)
import polars as pl
from datetime import timedelta


def add_technical_indicators(df: pl.DataFrame) -> pl.DataFrame:
    print("Technikai indikátorok (Feature Engineering) számítása...")
    
    # Daily Return százalékban
    df = df.with_columns(
        (pl.col("Close").pct_change() * 100).alias("Daily_Return")
    )
    
    # SMA 20
    df = df.with_columns(
        pl.col("Close").rolling_mean(window_size=20).alias("SMA_20")
    )
    
    # EMA 20
    df = df.with_columns(
        pl.col("Close").ewm_mean(span=20, ignore_nulls=True).alias("EMA_20")
    )
    
    # RSI 14 (Relative Strength Index)
    df = df.with_columns(pl.col("Close").diff().alias("diff"))

    df = df.with_columns([
        (pl.col("Close") / pl.col("Close").shift(1)).log().alias("log_return")
    ])

    df = df.with_columns([
        pl.col("log_return").shift(-1).alias("target_log_return")
    ])

    df = df.drop_nulls()
    
    df = df.with_columns([
        pl.when(pl.col("diff") > 0).then(pl.col("diff")).otherwise(0).alias("gain"),
        pl.when(pl.col("diff") < 0).then(-pl.col("diff")).otherwise(0).alias("loss")
    ])
    
    df = df.with_columns([
        pl.col("gain").rolling_mean(window_size=14).alias("avg_gain"),
        pl.col("loss").rolling_mean(window_size=14).alias("avg_loss")
    ])
    
    # RS és végleges RSI számítása
    df = df.with_columns((pl.col("avg_gain") / pl.col("avg_loss")).alias("rs"))
    df = df.with_columns(
        pl.when(pl.col("avg_loss") == 0).then(100)
        .otherwise(100 - (100 / (1 + pl.col("rs"))))
        .alias("RSI_14")
    )
    
    # Segédoszlopok törlése és a hiányzó (null) sorok eldobása
    df = df.drop(["diff", "gain", "loss", "avg_gain", "avg_loss", "rs"])
    df = df.drop_nulls()
    
    return df

def process_and_resample_sp500_data(csv_path: str):
    print(f"S&P 500 adatok betöltése innen: {csv_path}...")
    df = pl.read_csv(csv_path)
    
    df = df.with_columns(pl.col('Time').str.strptime(pl.Datetime, "%m/%d/%Y %H:%M"))
    df = df.sort('Time')
    
    # Napi aggregálás
    df = df.with_columns(pl.col('Time').dt.date().alias('Date'))
    daily_df = (
        df.group_by('Date')
        .agg([
            pl.col('Open').first().alias('Open'),
            pl.col('High').max().alias('High'),
            pl.col('Low').min().alias('Low'),
            pl.col('Close').last().alias('Close'),
            pl.col('Volume').sum().alias('Volume')
        ])
        .sort('Date')
    )
    
    # Indikátorok hozzáadása
    daily_df = add_technical_indicators(daily_df)
    
    # Adatok felbontása
    max_date = daily_df.select(pl.col('Date').max()).item()
    split_date = max_date - timedelta(days=365)
    
    print(f"\nVágási pont: {split_date}")
    
    train_df = daily_df.filter(pl.col('Date') <= split_date)
    test_df = daily_df.filter(pl.col('Date') > split_date)
    
    return train_df, test_df

if __name__ == "__main__":
    csv_file = "data/ES_5Years_8_11_2024.csv"
    
    try:
        train_data, test_data = process_and_resample_sp500_data(csv_file)
        
        print("\nSikeres adatfeldolgozás!")
        print(f"Tanító (Train) napok száma: {train_data.height}")
        print(f"Teszt (Test) napok száma: {test_data.height}")
        
        print("\nAdatbázis szerkezete az indikátorokkal (első 3 sor):")
        print(train_data.head(3))
        
    except FileNotFoundError:
        print(f"\nHiba: Nem található a fájl: {csv_file}")
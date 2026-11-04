import polars as pl
import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
import warnings
import holidays

warnings.filterwarnings("ignore") 
from data_loader import process_and_resample_sp500_data


def add_holiday_features(df_pandas, date_series):

    # Kinyerjük az amerikai ünnepeket a vizsgált évekre
    us_holidays = holidays.US(years=range(2019, 2026))
    holiday_dates = sorted(list(us_holidays.keys()))
    
    days_to_next = []
    days_since_last = []
    
    for current_date in date_series:
        # Konvertálás datetime.date objektummá
        curr_d = pd.to_datetime(current_date).date()
        
        # Jövőbeli ünnepek
        future_h = [h for h in holiday_dates if h > curr_d]
        if future_h:
            days_to_next.append((min(future_h) - curr_d).days)
        else:
            days_to_next.append(365) # Ha nincs adat, egy nagy számot adunk
            
        # Múltbeli ünnepek
        past_h = [h for h in holiday_dates if h <= curr_d]
        if past_h:
            days_since_last.append((curr_d - max(past_h)).days)
        else:
            days_since_last.append(365)
            
    df_pandas['Days_To_Holiday'] = days_to_next
    df_pandas['Days_Since_Holiday'] = days_since_last
    
    return df_pandas


def prepare_data_for_ml(train_df: pl.DataFrame, test_df: pl.DataFrame):
    print("\nAdatok előkészítése a modellezéshez (Loghozam célváltozó + Ünnepi anomáliák)...")
    
    # Indikátorok
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 
                    'Daily_Return', 'SMA_20', 'EMA_20', 'RSI_14']
    
    # Napi loghozam kiszámítása
    train_df = train_df.with_columns(
        (pl.col('Close') / pl.col('Close').shift(1)).log().alias('Log_Return')
    )
    test_df = test_df.with_columns(
        (pl.col('Close') / pl.col('Close').shift(1)).log().alias('Log_Return')
    )
    
    # Célváltozó a másnapi loghozam (shift(-1))
    train_df = train_df.with_columns(pl.col('Log_Return').shift(-1).alias('Target_Log_Return'))
    test_df = test_df.with_columns(pl.col('Log_Return').shift(-1).alias('Target_Log_Return'))
    
    train_df = train_df.drop_nulls()
    test_df = test_df.drop_nulls()
    
    # Dátumok a távolságszámításhoz
    train_dates = train_df.select('Date').to_numpy().ravel()
    test_dates = test_df.select('Date').to_numpy().ravel()
    
    # Pandas konverzió az XGBoost bemeneti adataihoz
    X_train = train_df.select(feature_cols).to_pandas()
    X_test = test_df.select(feature_cols).to_pandas()
    
    # Feature Engineering
    X_train = add_holiday_features(X_train, train_dates)
    X_test = add_holiday_features(X_test, test_dates)
    
    # Loghozam célváltozó beállítása
    y_train_log = train_df.select('Target_Log_Return').to_numpy().ravel()
    y_test_log = test_df.select('Target_Log_Return').to_numpy().ravel()
    
    close_today_test = test_df.select('Close').to_numpy().ravel()
    y_test_nominal = test_df.with_columns(pl.col('Close').shift(-1)).drop_nulls().select('Close').to_numpy().ravel()
    if len(y_test_nominal) < len(close_today_test):
         close_today_test = close_today_test[:-1]
         y_test_log = y_test_log[:-1]
         X_test = X_test.iloc[:-1]
         test_dates = test_dates[:-1]
    
    return X_train, y_train_log, X_test, y_test_log, y_test_nominal, close_today_test, test_dates


def train_and_evaluate_xgboost(X_train, y_train_log, X_test, y_test_nominal, close_today_test):
    print("\n--- XGBOOST MODELL TANÍTÁSA (Loghozam alapon) ---")
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    
    # Tanítás az új loghozam célváltozón
    model.fit(X_train, y_train_log)
    
    # Predikció a jövőbeli loghozamra
    y_pred_log = model.predict(X_test)
    
    # Visszakódolás nominális árrá
    y_pred_nominal = close_today_test * np.exp(y_pred_log)
    
    # Kiértékelés
    mse = mean_squared_error(y_test_nominal, y_pred_nominal)
    rmse = np.sqrt(mse)
    
    actual_direction = (y_test_nominal - close_today_test) > 0
    predicted_direction = (y_pred_nominal - close_today_test) > 0
    correct_predictions = np.sum(actual_direction == predicted_direction)
    hit_rate = (correct_predictions / len(y_test_nominal)) * 100
    
    print(f"XGBoost (Loghozam konverzió után) RMSE: {rmse:.2f} pont | Hit Rate: {hit_rate:.2f} %")
    
    # Kíiratjuk, hogy mik voltak a legfontosabb tényezők (Feature Importance)
    feature_importances = model.feature_importances_
    cols = X_train.columns
    print("\nXGBoost Jellemzők fontossága (Feature Importance):")
    for col, imp in sorted(zip(cols, feature_importances), key=lambda x: x[1], reverse=True)[:5]:
        print(f"- {col}: {imp*100:.2f}%")
        
    return y_pred_nominal


def train_and_evaluate_arima(train_df, test_df, y_test_nominal, close_today_test):
    print("\n--- ARIMA MODELL TANÍTÁSA ---")
    train_close = train_df.select('Close').to_pandas()['Close']
    test_close = test_df.select('Close').to_pandas()['Close']
    
    model = ARIMA(train_close, order=(5, 1, 0))
    fitted_model = model.fit()
    
    all_close = pd.concat([train_close, test_close], ignore_index=True)
    res = fitted_model.apply(all_close)
    
    N = len(train_close)
    M = len(test_close)
    y_pred_nominal = res.fittedvalues[N+1 : N+1+len(y_test_nominal)].values
    
    mse = mean_squared_error(y_test_nominal, y_pred_nominal)
    rmse = np.sqrt(mse)
    
    actual_direction = (y_test_nominal - close_today_test) > 0
    predicted_direction = (y_pred_nominal - close_today_test) > 0
    correct_predictions = np.sum(actual_direction == predicted_direction)
    hit_rate = (correct_predictions / len(y_test_nominal)) * 100
    
    print(f"ARIMA RMSE:   {rmse:.2f} pont | Hit Rate: {hit_rate:.2f} %")
    return y_pred_nominal


def plot_results(dates, actual, xgb_preds, arima_preds):
    plt.figure(figsize=(14, 7))
    plt.plot(dates, actual, label='Valós Záróár (Target)', color='black', linewidth=2)
    plt.plot(dates, arima_preds, label='ARIMA (5,1,0) Előrejelzés', color='blue', linestyle='--', alpha=0.8)
    plt.plot(dates, xgb_preds, label='XGBoost (Loghozam-alapú) Előrejelzés', color='red', linestyle='-.', alpha=0.8)
    
    plt.title('S&P 500 Előrejelzés: XGBoost (Log-transzformált) vs. ARIMA')
    plt.xlabel('Dátum')
    plt.ylabel('Árfolyam (Pont)')
    plt.legend(loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('eredmenyek_grafikon.png', dpi=300) 
    plt.close()


if __name__ == "__main__":
    csv_file = "data/ES_5Years_8_11_2024.csv"
    
    try:
        train_data, test_data = process_and_resample_sp500_data(csv_file)
        
        X_train, y_train_log, X_test, y_test_log, y_test_nominal, close_today_test, test_dates = prepare_data_for_ml(train_data, test_data)
        
        xgb_preds = train_and_evaluate_xgboost(X_train, y_train_log, X_test, y_test_nominal, close_today_test)
        arima_preds = train_and_evaluate_arima(train_data, test_data, y_test_nominal, close_today_test)
        
        plot_results(test_dates, y_test_nominal, xgb_preds, arima_preds)
        
    except FileNotFoundError:
        print(f"\nHiba: Nem található a fájl: {csv_file}")
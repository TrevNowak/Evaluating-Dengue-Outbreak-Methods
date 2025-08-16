import pandas as pd
import numpy as np
from meteostat import Daily
from datetime import datetime
from sklearn.metrics import mean_absolute_error, confusion_matrix, recall_score, precision_score, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from gluonts.dataset.common import ListDataset
from gluonts.torch.model.deepar import DeepAREstimator
from gluonts.evaluation.backtest import make_evaluation_predictions
import torch
import os

df = pd.read_csv("data/dengue_data_sg.csv")
filtered_df = df[df['T_res'].str.contains('Week')]
cols_to_drop = ['adm_0_name', 'adm_1_name', 'adm_2_name', 'full_name', 'ISO_A0', 'FAO_GAUL_code', 'RNE_iso_code', 'IBGE_code', 'UUID']
df_cleaned = filtered_df.drop(columns=cols_to_drop)

start = datetime(2001, 12, 30)
end = datetime(2022, 12, 24)
station_id = "48698"
daily = Daily(station_id, start, end).fetch().reset_index()
daily['time'] = pd.to_datetime(daily['time'])
daily.set_index('time', inplace=True)

weekly_weather = daily.resample('W-SUN').agg({
    'tavg': 'mean',
    'prcp': 'sum'
}).reset_index()
weekly_weather.columns = ['Week', 'Temperature_C', 'Rainfall_mm']
weekly_weather["Week_Num"] = weekly_weather["Week"].dt.isocalendar().week
weekly_weather["Seasonality"] = np.sin(2 * np.pi * weekly_weather["Week_Num"] / 52)

df_cleaned['Week'] = pd.to_datetime(df_cleaned['calendar_start_date'])
weekly_weather['Week'] = pd.to_datetime(weekly_weather['Week'])
df_merged = pd.merge(df_cleaned, weekly_weather, on='Week', how='left')

df_merged["Rainfall_mm_lag1"] = df_merged["Rainfall_mm"].shift(1)
df_merged["Rainfall_mm_lag2"] = df_merged["Rainfall_mm"].shift(2)
df_merged["Rainfall_mm_lag3"] = df_merged["Rainfall_mm"].shift(3)
df_merged["Weekly_Change"] = df_merged["dengue_total"].diff()
df_merged["Surge_Indicator"] = (df_merged["Weekly_Change"] > 100).astype(int)
df_merged = df_merged.dropna().copy()

df_merged["lag1_dengue"] = df_merged["dengue_total"].shift(1)
df_merged["lag2_dengue"] = df_merged["dengue_total"].shift(2)
df_merged["lag3_dengue"] = df_merged["dengue_total"].shift(3)
df_merged["rolling_avg_dengue"] = df_merged["dengue_total"].rolling(window=3).mean()
df_merged["rolling_std_dengue"] = df_merged["dengue_total"].rolling(window=3).std()
df_merged["Rainfall_Seasonality"] = df_merged["Rainfall_mm"] * df_merged["Seasonality"]
df_final = df_merged.dropna().copy()

dynamic_cols = [
    "Rainfall_mm", "Rainfall_mm_lag1", "Rainfall_mm_lag2",
    "Temperature_C", "Seasonality", "Rainfall_Seasonality"
]
scaler = StandardScaler()
df_final[dynamic_cols] = scaler.fit_transform(df_final[dynamic_cols])



os.environ["CUDA_VISIBLE_DEVICES"] = ""
torch.set_default_device("cpu")


start_time = df_final["Week"].min()
prediction_length = 52
context_length = 104
split_point = len(df_final) - prediction_length

target_series = df_final["dengue_total"].values
feat_dynamic_real = df_final[dynamic_cols].T.values.tolist()

train_ds = ListDataset(
    [{
        "start": start_time,
        "target": target_series[:split_point],
        "feat_dynamic_real": [f[:split_point] for f in feat_dynamic_real]
    }],
    freq="W"
)

test_ds = ListDataset(
    [{
        "start": df_final["Week"].iloc[split_point],
        "target": target_series[split_point - context_length:], 
        "feat_dynamic_real": [f[split_point - context_length:] for f in feat_dynamic_real]
    }],
    freq="W"
)


#Training
estimator = DeepAREstimator(
    prediction_length=prediction_length,
    context_length=context_length,
    freq="W",
    trainer_kwargs={
        "max_epochs": 100,
        "accelerator": "cpu",
        "enable_progress_bar": True,
        "logger": False
    }
)
predictor = estimator.train(train_ds)

forecast_it, ts_it = make_evaluation_predictions(
    dataset=test_ds,
    predictor=predictor,
    num_samples=100
)

forecast = list(forecast_it)[0]
ts = list(ts_it)[0]

print("forecast.mean shape:", forecast.mean.shape)

actual = ts[-prediction_length:].to_numpy()
predicted = forecast.quantile(0.9)[-prediction_length:]

print("actual shape:", actual.shape)
print("predicted shape:", predicted.shape)

actual = ts[-prediction_length:].to_numpy().flatten()
predicted = forecast.mean[-prediction_length:]
residuals = actual - predicted

#Unified outbreak definition
def label_outbreaks(cases, threshold, min_weeks=2):
    labels = np.zeros(len(cases))
    count = 0
    for i in range(len(cases)):
        if cases[i] > threshold:
            count += 1
        else:
            if count >= min_weeks:
                labels[i - count:i] = 1
            count = 0
    if count >= min_weeks:
        labels[len(cases) - count:] = 1
    return labels.astype(int)

train_actual = target_series[:split_point]
unified_threshold = np.mean(train_actual) + 2 * np.std(train_actual)

#Residual
residuals = actual - predicted

unified_threshold = np.mean(train_actual) + 2 * np.std(train_actual)

#Label outbreaks
actual_outbreak = label_outbreaks(actual, unified_threshold)
predicted_outbreak = label_outbreaks(residuals, unified_threshold)


actual_outbreak = np.asarray(actual_outbreak).astype(int).flatten()
predicted_outbreak = np.asarray(predicted_outbreak).astype(int).flatten()


tn, fp, fn, tp = confusion_matrix(actual_outbreak, predicted_outbreak).ravel()
sensitivity = recall_score(actual_outbreak, predicted_outbreak)
specificity = tn / (tn + fp)
false_positive_rate = fp / (fp + tn)
precision = precision_score(actual_outbreak, predicted_outbreak, zero_division=0)
f1 = f1_score(actual_outbreak, predicted_outbreak, zero_division=0)
accuracy = accuracy_score(actual_outbreak, predicted_outbreak)

print(f"MAE: {np.mean(np.abs(actual - predicted)):.2f}")
print(f"RMSE: {np.sqrt(np.mean((actual - predicted)**2)):.2f}")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Sensitivity (Recall): {sensitivity:.2f}")
print(f"Specificity: {specificity:.2f}")
print(f"False Positive Rate: {false_positive_rate:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"TP={tp}, FP={fp}, FN={fn}, TN={tn}")


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from meteostat import Daily
from datetime import datetime

df = pd.read_csv("data/dengue_data_sg.csv")
filtered_df = df[df['T_res'].str.contains('Week')]
cols_to_drop = ['adm_0_name', 'adm_1_name', 'adm_2_name', 'full_name', 'ISO_A0', 'FAO_GAUL_code', 'RNE_iso_code', 'IBGE_code', 'UUID']
df_cleaned = filtered_df.drop(columns=cols_to_drop)
start = datetime(2001, 12, 30)
end = datetime(2022, 12, 24)
station_id = "48698"
data = Daily(station_id, start, end)
daily = data.fetch().reset_index()
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

#Add lag & interaction features
for i in range(1, 4):
    df_merged[f"Rainfall_mm_lag{i}"] = df_merged["Rainfall_mm"].shift(i)
df_merged["Rainfall_Temp_Interaction"] = df_merged["Rainfall_mm"] * df_merged["Temperature_C"]

df_final = df_merged.dropna()

#Rolling 85th percentile threshold
rolling_threshold = df_final["dengue_total"].rolling(window=10, center=True).quantile(0.85)

#Rainfall-adjusted threshold
rain_adjusted_threshold = rolling_threshold * (1 + 0.002 * df_final["Rainfall_mm"])
df_final["Predicted_Outbreak_Binary"] = (df_final["dengue_total"] > rain_adjusted_threshold).astype(int)

#Unified outbreak threshold:
mean_baseline = df_final["dengue_total"].mean()
std_baseline = df_final["dengue_total"].std()
unified_threshold = mean_baseline + 2 * std_baseline

def label_outbreaks(cases, threshold, min_weeks=2):
    labels = np.zeros(len(cases))
    count = 0
    for i in range(len(cases)):
        if cases.iloc[i] > threshold:
            count += 1
        else:
            if count >= min_weeks:
                labels[i - count:i] = 1
            count = 0
    if count >= min_weeks:
        labels[len(cases) - count:] = 1
    return labels.astype(int)

df_final["Severe_Outbreak_Binary"] = label_outbreaks(df_final["dengue_total"], unified_threshold)

y_true = df_final["Severe_Outbreak_Binary"]
y_pred = df_final["Predicted_Outbreak_Binary"]
conf_matrix = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = conf_matrix.ravel()

print("Confusion Matrix:\n", conf_matrix)
print(f"Accuracy: {accuracy_score(y_true, y_pred):.2f}")
print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.2f}")
print(f"Recall (Sensitivity): {recall_score(y_true, y_pred, zero_division=0):.2f}")
print(f"Specificity: {tn / (tn + fp):.2f}")
print(f"False Positive Rate: {fp / (fp + tn):.2f}")
print(f"F1 Score: {f1_score(y_true, y_pred, zero_division=0):.2f}")
print(f"TP={tp}, FP={fp}, FN={fn}, TN={tn}")

plt.figure(figsize=(14, 6))
plt.plot(df_final["Week"], df_final["dengue_total"], label='Dengue Cases')
plt.plot(df_final["Week"], rolling_threshold, color='red', linestyle='--', label='Rolling 85th Percentile')
plt.plot(df_final["Week"], rain_adjusted_threshold, color='orange', linestyle='--', label='Rainfall-Adjusted Threshold')
plt.axhline(y=severe_threshold, color='green', linestyle='--', label='Static 95th Percentile (Severe Outbreak)')
plt.title("Dengue Outbreak Detection with Rainfall-Adjusted Dynamic Threshold")
plt.xlabel

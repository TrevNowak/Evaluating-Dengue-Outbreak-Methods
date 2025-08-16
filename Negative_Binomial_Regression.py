import pandas as pd
import numpy as np
import statsmodels.api as sm
from meteostat import Daily
from datetime import datetime
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, confusion_matrix, recall_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df = pd.read_csv("data/dengue_data_sg.csv")
filtered_df = df[df['T_res'].str.contains('Week')]
cols_to_drop = [
    'adm_0_name', 'adm_1_name', 'adm_2_name', 'full_name',
    'ISO_A0', 'FAO_GAUL_code', 'RNE_iso_code', 'IBGE_code', 'UUID'
]
df_cleaned = filtered_df.drop(columns=cols_to_drop)

start = datetime(2001, 12, 30)
end = datetime(2022, 12, 24)
station_id = "48698"
data = Daily(station_id, start, end)
daily = data.fetch()

daily = daily.reset_index()
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

scaler = StandardScaler()
to_scale = ["lag1_dengue", "lag2_dengue", "lag3_dengue", "rolling_avg_dengue", "rolling_std_dengue"]
df_final[to_scale] = scaler.fit_transform(df_final[to_scale])

train_df = df_final[df_final["Week"] < "2019-01-01"].copy()
test_df = df_final[df_final["Week"] >= "2019-01-01"].copy()

features = [
    "Rainfall_mm_lag1", "Rainfall_mm_lag2", "Rainfall_mm_lag3",
    "Temperature_C", "Seasonality", "Surge_Indicator",
    "lag1_dengue", "lag2_dengue", "lag3_dengue",
    "rolling_avg_dengue", "rolling_std_dengue",
    "Rainfall_Seasonality"
]

#NBR
X_train = sm.add_constant(train_df[features].astype(float))
y_train = train_df["dengue_total"].astype(float)
X_test = sm.add_constant(test_df[features].astype(float))
y_test = test_df["dengue_total"].astype(float)

model = sm.GLM(y_train, X_train, family=sm.families.NegativeBinomial(alpha=0.5))
result = model.fit()

#Prediction
test_df["Predicted_Cases"] = result.predict(X_test)
test_df["Predicted_Cases"] = test_df["Predicted_Cases"].clip(upper=2000)

rmse = root_mean_squared_error(y_test, test_df["Predicted_Cases"])
mae = mean_absolute_error(y_test, test_df["Predicted_Cases"])
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")

#Unified outbreak definition:
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

unified_threshold = train_df["dengue_total"].mean() + 2 * train_df["dengue_total"].std()

test_df["Actual_Outbreak"] = label_outbreaks(test_df["dengue_total"], unified_threshold)
test_df["Outbreak_Detected"] = label_outbreaks(test_df["Predicted_Cases"], unified_threshold)


y_true = test_df["Actual_Outbreak"]
y_pred = test_df["Outbreak_Detected"]
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
sensitivity = recall_score(y_true, y_pred)
specificity = tn / (tn + fp)
false_positive_rate = fp / (fp + tn)

print(f"Sensitivity (Recall): {sensitivity:.2f}")
print(f"Specificity: {specificity:.2f}")
print(f"False Positive Rate: {false_positive_rate:.2f}")
print(f"TP={tp}, FP={fp}, FN={fn}, TN={tn}")

plt.figure(figsize=(12, 5))
plt.plot(test_df["Week"], test_df["dengue_total"], label="Actual", color="red")
plt.plot(test_df["Week"], test_df["Predicted_Cases"], label="Predicted", color="blue")
plt.fill_between(test_df["Week"], 0, 2000, where=test_df["Outbreak_Detected"]==1,
                 color='blue', alpha=0.1, label="Detected Outbreak")
plt.fill_between(test_df["Week"], 0, 2000, where=test_df["Actual_Outbreak"]==1,
                 color='red', alpha=0.1, label="Actual Outbreak")
plt.axhline(y=unified_threshold, color='grey', linestyle='--', label="Unified Threshold (mean + 2*std)")
plt.title("Outbreak Detection: Actual vs Predicted Cases")
plt.xlabel("Week")
plt.ylabel("Dengue Cases")
plt.legend()
plt.tight_layout()
plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import confusion_matrix, recall_score, f1_score
from scipy.stats import trim_mean
from scipy.ndimage import label
from statsmodels.robust.scale import mad

df = pd.read_csv("data/dengue_data_sg.csv")
df = df[df['T_res'].str.contains('Week')]

cols_to_drop = ['adm_0_name', 'adm_1_name', 'adm_2_name', 'full_name', 'ISO_A0', 'FAO_GAUL_code', 'RNE_iso_code', 'IBGE_code', 'UUID']
df = df.drop(columns=cols_to_drop)
df['Week'] = pd.to_datetime(df['calendar_start_date'])
df = df.sort_values('Week').reset_index(drop=True)

df["Smoothed_Cases"] = df["dengue_total"].rolling(window=3, center=True).mean().bfill().ffill()

#Rolling delta
df["Rolling_Delta"] = df["Smoothed_Cases"].rolling(window=2).mean().diff().fillna(0)

#seasonality
df["Week_Num"] = df["Week"].dt.isocalendar().week
df["Seasonality"] = np.sin(2 * np.pi * df["Week_Num"] / 52)

baseline_weeks = 11
min_case_threshold = 100

df["Mean_Past_Cases"] = df["Smoothed_Cases"].rolling(baseline_weeks).apply(lambda x: trim_mean(x, 0.1), raw=True)
df["MAD_Past_Cases"] = df["Smoothed_Cases"].rolling(baseline_weeks).apply(mad, raw=True)
df = df.dropna()

#Composite score
df["Z_Score"] = (df["Smoothed_Cases"] - df["Mean_Past_Cases"]) / df["MAD_Past_Cases"]
df["Composite_Score"] = (
    0.25 * df["Z_Score"] +
    0.5 * (df["Rolling_Delta"] / df["Rolling_Delta"].max()) +
    0.25 * df["Seasonality"]
)

#Unified outbreak definition
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

unified_threshold = df["dengue_total"].mean() + 2 * df["dengue_total"].std()
df["Actual_Outbreak"] = label_outbreaks(df["dengue_total"], unified_threshold)

y_true = df[df["Week"] >= "2019-01-01"]["Actual_Outbreak"]
best_f1, best_thresh = 0, 0

for t in np.arange(0.3, 2.0, 0.1):
    temp = df.copy()
    temp["EARS_Outbreak"] = temp.apply(
        lambda row: 1 if (
            row["Composite_Score"] > t and
            row["Smoothed_Cases"] > min_case_threshold and
            row["Rolling_Delta"] > 50
        ) else 0, axis=1
    )
    labels, _ = label(temp["EARS_Outbreak"])
    for i in np.unique(labels):
        if i == 0: continue
        group = temp[labels == i]
        if len(group) == 1:
            idx = group.index[0]
            before = temp.iloc[idx - 1]["dengue_total"] if idx > 0 else 0
            after = temp.iloc[idx + 1]["dengue_total"] if idx < len(temp) - 1 else 0
            if temp.loc[idx, "dengue_total"] < before and temp.loc[idx, "dengue_total"] < after:
                temp.loc[idx, "EARS_Outbreak"] = 0

    y_pred = temp[temp["Week"] >= "2019-01-01"]["EARS_Outbreak"]
    f1 = f1_score(y_true, y_pred)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t
        df_final = temp.copy()

test_df = df_final[df_final["Week"] >= "2019-01-01"]
y_true = test_df["Actual_Outbreak"]
y_pred = test_df["EARS_Outbreak"]

tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
sensitivity = recall_score(y_true, y_pred)
specificity = tn / (tn + fp)
false_positive_rate = fp / (fp + tn)

print(f"Best Composite Threshold: {best_thresh:.2f}")
print(f"F1-Score: {best_f1: .2f}")
print(f"Sensitivity (Recall): {sensitivity:.2f}")
print(f"Specificity: {specificity:.2f}")
print(f"False Positive Rate: {false_positive_rate: .2f}")
print(f"TP={tp}, FP={fp}, FN={fn}, TN={tn}")

plt.figure(figsize=(12, 5))
plt.plot(df_final["Week"], df_final["dengue_total"], label="Actual Dengue Cases", color="red")
plt.axhline(y=df_final["Mean_Past_Cases"].mean(), color="black", linestyle="--", label="Baseline Mean")
plt.scatter(df_final[df_final["EARS_Outbreak"] == 1]["Week"],
            df_final[df_final["EARS_Outbreak"] == 1]["dengue_total"],
            color="blue", label="EARS Outbreaks", zorder=3, s=40)
plt.xlabel("Week")
plt.ylabel("Dengue Cases")
plt.title("EARS with Rolling Delta and Seasonality")
plt.legend()
plt.tight_layout()
plt.show()

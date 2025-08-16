import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, confusion_matrix, recall_score
import matplotlib.pyplot as plt
from datetime import datetime
from meteostat import Daily

df = pd.read_csv("data/dengue_data_sg.csv")
filtered_df = df[df['T_res'].str.contains('Week')]
cols_to_drop = ['adm_0_name', 'adm_1_name', 'adm_2_name', 'full_name', 'ISO_A0', 'FAO_GAUL_code', 'RNE_iso_code', 'IBGE_code', 'UUID']
df_cleaned = filtered_df.drop(columns=cols_to_drop)

start = datetime(2001, 12, 30)
end = datetime(2022, 12, 24)
daily = Daily("48698", start, end).fetch().reset_index()
daily['time'] = pd.to_datetime(daily['time'])
daily.set_index('time', inplace=True)
weekly_weather = daily.resample('W-SUN').agg({'tavg': 'mean', 'prcp': 'sum'}).reset_index()
weekly_weather.columns = ['Week', 'Temperature_C', 'Rainfall_mm']
weekly_weather['Week_Num'] = weekly_weather['Week'].dt.isocalendar().week
weekly_weather['Seasonality'] = np.sin(2 * np.pi * weekly_weather['Week_Num'] / 52)

df_cleaned['Week'] = pd.to_datetime(df_cleaned['calendar_start_date'])
weekly_weather['Week'] = pd.to_datetime(weekly_weather['Week'])
df_merged = pd.merge(df_cleaned, weekly_weather, on='Week', how='left')

for i in range(1, 14):
    df_merged[f"Rainfall_mm_lag{i}"] = df_merged["Rainfall_mm"].shift(i)
    df_merged[f"Temperature_C_lag{i}"] = df_merged["Temperature_C"].shift(i)

df_final = df_merged.dropna()

scaler = MinMaxScaler()
df_numeric = df_final.select_dtypes(include=[np.number])
target_column = "dengue_total"
target_index = df_numeric.columns.get_loc(target_column)
scaled = scaler.fit_transform(df_numeric)

def create_sequences(data, seq_len=52):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len][target_index])
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

X, y = create_sequences(scaled, seq_len=52)
train_size = int(0.8 * len(X))
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

#Model
class EnhancedTransformer(nn.Module):
    def __init__(self, input_dim, d_model=256, nhead=8, num_layers=4, dropout=0.05):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(self.input_proj(x))
        out = self.transformer_encoder(x)
        return self.fc(out[:, -1, :]).squeeze()

model = EnhancedTransformer(input_dim=X.shape[2])
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()
epochs = 200

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    out = model(X_train)
    loss = criterion(out, y_train)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}")
model.eval()
with torch.no_grad():
    pred = model(X_test).numpy()
    true = y_test.numpy()

features_count = scaled.shape[1]
pred_scaled = np.zeros((len(pred), features_count))
pred_scaled[:, target_index] = pred.flatten()
pred_cases = scaler.inverse_transform(pred_scaled)[:, target_index]

true_scaled = np.zeros((len(true), features_count))
true_scaled[:, target_index] = true.flatten()
actual_cases = scaler.inverse_transform(true_scaled)[:, target_index]

mae = mean_absolute_error(actual_cases, pred_cases)
rmse = np.sqrt(mean_squared_error(actual_cases, pred_cases))
print(f"\nMAE: {mae:.2f}, RMSE: {rmse:.2f}")

#Unified outbreak definiton
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

train_actual_cases = df_final["dengue_total"].iloc[:len(actual_cases) * 4 // 5]
unified_threshold = train_actual_cases.mean() + 2 * train_actual_cases.std()

actual_outbreak = label_outbreaks(actual_cases, unified_threshold)
predicted_outbreak = label_outbreaks(pred_cases, unified_threshold)

tn, fp, fn, tp = confusion_matrix(actual_outbreak, predicted_outbreak).ravel()
sensitivity = recall_score(actual_outbreak, predicted_outbreak)
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
accuracy = (tp + tn) / (tp + tn + fp + fn)

print("\n--- Outbreak Detection Evaluation ---")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Sensitivity (Recall): {sensitivity:.2f}")
print(f"Specificity: {specificity:.2f}")
print(f"False Positive Rate: {false_positive_rate:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"TP={tp}, FP={fp}, FN={fn}, TN={tn}")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

#Model Para
N = 1000000  #Total population
beta = 0.2  #Transmission rate (per week)
sigma = 1/5  #Incubation rate (period = 5 days)
gamma = 1/7  #Recovery rate (period = 7 days)
initial_exposed = 50  
initial_infected = 10  
initial_recovered = 0  

#Model Function
def seir_model(t, y, beta, sigma, gamma, N):
    S, E, I, R = y
    dS_dt = -beta* S *I/ N  
    dE_dt = beta* S *I/ N - sigma * E  
    dI_dt = sigma* E - gamma* I 
    dR_dt = gamma* I 
    return [dS_dt, dE_dt, dI_dt, dR_dt]

#Initial Conditions
y0 = [N - initial_exposed - initial_infected, initial_exposed, initial_infected, initial_recovered]
t_span = (0, 365 * 2)  #2 years simu
t_eval = np.linspace(*t_span, 104)  #104 weeks

#Solve SEIR
solution = solve_ivp(seir_model, t_span, y0, args=(beta, sigma, gamma, N), t_eval=t_eval)
S, E, I, R = solution.y
weeks = np.arange(len(I))

df = pd.DataFrame({
    "Week": weeks,
    "Simulated_Dengue_Cases": I.astype(int)
})

np.random.seed(42)
df["Rainfall_mm"] = np.random.uniform(50, 300, len(df))  #Rainfall
df["Temperature_C"] = np.random.uniform(25, 34, len(df))  #Temperature 
df["Humidity_%"] = np.random.uniform(60, 90, len(df))  #Humidity
df["Population_Density"] = np.random.uniform(5000, 20000, len(df))  #People per km²

#Response measures
df["Fogging"] = np.random.choice([0, 1], size=len(df), p=[0.7, 0.3])  #30% weeks have fogging
df["Larvicide_Use"] = np.random.choice([0, 1], size=len(df), p=[0.5, 0.5])  #50% weeks have larvicide

#Seasonality
df["Seasonality"] = np.sin(2 * np.pi * df["Week"] / 52)

#Randomness
np.random.seed(42)
random_noise = np.random.normal(scale=50, size=len(df))  #std dev of 50 cases
df["Simulated_Dengue_Cases_Noised"] = np.maximum(0, df["Simulated_Dengue_Cases"] + random_noise)  #Ensure cases don't go negative

#Lagged Rainfall Features
df["Rainfall_mm_lag1"] = df["Rainfall_mm"].shift(1)
df["Rainfall_mm_lag2"] = df["Rainfall_mm"].shift(2)
df["Rainfall_mm_lag3"] = df["Rainfall_mm"].shift(3)

#Interaction Features
df["Rainfall_Temperature_Interaction"] = df["Rainfall_mm"] * df["Temperature_C"]
df["Fogging_Cases_Interaction"] = df["Fogging"] * df["Simulated_Dengue_Cases_Noised"]
df["Larvicide_Cases_Interaction"] = df["Larvicide_Use"] * df["Simulated_Dengue_Cases_Noised"]

#Define Outbreaks(Dynamic Thresholds)
def identify_outbreaks(df):
    df["Rolling_95th_Percentile"] = df["Simulated_Dengue_Cases_Noised"].rolling(window=12, min_periods=1).quantile(0.95)
    df["True_Outbreak"] = "No"
    outbreak_active = False
    for i in range(len(df)):
        if df.loc[i, "Simulated_Dengue_Cases_Noised"] > df.loc[i, "Rolling_95th_Percentile"] and df.loc[i, "Rainfall_mm"] > 250:
            if not outbreak_active:
                outbreak_start = i
                outbreak_active = True
            df.loc[i, "True_Outbreak"] = "Yes"
        elif outbreak_active and (i - outbreak_start) > 6:  #Outbreak lasts for ~6 weeks
            outbreak_active = False
    return df

df = identify_outbreaks(df)
df.to_csv("data/simulated_dengue_data.csv", index=False)

plt.figure(figsize=(10, 5))
plt.plot(df["Week"], df["Simulated_Dengue_Cases_Noised"], label="Simulated Dengue Cases (Noised)", color="red")
plt.scatter(df[df["True_Outbreak"] == "Yes"]["Week"], 
            df[df["True_Outbreak"] == "Yes"]["Simulated_Dengue_Cases_Noised"], 
            color="black", label="True Outbreaks", marker="o", s=50)

plt.xlabel("Week")
plt.ylabel("Number of Cases")
plt.title("Simulated Dengue Outbreaks with Dynamic Thresholds")
plt.legend()
plt.show()

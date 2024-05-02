import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import math


data = pd.read_csv("health_data.csv")
# print(data)

df = pd.DataFrame(data)
# print(df)

median_glucose = math.floor(df.Glucose.median())
# print(median_glucose)

df.Glucose = df.Glucose.fillna(median_glucose)
# print(df)

median_bmi = math.floor(df.BMI.median())
# print(median_bmi)

df.BMI = df.BMI.fillna(median_bmi)
# print(df)

median_insulin = math.floor(df.Insulin.median())
# print(median_insulin)

df.Insulin = df.Insulin.fillna(median_insulin)
# print(df)


reg = LinearRegression()
reg.fit(df[["Glucose","BMI","Insulin","Age"]],df["Diabetes"])

print("==== Enter the Values ====")

glucose = int(input("Enter the Glucose Level : "))
bmi = int(input("Enter the BMI : "))
insulin = int(input("Enter the Insulin : "))
age = int(input("Enter the Age Level : "))

diabetes_chance = reg.predict([[glucose, bmi, insulin, age]])
print(f"The approximate chance of Diabetis : {diabetes_chance}")
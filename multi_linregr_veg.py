import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import math


data = pd.read_csv("vegetable_rice_data.csv")

df = pd.DataFrame(data)
# print(df)

median_tax = math.floor(df.Road_Tax.median())
# print(median_tax)

df.Road_Tax = df.Road_Tax.fillna(median_tax)
# print(df)

median_price = math.floor(df.Price.median())
# print(median_price)

df.Price = df.Price.fillna(median_price)
# print(df)

reg = LinearRegression()
reg.fit(df[["Petrol_Charge", "Road_Tax", "KM_travelled"]], df["Price"])

print("** Enter the Values **")

pc = int(input("Enter the Petrol Charge : "))
rt = int(input("Enter the Road Tax : "))
km = int(input("Enter the KiloMeters Travelled : "))

new_price = reg.predict([[pc, rt, km]])
print(f"The approximate price is : {new_price}")
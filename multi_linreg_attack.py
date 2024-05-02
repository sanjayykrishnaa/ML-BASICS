import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import math


data = pd.read_csv("attack_rate.csv")
# print(data)

df = pd.DataFrame(data)
# print(df)

reg = LinearRegression()
reg.fit(df[["Troponin","Age","Blood","Cholesterol"]],df["Chance"])

print("==== Enter the Values ====")

tl = int(input("Enter the Troponin Level : "))
age = int(input("Enter the Age : "))
bp = int(input("Enter the Blood Pressure : "))
cl = int(input("Enter the Cholesterol Level : "))

attacke_rate = reg.predict([[tl, age, bp, cl]])
print(f"The approximate chance of heart attack rate is : {attacke_rate}")
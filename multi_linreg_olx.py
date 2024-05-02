import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import math


data = [
    {"Year": 2011, "Price": 155000, "Kilometer": 56155.0},
    {"Year": 2014, "Price": 190000, "Kilometer": 76000.0},
    {"Year": 2016, "Price": 325000, "Kilometer": 65000.0},
    {"Year": 2014, "Price": 221000, "Kilometer": 53000.0},
    {"Year": 2010, "Price": 140000, "Kilometer": 34000.0},
    {"Year": 2012, "Price": 230000, "Kilometer": 84882},
    {"Year": 2016, "Price": 225000, "Kilometer": 78000},
    {"Year": 2010, "Price": 185000, "Kilometer": 12800},
    {"Year": 2013, "Price": 179000, "Kilometer": 91000.0},
    {"Year": 2012, "Price": 175000, "Kilometer": 85000},
    {"Year": 2010, "Price": 180000, "Kilometer": 80000},
    {"Year": 2013, "Price": 225000, "Kilometer": 87000.0},
    {"Year": 2012, "Price": 165000, "Kilometer": 78000.0},
    {"Year": 2012, "Price": 170000, "Kilometer": 75285.0},
    {"Year": 2013, "Price": 160000, "Kilometer": 95000},
    {"Year": 2013, "Price": 215000, "Kilometer": 34000},
    {"Year": 2013, "Price": 175000, "Kilometer": 105600.0},
    {"Year": 2010, "Price": 190000, "Kilometer": 29000.0},
    {"Year": 2012, "Price": 175000, "Kilometer": 159000},
    {"Year": 2015, "Price": 160000, "Kilometer": 59000.0}
]

df = pd.DataFrame(data)
# print(df)

reg = LinearRegression()
reg.fit(df[["Year", "Kilometer"]], df["Price"])

print("== Enter the values ==")

year = int(input("Enter the Year : "))
km = int(input("Enter the KM driven : "))

price = math.floor(reg.predict([[year, km]]))
print(f"The approximate price is : {price}")
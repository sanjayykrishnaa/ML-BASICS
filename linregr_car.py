import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import math


data = {
    'Mileage': [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 
                15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 
                25, 26, None, 28, 29, None, 31, 32, 33, 34, 
                35, 36, 37, 38, 39, 40, 41, 42, None, 44, 
                45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
    'Price': [12000, 12400, 12800, 13200, 13600, 14000, 14400, 14800, 15200, 15600, 
              16000, 16400, 16800, 17200, None, 18000, 18400, 18800, 19200, 19600, 
              20000, 20400, 20800, 21200, 21600, 22000, 22400, 22800, 23200, 23600, 
              24000, 24400, 24800, 25200, None, 26000, 26400, 26800, 27200, 27600, 
              28000, 28400, 28800, 29200, 29600, 30000, 30400, 30800, 31200, None]
}

df = pd.DataFrame(data)
# print(df)

median_mileage = math.floor(df.Mileage.median())
# print(median_mileage)

df.Mileage = df.Mileage.fillna(median_mileage)
# print(df)

median_price = math.floor(df.Price.median())
# print(median_price)

df.Price = df.Price.fillna(median_price)
# print(df)


reg = LinearRegression()
reg.fit(df[["Mileage"]], df["Price"])

print("--Enter the mileage to know the price--")
Mileage = int(input("Enter the mileage : "))

price = reg.predict([[Mileage]])
print(f"Approx price of the car is : {price}")
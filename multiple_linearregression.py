import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import math


data = {
  "Area": [2600, 3000, 3200, 3600, 4000],
  "bedrooms": [3, 4, None, 3, 5],
  "age": [20, 15, 18, 30, 8],
  "price": [550000, 565000, 610000, 595000, 760000]
}


df = pd.DataFrame(data)
# print(df)

medianbedroom = math.floor(df.bedrooms.median())
# print(medianbedroom)

df.bedrooms = df.bedrooms.fillna(medianbedroom)
# print(df)

reg = LinearRegression()
reg.fit(df[["Area", "bedrooms", "age"]], df["price"])

print("---- Enter the values of the house to predict the values----")
Area = int(input("Enter the area : "))
Bedrooms = int(input("Enter the number of bedrooms : "))
Age = int(input("Enter the age of house : "))

new_price = reg.predict([[Area, Bedrooms, Age]])
print(f"* Approximate price will be * {new_price}")


# print(reg.coef_)
# print(reg.intercept_)
# result = 134.25 * 3000 - 26025 * 3 - 6825 * 40 + 40 + 383724.9999999998
# print(result)

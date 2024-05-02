import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


data = {
    'Area': [2600, 3000, 3200, 3600, 4000],
    'Price': [550000, 565000, 610000, 680000, 725000]
}

df = pd.DataFrame(data)

print(df)
print(df.columns)

plt.scatter(df[["Area"]], df["Price"], color = "red", marker = "+")
plt.xlabel("Area")
plt.ylabel("Price $")
# plt.show()


reg = LinearRegression()
reg.fit(df[["Area"]], df["Price"])

plt.plot(df["Area"], reg.predict(df[["Area"]]), color = "blue") #plotting regression line

price = int(input("Enter the area to predict price : "))
val = reg.predict([[price]])
print("M", reg.coef_)
print("B", reg.intercept_)
print(val)
plt.show()

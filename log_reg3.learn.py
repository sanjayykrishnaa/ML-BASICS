import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = {
    'age':[30,35,40,45,50,55,60,65,70,75],
    'income':[50000,55000,60000,65000,70000,75000,80000,85000,90000,95000],
    'buy_car':[0,0,1,1,1,1,1,1,1,1]
}

# 1 will represent "will buy a car", 0 represent "will not buy a car"

df = pd.DataFrame(data)
# print(df)

age = df["age"]
income = df["income"]
buy_car = df["buy_car"]
# print(age, income, buy_car)

plt.figure(figsize=(8,6))
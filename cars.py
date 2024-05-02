import pandas as pd

df = pd.read_csv("cars.csv")
# print(df.head())

inputs = df.drop("price_more_than_50k", axis = "columns")

targets = df["price_more_than_50k"]

from sklearn.preprocessing import LabelEncoder

le_car_brand = LabelEncoder()
le_car_type = LabelEncoder()
le_engine_size = LabelEncoder()

inputs["brand_n"] = le_car_brand.fit_transform(inputs["car_brand"])
inputs["type_n"] = le_car_type.fit_transform(inputs["car_type"])
inputs["engine_n"] = le_engine_size.fit_transform(inputs["engine_size"])
# print(inputs)

inputs_n = inputs.drop(["car_brand", "car_type", "engine_size"], axis = "columns")
# print(inputs_n)

from sklearn import tree

model = tree.DecisionTreeClassifier()
model.fit(inputs_n, targets)

print("The Price will be ", model.predict([[2,2,0]]))
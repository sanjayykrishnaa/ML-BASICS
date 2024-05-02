import pandas as pd
from matplotlib import pyplot as plt

data = {
    'age': [22, 25, 47,  52,  46,  56,  55,  60,  62,  61,  18,  28,  27,  29,  49,  55,  25, 58,  19,  18,  21,  26,  40,  45,  50,  54,  23],
    'bought_insurance': [ 0, 0, 1,  0, 1,1, 0,1, 1, 1, 0, 0,0,0, 1, 1, 1, 1, 0,0,0,0,1,1,1,1, 0]
}

df = pd.DataFrame(data)
# print(df)

# plt.scatter(df["age"],df["bought_insurance"], marker = "+", color = "red")
# plt.show()

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(df[["age"]], df["bought_insurance"], test_size = 0.1)
# print(len(x_train))
# print(x_test)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(x_train, y_train)

print(x_test)
predicted = model.predict(x_test)
print(predicted)

input = int(input("Enter the age to predict : "))
answer = model.predict([[input]])

print(f"Predicted for age {input} is : {answer}")

chances = model.predict_proba(x_test)
print(chances)
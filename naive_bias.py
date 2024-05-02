import pandas as pd

dataset = pd.read_csv("dataset.csv")

x = dataset.iloc[:, [1, 2, 3]].values
print(x)

y = dataset.iloc[:, -1].values
print(y)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
x[:, 0] = le.fit_transform(x[:, 0])
print(x)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Naive Bayes model

from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
model.fit(x_train, y_train)

print("----- Prediction -----")

gender = input("Enter Gender (Male/Female): ")
age = int(input("Enter age: "))
salary = int(input("Enter salary: "))

gender = le.transform([gender])[0]
prediction = model.predict([[gender, age, salary]])  # Corrected this line

if prediction == 0:
    print("The user will not purchase.")
else:
    print("The user is likely to purchase.")

print("----- Actual Data -----")
print(x_test)
#####iris####
'''from sklearn import datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

#load the data  of iris

iris = datasets.load_iris()
# print(iris)
# print(iris.target_names)
# print(iris.feature_names)

data = pd.DataFrame({
    "sepal length" : iris.data[:,0],
    "sepal width" : iris.data[:,1],
    "petal length" : iris.data[:,2],
    "petal width" : iris.data[:,3],
    "species" : iris.target
})
print(data)

x = data[["sepal length", "sepal width", "petal length", "petal width"]]
y = data["species"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)

clf = RandomForestClassifier(n_estimators = 100, criterion = "gini")
clf.fit(x_train, y_train)
print("TTS", clf.predict(x_test))
# print("accuracy", clf.score(x_test, y_test))

sepallength = int(input("Enter the sepal length : "))
sepalwidth = int(input("Enter the sepal width : "))
petallength = int(input("Enter the petal length : "))
petalwidth = int(input("Enter the petal width : "))

val = clf.predict([[sepallength, sepalwidth, petallength, petalwidth]])

if val == 1:
    print("Setosa")
    
elif val == 2:
    print("Vasicolor")
    
else : 
    print("Virginica")'''
    

    
    

    
    
    
    
    
    
    
    #####diabetis####
    
from sklearn import datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

diabetes = datasets.load_diabetes()

data = pd.DataFrame({
    "age": diabetes.data[:,0],
    "sex": diabetes.data[:,1],
    "bmi": diabetes.data[:,2],
    "bp": diabetes.data[:,3],
    "s1": diabetes.data[:,4],
    "s2": diabetes.data[:,5],
    "s3": diabetes.data[:,6],
    "s4": diabetes.data[:,7],
    "s5": diabetes.data[:,8],
    "s6": diabetes.data[:,9],
    "target": diabetes.target
})

x = data[['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']]
y = data["target"]  # Corrected target variable

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

clf = RandomForestClassifier(n_estimators=100, criterion="gini")
clf.fit(x_train, y_train)

print("TTS Predictions:", clf.predict(x_test))
print("Accuracy:", clf.score(x_test, y_test))

age = int(input("Enter  Age : "))
sex = input("Enter  sex : ")
bmi = int(input("Enter  bmi : "))
bp = int(input("Enter  bp : "))
s1 = int(input("Enter  s1 : "))
s2 = int(input("Enter  s2 : "))
s3 = int(input("Enter  s3 : "))
s4 = int(input("Enter  s4 : "))
s5 = int(input("Enter  s5 : "))
s6 = int(input("Enter  s6 : "))


######???????#########
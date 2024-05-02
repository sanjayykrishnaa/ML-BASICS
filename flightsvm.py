import pandas as pd

dataset = pd.read_csv("flight.csv")
x = dataset.iloc[:,0:3]
# print(x)

y = dataset.iloc[:,-1]
# print(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
# print(x_train)
# print(x_test)

from sklearn.svm import SVC
model = SVC(kernel="linear")
model.fit(x_train, y_train)

# prediction = model.predict(x_test)
# print(prediction)

departure = int(input("Enter the departure delay time : ")) 
arrival = int(input("Enter the arrival delay time : ")) 
distance =  int(input("Enter the Flight distance : ")) 

prediction = model.predict([[departure, arrival, distance]])
print(f"Chance of flight delay is : {prediction}")

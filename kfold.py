from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits

digits = load_digits()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.3)

lr = LogisticRegression(solver = "liblinear", multi_class = "ovr")
lr.fit(x_train, y_train)
# print(lr.score(x_test, y_test))

svm = SVC(gamma = "auto")
svm.fit(x_train, y_train)
# print(svm.score(x_test, y_test))

rf = RandomForestClassifier(n_estimators=40)
rf.fit(x_train, y_train)
# print(rf.score(x_test, y_test))

from sklearn.model_selection import KFold
kf = KFold(n_splits=3)

# print(kf.split([1,2,3,4,5,6,7,8,9]))
for train_index, test_index in kf.split([1,2,3,4,5,6,7,8,9]):
    print(train_index, test_index)
    
def getscore(model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    return model.score(x_test, y_test)

# print(getscore(SVC(), x_train, x_test, y_train, y_test))


from sklearn.model_selection import StratifiedKFold
folds = StratifiedKFold(n_splits=3)

score_logistc = []
score_svm = []
score_rf = []

for train_index, test_index in folds.split(digits.data, digits.target):
    x_train, x_test, y_train, y_test=digits.data[train_index],digits.data[test_index],digits.target[train_index],digits.target[test_index]
    score_svm.append(getscore(SVC(gamma="auto"), x_train, x_test, y_train, y_test))
    score_logistc.append(getscore(LogisticRegression(solver="liblinear"), x_train, x_test, y_train, y_test))
    score_rf.append(getscore(RandomForestClassifier(n_estimators=40), x_train, x_test, y_train, y_test))

print(score_logistc)
print(score_rf)
print(score_svm)








#NEW MODEL

from sklearn.model_selection import cross_val_score

print(cross_val_score(LogisticRegression(solver="liblinear", multi_class="ovr"), digits.data, digits.target, cv=3))
print(cross_val_score(RandomForestClassifier(n_estimators=40), digits.data, digits.target, cv=3))
print(cross_val_score(SVC(gamma="auto"), digits.data, digits.target, cv=3))

import numpy as np
score1 = cross_val_score(RandomForestClassifier(n_estimators=100), digits.data, digits.target, cv=3)
print(np.average(score1))
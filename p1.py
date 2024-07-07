#import lib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report

#load the data
data = pd.read_csv("sna_march24.csv")
print(data)

#check for null data
print(data.isnull().sum())

#check for dypliacted data
print(data.duplicated().sum())

#feature and target 
features = data.drop(["User ID", "Purchased"], axis = "columns")
target = data["Purchased"]

#handle cat data
nfeatures = pd.get_dummies(features)
print(features)
print(nfeatures)

#train aand test
x_train, x_test, y_train, y_test = train_test_split(nfeatures.values, target)

#model
model = GaussianNB()
model.fit(x_train, y_train)

#confusion matrix
cm = confusion_matrix(y_test, model.predict(x_test))
print(cm)

#classification report
cr = classification_report(y_test, model.predict(x_test))
print(cr)

#predict
age = int(input("Enter age "))
salary = float(input("Enter Salary "))
gender = int(input("Press 1 for Female and 2 for Male "))
if gender == 1:
	d = [[age, salary, 1, 0]]
else :
	d = [[age, salary,0, 1]]

#internal 
ans = model.predict(d)
print(ans)

res = model.predict_proba(d)
print(res)
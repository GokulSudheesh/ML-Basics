import tensorflow as tf
import sklearn
from sklearn import model_selection
from sklearn import linear_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
data = pd.read_csv("student-mat.csv", sep=";")
data = data[["G1","G2","G3","studytime","failures","absences"]]#600 students
#print(data.head()) head() #returns top 5 rows
predict = "G3"
x = np.array(data.drop([predict], 1))#1 means its for columns
y = np.array(data[predict])# dependant variable

#Saving the model with an accuracy>95
'''
acc = 0
while (acc < 0.98):
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.1)
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    #test size is 10% its takes 10% data for testing and training and splits them accordingly.
    linear = linear_model.LinearRegression()#setting up linear regression

    linear.fit(x_train,y_train)
    acc = linear.score(x_test, y_test)

print(acc)
with open("studentModel.pickle", "wb") as f:
    pickle.dump(linear,f)
    pickle.dump(x_train, f)
    pickle.dump(x_test, f)
    pickle.dump(y_train, f)
    pickle.dump(y_test, f)

#we trained it once and saved the object "linear" into a pickle file this object can be saved and used later
'''
pickle_in = open("studentModel(98).pickle", "rb")
linear = pickle.load(pickle_in)
x_train = pickle.load(pickle_in)
x_test = pickle.load(pickle_in)
y_train = pickle.load(pickle_in)
y_test = pickle.load(pickle_in)

print (linear.score(x_test, y_test))
print(x_train)
print(x_test)
print(y_train)
print(y_test)
print("Coefficients: \n", linear.coef_) # b1x1 + b2x2 + b3x3 + b4x4 + b5x5 + .... + intercept
print("Intercept: \n", linear.intercept_)
predictions = linear.predict(x_test)
#Predictions based on user-Input
'''predictions = linear.predict([[12,12,5,0,0],[2,2,4,0,0]])
print(predictions)'''
#we are preedicting y (dependant variable)
i = 0
for p in predictions:
    print("Predicted val: ",round(p,2),"| Predictors: ",x_test[i],"| Actual val: ",y_test[i],"| Diff: ",y_test[i]-p)
    i+=1
#output would look like: (predicted value of y) [list of other attributes y depends on (predictors)] (actual value of y)
p = "failures"
style.use("ggplot")#grids
plt.scatter(data[p],data[predict])#plt.scatter(x,y)
plt.xlabel(p)
plt.ylabel(predict)
plt.show()

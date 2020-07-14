import pandas as pd
import numpy as np
import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, preprocessing

data = pd.read_csv("car.data")
print(data)
#https://towardsdatascience.com/choosing-the-right-encoding-method-label-vs-onehot-encoder-a4434493149b
#Label Encoding is just assigning numerical values to data
le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))
cls_text = ["acc","vgood","unacc","good"]

'''print(cls)
for x in cls:
    print (x)
A = [1,2,3]
B = [4,5,6]
C = [7,8,9]
D = list(zip(A, B, C))
print(D)'''
X = list(zip(buying,maint,door,persons,lug_boot,safety)) #Zip pairs one list with the other (To make a row)
Y = list(cls)
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.1)
#print(x_train,x_test,y_train,y_test )
model = KNeighborsClassifier(n_neighbors = 9)#n_neighbors is K
model.fit(x_train,y_train)

acc = model.score(x_test,y_test)
print(acc)

data_text = ["unacc","acc","good","vgood"]
predictions = model.predict(x_test)
#we are preedicting y (dependant variable)
i = 0
for p in predictions:
    print("Predicted val: ",cls_text[p],"| Predictors: ",x_test[i],"| Actual val: ",cls_text[y_test[i]])
    N = model.kneighbors([x_test[i]], 9, True)
    i += 1
    #Returns the no.of neighbors as a list with each element in the list representing the distance
    #function takes in a 2D Array,no.of neighbors (K) and a boolean which would return distance if True
    #print(N)
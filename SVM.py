import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics

# Loading the built-in data set
cancer_data = datasets.load_breast_cancer()
#print(cancer_data.feature_names)
#print(cancer_data.target_names)

x = cancer_data.data
y = cancer_data.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.2)
#print(x_train, y_train)
classes = ['malignant', 'benign']
features = ['mean radius', 'mean texture', 'mean perimeter', 'mean area',
            'mean smoothness', 'mean compactness', 'mean concavity',
            'mean concave points', 'mean symmetry', 'mean fractal dimension',
            'radius error', 'texture error', 'perimeter error', 'area error',
            'smoothness error', 'compactness error', 'concavity error',
            'concave points error', 'symmetry error', 'fractal dimension error',
            'worst radius', 'worst texture', 'worst perimeter', 'worst area',
            'worst smoothness', 'worst compactness', 'worst concavity',
            'worst concave points', 'worst symmetry', 'worst fractal dimension']

model = svm.SVC(kernel = 'linear')
# 'C' is setting the soft margin for the data set
#model = svm.SVC(kernel = 'linear', C = 2) # C = 0 for hard margin

model.fit(x_train, y_train)
predictions = model.predict(x_test)
accuracy = metrics.accuracy_score(y_test, predictions)
print(accuracy)

i = 0
for p in predictions:
    print(x_test[i], classes[predictions[i]], classes[y_test[i]])
    i +=1
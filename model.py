#Importing all the necessary libraries
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import joblib
print("Imported all pacakges")

#fetching mnist dataset from scikit-learn
mnist=fetch_openml(name='mnist_784', version=1, cache=True)
mnist.target=mnist.target.astype(np.int8)
print("fetched the data")

# sorting the data 
def sort_by_target(mnist):
    reorder_train = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[:60000])]))[:, 1]
    reorder_test = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[60000:])]))[:, 1]
    mnist.data[:60000] = mnist.data[reorder_train]
    mnist.target[:60000] = mnist.target[reorder_train]
    mnist.data[60000:] = mnist.data[reorder_test + 60000]
    mnist.target[60000:] = mnist.target[reorder_test + 60000]

print("defined the sort by target")

sort_by_target(mnist)
X,y=mnist['data'],mnist['target']
X_train,X_test,y_train,y_test=X[:60000],X[60000:],y[:60000],y[60000:]
shuffle_index=np.random.permutation(60000)
X_train,y_train=X_train[shuffle_index],y_train[shuffle_index]

print("sorted the data")

# Converting all pixel values to one and zero
def convert_to_binary(X):
    for x in X:
        for i in range(len(x)):
            if x[i]!=0:
                x[i]=1

print("Converted to binary")
convert_to_binary(X_train)
# Using KNeighborsClassifier
knn_clf = KNeighborsClassifier(weights='distance', n_neighbors=4)
print("model created")

#training the model
knn_clf.fit(X_train,y_train)
print("model is trained")

#Saving the model
file='model.sav'
joblib.dump(knn_clf,file)
# score=cross_val_score(knn_clf,X_train,y_train,cv=10)
# print(f'Cross val score: {score}')


some_digit = X[36000]
print(f'some_digit is {y[36000]}')
for i in range(len(some_digit)):
    if some_digit[i]!=0:
        some_digit[i]=1
print("some digit converted to binary")
model=joblib.load(file)
predict=model.predict([some_digit])
print(f'some digit predicted is:{predict[0]}')
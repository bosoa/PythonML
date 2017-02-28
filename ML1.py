# ML.py : machine learning studying
import matplotlib.pyplot as plt

# using scikit
from sklearn import datasets
from sklearn import svm # support vector machine

digits = datasets.load_digits()

# images
#print(digits.data) 
#print(digits.target)
#print(digits.images[0])

# gamma means gap between calculation
clf = svm.SVC(gamma=0.0001, C=100)


print(len(digits.data)) # number of data
# storage all of the test sets
# training data here
x,y = digits.data[:-10], digits.target[:-10]
clf.fit(x,y)

# predict here
predictIndex = -6
print('Prediction:',clf.predict(digits.data[predictIndex]))
plt.imshow(digits.images[predictIndex], cmap=plt.cm.gray_r, interpolation="nearest") 
plt.show()

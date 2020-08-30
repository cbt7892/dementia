import numpy as np
from sklearn import svm
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

# load data
data_set = np.genfromtxt("feature_set_dem.csv", delimiter=",")
data_set = data_set[1:]  # remove label row
np.random.shuffle(data_set)

# split into input and output
x = data_set[:, 1:]
y = data_set[:, 0]

# separate into testing and training sets
split_mark = int(.75 * len(x))
x_train = x[0:split_mark]
x_test = x[split_mark:]
y_train = y[0:split_mark]
y_test = y[split_mark:]

# feature scaling
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# make model
clf = svm.SVC(kernel="linear")
clf.fit(x_train, y_train)

# evaluate model
print(metrics.accuracy_score(y_test, clf.predict(x_test)))


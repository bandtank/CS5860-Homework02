from sklearn import svm

X = [[0,0], [1,1]]
y = [0,1]

clf = svm.SVC()
clf.fit(X,y)

p = clf.predict([[2,2]])
print(p)
print(clf.support_vectors_)
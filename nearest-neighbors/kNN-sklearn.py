import numpy as np
from sklearn import preprocessing,cross_validation,neighbors
import pandas as pd
from simData import simClust


df = pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?',-99999, inplace=True)

# Don't need id column
df.drop(['id'],1,inplace=True)

X = np.array(df.drop(['class'],1))
y = np.array(df['class'])

### Try with own simulated data
##data = simClust([[0,1],[2,3]],500)
##headers = ['x1','x2','y']
##df = pd.DataFrame(data,columns = headers)
##df['y'] = df['y'].apply(int)
##X = np.array(df.drop(['y'],1))
##y = np.array(df['y'])

X_train,X_test,y_train,y_test = cross_validation.train_test_split(X,y,test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train,y_train)

accuracy = clf.score(X_test,y_test)
print(accuracy)

sample_measures = np.array([[4,2,1,1,1,2,3,2,1],[4,2,1,3,1,2,3,2,2]])
# get rid of warning
sample_measures.reshape(len(sample_measures), -1)

prediction= clf.predict(sample_measures)

print(prediction)



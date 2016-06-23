import numpy as np
from matplotlib import style
import matplotlib.pyplot as plt
from collections import Counter
style.use('fivethirtyeight')
import warnings
import pandas as pd
import random

from simData import simClust

##for i in dataset:
##    for ii in dataset[i]:
##        plt.scatter(ii[0],ii[1], s=100,color=i)
##
##plt.show

def kNN(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups')

    distances = []
    for group in data:
        for features in data[group]:
            distance = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([distance, group])

    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1] / k
            
    return vote_result, confidence


df = pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?',-99999, inplace=True)

# Don't need id column
df.drop(['id'],1,inplace=True)
full_data = df.astype(float).values.tolist()

##full_data = simClust([[0,0],[2,2]],100,labels=['k','r'])

test_size = 0.2
train_set = {2:[],4:[]}
test_set = {2:[],4:[]}

##train_set = {'r':[],'k':[]}
##test_set = {'r':[],'k':[]}

random.shuffle(full_data)
train_data = full_data[:-int(test_size*len(full_data))]
test_data = full_data[-int(test_size*len(full_data)):]

for i in train_data:
    train_set[i[-1]].append(i[:-1])

for i in test_data:
    test_set[i[-1]].append(i[:-1])

correct = 0
total = 0

for group in test_set:
    for data in test_set[group]:
        total += 1
        vote, confidence = kNN(train_set, data, k=3)        
        if group == vote:
            correct += 1

accuracies.append(correct/total)
    
print(np.mean(accuracies))





##
##predict  = [2,2]
##[plt.scatter(x[0],x[1],c=x[2]) for x in data]
##plt.scatter(predict[0],predict[1],c='g',s=100)
##
##print(kNN(full_data,[2,2],k=5))
##plt.show()

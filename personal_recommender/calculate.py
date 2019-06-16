import csv
import pandas as pd

test = pd.read_csv('../data/personal/test.csv')
usrid = []
movieid = []
for i in range(len(test['userId'])):
    if not test['userId'][i] in usrid:
        usrid.append(test['userId'][i])
    if not test['movieId'][i] in movieid:
        movieid.append(test['movieId'][i])

data_all = []
index = 0
for user in usrid:
    this_user = []
    if index >= len(test['userId']):
        break
    while test['userId'][index] == user:
        index += 1
        if index >= len(test['userId']):
            break
        this_user.append(test['movieId'][index])
    print(len(this_user))
    data_all.append(this_user)

print('data all', len(data_all))

result = pd.read_csv('../ensemble_recommender/result.csv')
print('pred', len(result['userId']))
posi = 0
neg = 0
for i in range(len(result['userId'])):
    print(i)
    temp = result['result'][i]
    temp = temp[1:-1].split(',')
    temp = [int(x) for x in temp]
    # print(temp)
    # break
    # print(temp)
    # print(data_all[i])
    # break
    for movieid in list(temp):
        if movieid in data_all[i]:
            posi += 1
        else:
            neg += 1
print(posi, neg, posi / float(posi + neg))




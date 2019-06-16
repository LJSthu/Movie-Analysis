import pandas as pd
import numpy as np
import os
from surprise import Reader, Dataset, SVD, evaluate
from surprise import KNNBaseline
from surprise import KNNWithMeans
from surprise import KNNBasic
import csv
import random

ratings = pd.read_csv('../data/personal/ratings.csv')
usrid = []
movieid = []
for i in range(len(ratings['userId'])):
    if not ratings['userId'][i] in usrid:
        usrid.append(ratings['userId'][i])
    if not ratings['movieId'][i] in movieid:
        movieid.append(ratings['movieId'][i])


print(len(usrid))
print(len(movieid))

train = []
valid = []
data_all = []
index = 0
for user in usrid:
    this_user = []
    if index >= len(ratings['userId']):
        break
    while ratings['userId'][index] == user:
        index += 1
        if index >= len(ratings['userId']):
            break
        temp = [ratings['userId'][index], ratings['movieId'][index], ratings['rating'][index]]
        this_user.append(temp)
    print(len(this_user))
    data_all.append(this_user)

threshold = 0.85
test_data = []
with open("./train.csv", "w") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['userId', 'movieId','rating'])
    for this_user in data_all:
        length = len(this_user)
        for i in range(length):
            temp = random.random()
            if temp < threshold:
                writer.writerow(this_user[i])
            else:
                test_data.append(this_user[i])



with open("./test.csv", "w") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['userId', 'movieId', 'rating'])
    for row in test_data:
        writer.writerow(row)


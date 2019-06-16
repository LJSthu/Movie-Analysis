#-*-coding:utf-8-*-
# 在评分矩阵中使用kNN去度量用户之间的相似度

import pandas as pd
import numpy as np
import os
from surprise import Reader, Dataset, SVD, evaluate
from surprise import KNNBaseline
from surprise import KNNWithMeans
from surprise import KNNBasic
import csv


class Personal_KNN_recommender:
    def __init__(self, mode=0):
        self.index = pd.read_csv('../data/personal/movies.csv')
        self.reader = Reader()
        self.ratings = pd.read_csv('../data/personal/train.csv')
        self.testings = pd.read_csv('../data/personal/test.csv')
        data = Dataset.load_from_df(self.ratings[['userId', 'movieId', 'rating']], self.reader)
        trainset = data.build_full_trainset()
        sim_options = {'name': 'pearson_baseline', 'user_based': True}
        if mode == 0:
            self.algo = KNNBaseline(sim_options=sim_options)
        elif mode == 1:
            self.algo = KNNWithMeans(sim_options=sim_options)
        elif mode == 2:
            self.algo = KNNBasic(sim_options=sim_options)
        else:
            exit(0)
        self.userid = []
        for i in range(len(self.testings['userId'])):
            if not self.testings['userId'][i] in self.userid:
                self.userid.append(self.testings['userId'][i])
        self.algo.fit(trainset)

    def get_similar_users(self, usrID, num=10):
        user_inner_id = self.algo.trainset.to_inner_uid(usrID)
        user_neighbors = self.algo.get_neighbors(user_inner_id, k=num)
        user_neighbors = [self.algo.trainset.to_raw_uid(inner_id) for inner_id in user_neighbors]
        # print(user_neighbors)
        return user_neighbors

    def debug(self):
        similar_users = self.get_similar_users(1, 1)
        print(self.ratings[self.ratings.userId == 1].head())
        for i in similar_users:
            print(list(self.ratings[self.ratings.userId == i]['movieId']))

    def recommend(self, usrID, num=5):
        existed_movie = list(self.ratings[self.ratings.userId==usrID]['movieId'])
        similar_users = self.get_similar_users(usrID, num)
        movies_dict = {}
        for i in similar_users:
            movie = list(self.ratings[self.ratings.userId == i]['movieId'])
            vote = list(self.ratings[self.ratings.userId == i]['rating'])
            for j in range(len(vote)):
                if not (movie[j] in existed_movie):
                    if movie[j] in movies_dict.keys():
                        movies_dict[movie[j]] += vote[j]
                    else:
                        movies_dict[movie[j]] = vote[j]   # 从最相似的用户中挑选出没看过的电影，评分相加
        result = sorted(movies_dict.items(), key=lambda x: x[1], reverse=True)  # 对评分进行排序
        result = result[:num]  # 挑选出最高评分的10部电影
        # print(result)
        recommending = []
        recommending_id = []
        for i in result:
            recommending.append(self.index[self.index.movieId==i[0]]['title'])
            recommending_id.append(i[0])
        return recommending, recommending_id  # 返回推荐的电影名字和id

    def test(self, num = 10):
        result = []
        for user in self.userid:
            _, ids = self.recommend(user, num)
            # print(ids)
            result.append(ids)

        with open("./result.csv", "w") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['userId', 'result'])
            for i,row in enumerate(result):
                writer.writerow([self.userid[i], row])




test = Personal_KNN_recommender()
# result = test.recommend(6, 10)
# for i in result:
#     print(i)

test.test(10)


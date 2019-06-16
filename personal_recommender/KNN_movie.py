#-*-coding:utf-8-*-
# 在评分矩阵中使用kNN去度量用户之间的相似度

import pandas as pd
import numpy as np
import os
from surprise import Reader, Dataset, SVD, evaluate
from surprise import KNNBaseline
from surprise import KNNWithMeans
from surprise import KNNBasic


class Movie_KNN_recommender:
    def __init__(self, mode=0):
        self.index = pd.read_csv('../data/personal/movies.csv')
        self.reader = Reader()
        self.ratings = pd.read_csv('../data/personal/ratings.csv')
        data = Dataset.load_from_df(self.ratings[['userId', 'movieId', 'rating']], self.reader)
        trainset = data.build_full_trainset()
        sim_options = {'name': 'pearson_baseline', 'user_based': False}
        if mode == 0:
            self.algo = KNNBaseline(sim_options=sim_options)
        elif mode == 1:
            self.algo = KNNWithMeans(sim_options=sim_options)
        elif mode == 2:
            self.algo = KNNBasic(sim_options=sim_options)
        else:
            exit(0)

        self.algo.fit(trainset)

    def get_similar_movies(self, movieID, num=10):
        movie_inner_id = self.algo.trainset.to_inner_iid(movieID)
        movie_neighbors = self.algo.get_neighbors(movie_inner_id, k=num)
        movie_neighbors = [self.algo.trainset.to_raw_iid(inner_id) for inner_id in movie_neighbors]
        print(movie_neighbors)
        return movie_neighbors

    def debug(self):
        similar_users = self.get_similar_movies(1, 1)
        print(self.ratings[self.ratings.userId == 1].head())
        for i in similar_users:
            print(list(self.ratings[self.ratings.userId == i]['movieId']))

    def recommend(self, movieID, num=10):
        movie_similar = self.get_similar_movies(movieID, num)
        recommending = []
        for i in movie_similar:
            recommending.append(self.index[self.index.movieId == i]['title'])
        return recommending




test = Movie_KNN_recommender()
result = test.recommend(122922, 10)
for i in result:
    print(i.values[0])



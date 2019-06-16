#-*-coding:utf-8-*-

import pandas as pd
import numpy as np
import os
from surprise import Reader, Dataset, SVD, evaluate
from surprise import KNNBaseline
from surprise import KNNWithMeans
from surprise import KNNBasic
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import sys
import os
sys.path.insert(0, '..')
from personal_recommender.KNN_movie import Movie_KNN_recommender
from personal_recommender.KNN_user import Personal_KNN_recommender
from personal_recommender.Personal_SVD import Personal_SVD_recommender

# 首先用KNN对输入的用户进行相似度匹配，然后挑选出最接近的10个其他用户
# 之后对于选出的电影，根据与用户给出的电影相似度推荐靠前的十部电影
# 接受的输入为用户ID以及电影ID

class KNN_ensemble:
    def __init__(self, mode=0):
        # self.movie = Movie_KNN_recommender()
        self.user = Personal_KNN_recommender()
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
        self.sim = self.algo.compute_similarities()
    def cal_similarity(self, movieID, waitingID):
        movie_inner_id = self.algo.trainset.to_inner_iid(movieID)
        waiting_inner_id = self.algo.trainset.to_inner_iid(waitingID)
        return self.sim[movie_inner_id, waiting_inner_id]
    def showSeenMovies(self, usrID):
        print("\n\nThe user has seen movies below: ")
        movies = []
        for i in range(len(self.ratings['userId'])):
            if self.ratings['userId'][i] == usrID:
                movies.append(self.index[self.index.movieId == self.ratings['movieId'][i]]['title'])
        for i in movies:
            print(i.values[0])
    def showInputMovie(self, movieID):
        print("\n\nThe user's input movie is: ")
        print(self.index[self.index.movieId==movieID]['title'])
        print('\n\n')
    def recommend(self, usrID, movieID, num=10):
        self.showSeenMovies(usrID)
        self.showInputMovie(movieID)
        _, first_ids = self.user.recommend(usrID, 50)

        similarity = {}
        for i in first_ids:
            similarity[i] = self.cal_similarity(movieID, i)
        result = sorted(similarity.items(), key=lambda x: x[1], reverse=True)  # 对相似度进行排序
        result = result[:num]
        movie = []
        for i in result:
            movie.append(self.index[self.index.movieId == i[0]]['title'])
        return movie




test = KNN_ensemble()
result = test.recommend(34,480)

for i in result:
    print(i.values[0])
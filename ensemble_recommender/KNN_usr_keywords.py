#-*-coding:utf-8-*-

import pandas as pd
import numpy as np
import os
from surprise import Reader, Dataset, SVD, evaluate
from surprise import KNNBaseline
from surprise import KNNWithMeans
from surprise import KNNBasic
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import sys
import os
sys.path.insert(0, '..')
from personal_recommender.KNN_movie import Movie_KNN_recommender
from personal_recommender.KNN_user import Personal_KNN_recommender
from personal_recommender.Personal_SVD import Personal_SVD_recommender

# 首先用KNN对输入的用户进行相似度匹配，然后挑选出最接近的10个其他用户
# 之后对于选出的电影，根据与用户给出的关键词提取词向量，由相似度推荐靠前的十部电影
# 接受的输入为用户ID以及关键词什么的

class KNN_usr_keywords_ensemble:
    def __init__(self, mode=0):
        # self.movie = Movie_KNN_recommender()
        self.user = Personal_KNN_recommender()
        self.map = pd.read_csv('../data/personal/links.csv')
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
        # self.sim = self.algo.compute_similarities()

    def showSeenMovies(self, usrID):
        print("\n\nThe user has seen movies below: ")
        movies = []
        for i in range(len(self.ratings['userId'])):
            if self.ratings['userId'][i] == usrID:
                movies.append(self.index[self.index.movieId == self.ratings['movieId'][i]]['title'])
        for i in movies:
            print(i.values[0])

    def handle_keywords(self, keywords):
        self.movies = pd.read_csv('../data/tmdb_5000_movies.csv')
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.search_id = self.movies['overview'].shape[0]

        self.overview = self.movies['overview']
        self.overview.loc[self.overview.shape[0]] = keywords   # 把输入的关键词列表添加进去作为最后一项
        self.overview = self.overview.fillna('')
        tfidf_matrix = self.tfidf.fit_transform(self.overview)
        self.cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
        self.indices = pd.Series(self.movies.index, index=self.movies['title']).drop_duplicates()

    def convert_id2tmdbid(self, idlist):
        ids = []
        for id in idlist:
            ids.append(int(self.map[self.map['movieId'] == id]['tmdbId'].values[0]))

        return ids
    def convert_id2title(self, idlist):
        titles = []
        for id in idlist:
            titles.append(self.movies[self.movies['id']==id]['title'])
        return titles

    def cal_similarity(self, titles):
        similarity = {}
        for title in titles:
            temp = title.to_dict().values()
            if len(temp) == 0:
                print(title)
                continue
            temp0 = temp[0]
            idx = self.indices[temp0]
            similarity[temp0] = self.cosine_sim[idx, self.search_id]
        return similarity


    def recommend(self, usrID, keywords, num=10):   # keywords中是一组关键词
        self.handle_keywords(keywords)
        self.showSeenMovies(usrID)
        print("\n\nThe user input the keywords: ")
        print(keywords)
        _, first_ids = self.user.recommend(usrID, 50)
        # print(first_ids)
        tmdb_id = self.convert_id2tmdbid(first_ids)
        titles = self.convert_id2title(tmdb_id)
        similarity = self.cal_similarity(titles)
        result = sorted(similarity.items(), key=lambda x: x[1], reverse=True)  # 对评分进行排序
        result = result[:num]
        return result




test = KNN_usr_keywords_ensemble()
result = test.recommend(1,'spy hit strike hero war death soldier army')
for i in result:
    print(i)

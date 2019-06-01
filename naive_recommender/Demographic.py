# This function is for Demographic recommender.

import pandas as pd
import numpy as np

# a general recommender method, not sensitive to the interests and tastes of a particular user
class Demographic_recommender:
    def __init__(self):
        df1 = pd.read_csv('../data/tmdb_5000_credits.csv')
        df2 = pd.read_csv('../data/tmdb_5000_movies.csv')
        df1.columns = ['id', 'tittle', 'cast', 'crew']
        df2 = df2.merge(df1, on='id')
        self.average_vote = df2['vote_average'].mean()
        self.average_count = df2['vote_count'].quantile(0.9)
        # choose more votes than at least 90% of the movies in the list
        self.movies = df2.copy().loc[df2['vote_count'] >= self.average_count]

    def weighted_rating(self, x):
        m = self.average_count
        c = self.average_vote
        v = x['vote_count']
        r = x['vote_average']
        # Calculation based on the IMDB formula
        return (v / (v + m) * r) + (m / (m + v) * c)

    def recommend(self, n=10):
        self.movies['score'] = self.movies.apply(self.weighted_rating, axis=1)
        self.movies = self.movies.sort_values('score', ascending=False)
        print('Using the Demographic recommending method, followings are the top %d score movies' % n)
        print(self.movies[['title', 'vote_count', 'vote_average', 'score']].head(n))





# test = Demographic_recommender()
# test.score(10)
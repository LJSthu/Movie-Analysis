import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

class Content_recommender:
    def __init__(self):
        df1 = pd.read_csv('../data/tmdb_5000_credits.csv')
        df2 = pd.read_csv('../data/tmdb_5000_movies.csv')
        df1.columns = ['id', 'tittle', 'cast', 'crew']
        self.movies = df2.merge(df1, on='id')

        self.tfidf = TfidfVectorizer(stop_words='english')
        self.movies['overview'] = self.movies['overview'].fillna('')
        tfidf_matrix = self.tfidf.fit_transform(self.movies['overview'])
        self.cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
        self.indices = pd.Series(self.movies.index, index=self.movies['title']).drop_duplicates()

    def recommend(self, title):
        # Get the index of the movie that matches the title
        idx = self.indices[title]

        # Get the pairwsie similarity scores of all movies with that movie
        sim_scores = list(enumerate(self.cosine_sim[idx]))

        # Sort the movies based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get the scores of the 10 most similar movies
        sim_scores = sim_scores[1:11]

        # Get the movie indices
        movie_indices = [i[0] for i in sim_scores]

        # Return the top 10 most similar movies
        return self.movies['title'].iloc[movie_indices]


test = Content_recommender()
print(test.recommnd('The Dark Knight Rises'))


import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class Keyword_recommender:
    def __init__(self):
        df1 = pd.read_csv('../data/tmdb_5000_credits.csv')
        df2 = pd.read_csv('../data/tmdb_5000_movies.csv')
        df1.columns = ['id', 'tittle', 'cast', 'crew']
        self.movies = df2.merge(df1, on='id')
        features = ['cast', 'crew', 'keywords', 'genres']
        for feature in features:
            self.movies[feature] = self.movies[feature].apply(literal_eval)
    def get_director(self, x):
        for i in x:
            if i['job'] == 'Director':
                return i['name']
        return np.nan

    def get_list(self, x):
        if isinstance(x, list):
            names = [i['name'] for i in x]
            # Check if more than 3 elements exist. If yes, return only first three. If no, return entire list.
            if len(names) > 3:
                names = names[:3]
            return names
        return []
    def clean_data(self, x):
        if isinstance(x, list):
            return [str.lower(i.replace(" ", "")) for i in x]
        else:
            # Check if director exists. If not, return empty string
            if isinstance(x, str):
                return str.lower(x.replace(" ", ""))
            else:
                return ''
    def process_data(self):
        self.movies['director'] = self.movies['crew'].apply(self.get_director)

        features = ['cast', 'keywords', 'genres']
        for feature in features:
            self.movies[feature] = self.movies[feature].apply(self.get_list)
        # print(self.movies[['title', 'cast', 'director', 'keywords', 'genres']].head(3))
        features = ['cast', 'keywords', 'director', 'genres']

        for feature in features:
            self.movies[feature] = self.movies[feature].apply(self.clean_data)

    def create_soup(self, x):
        return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])

    def recommend(self, title):
        self.process_data()
        self.movies['soup'] = self.movies.apply(self.create_soup, axis=1)
        count = CountVectorizer(stop_words='english')
        count_matrix = count.fit_transform(self.movies['soup'])
        self.cosine_sim = cosine_similarity(count_matrix, count_matrix)
        self.movies = self.movies.reset_index()
        indices = pd.Series(self.movies.index, index=self.movies['title'])
        idx = indices[title]

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


test = Keyword_recommender()
print(test.recommend('The Dark Knight Rises'))
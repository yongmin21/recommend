# 컨텐츠 기반 필터링
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from helpers import weighted_rating
from api import DataApi

# Garbage Collector - use it like gc.collect()
import gc

class ContentBasedF:
    """Recommnder Class from query_id inputs and outs similar movie ids
    Arguments:
    X (pd.DataFrame) : Movies Data
    query_id (int) : movie_id that you want to get recommend  
    """
    def __init__(self, X: pd.DataFrame, query_id: int, top_n=10):
        self._X = X
        self.top_n = top_n
        self._query_id = query_id
        self._matrix = None
        self._sim = None

    def _calculate_weighted_ratings(self, f=1):
        weighted_rates = weighted_rating(self._X['meta.rate_counts'], self._X['meta.rate_average'], f)
        self._X['weighted_rating'] = weighted_rates

    def preprocess(self, f=1):
        self._X['new_genre'] = self._X['genres'].str.replace('|', ' ', regex=False)
        self._X['movieId'] = self._X['movieId'].astype(int)
        self._calculate_weighted_ratings(f)
        print("Data Preprocess Done ")
        print(self._X.head(5))

    def fit(self):
        # calculate weighted vote and split genres
        self.preprocess()

        # make countvector to genre
        vectorizer = CountVectorizer()
        self._matrix = vectorizer.fit_transform(self._X['new_genre'])

        query_genre = self._X[self._X['movieId'] == self._query_id]['new_genre']
        query_vec = vectorizer.transform(query_genre)

        # calculate cosine similarity between one movie and cosine matrix
        self._sim = cosine_similarity(query_vec, self._matrix).flatten()
        print("Calculate Similarity Done")
        gc.collect()

    def recommend(self):
        self.fit()

        sorted_ind = self._sim.argsort()[-(self.top_n * 2):][::-1]
        title_index = self._X.loc[self._X['movieId'] == self._query_id].index.values

        similar_indexes = sorted_ind[sorted_ind != title_index]
        return self._X.iloc[similar_indexes].sort_values('weighted_rating', ascending=False)[:self.top_n].movieId.values

if __name__ == "__main__":
    # body = "movies"
    # api = DataApi(body)
    # data = pd.json_normalize(api.READ_DATA())
    
    # query_id = 1
    # top_n = 10
    
    # recommender = ContentBasedF(data, query_id, top_n)
    # recommender.fit()
    # recommendation = recommender.recommend()

    # print(recommendation)
    pass
import pandas as pd
import numpy as np

from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# for Custom Transformer and pipelining
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

import sys
sys.path.append("../src")

from helpers import weighted_rating
from api import DataApi
import gc

body = "movies"
api = DataApi(body)
data = pd.json_normalize(api.READ_DATA())

class PreprocessTransformer(BaseEstimator, TransformerMixin):
    """Custom Transformer Class for sklearn Pipeline that calculate weighted ratings
    Arguments:
    X (pd.DataFrame) : Data to calculate weighted_rates
    """
    def __init__(self):
        print(">>> init() called.")
    
    def fit(self, X, y=None):
        print(">>> fit() called.")
        return self
    
    def transform(self, X, y=None):
        _X = X.copy() # make copy of X
        _X['new_genre'] = _X['genres'].str.replace('|', ' ', regex=False)
        return _X

class Recommender:
    """Recommnder Class from query_id inputs and outs similar movie ids
    Arguments:
    X (pd.DataFrame) : Movies Data
    query_id (int) : movie_id that you want to get recommend  
    """
    def __init__(self, X, query_id):
        self._X = X
        self._pipeline = Pipeline(steps = [
                                      ('transformer', PreprocessTransformer()),
                                      ('vectorizer', CountVectorizer(dtype=np.int16))
                                      ]
                                    )
        self.outs = {}
        self._query_id = query_id

    def _calculate_weighted_ratings(self):
        weighted_rates = weighted_rating(self._X['meta.rate_counts'], self._X['meta.rate_average'], f=1)
        self._X['weighted_rating'] = weighted_rates
    
    def fit(self):
        self._calculate_weighted_ratings()
        print("Calculate Weighted Votes Done ")
        
        self._X = self._pipeline['transformer'].transform(self._X)
        self.outs["matrix"] = self._pipeline.fit_transform(self._X)
        
        # calculate cosine similarity between one movie and cosine matrix
        X_query = self._X[self._X['movieId'] == self._query_id]['new_genre']
        query_vec = self._pipeline['vectorizer'].transform(X_query)
        self.outs["query_sim"] = cosine_similarity(query_vec, self.outs["matrix"]).flatten()
        print("Calculate Similarity Done")
        gc.collect()
    
    def transform(self, top_n=10):
        query_sim = self.outs["query_sim"]
        sorted_index = query_sim.argsort()[-(top_n * 2):][::-1]
        return_index = self._X.loc[self._X['movieId'] == self._query_id].index.values
        similar_indexes = sorted_index[sorted_index != return_index]
        
        response = (self._X.iloc[similar_indexes]
                    .sort_values('weighted_rating', ascending=False)[:top_n]
                    .movieId
                    .values)
        return response
    
if __name__ == "__main__":
    recommender = Recommender(data, 1)
    recommender.fit()
    recommendations = recommender.transform()
    print(recommendations)
    

    


    

import pandas as pd
import numpy as np

class DataCleansing:
    def __init__(self, data: pd.DataFrame, regex_pattern: str):
        self._data = data
        #self._col_name = col_name
        self._regex_pattern = regex_pattern

    @property
    def data(self):
        return self._data

    # def get_col_name(self):
    #     return self._col_name

    @property
    def regex_pattern(self):
        return self._regex_pattern

    @regex_pattern.setter
    def regex_pattern(self, regex_pattern: str):
        self._regex_pattern = regex_pattern

    def split_genres(self, inplace: bool):
        """
        장르1|장르2 형태를 [장르1, 장르2]로 분리하는 함수
        """
        if inplace:
            self._data['genres'] = self._data['genres'].str.split("|")
        else:
            return self._data['genres'].str.split("|")

    def extract_regex_in_title(self):
        """
        title에 있는 상영연도를 분리하는 함수
        """
        return self._data['title'].str.extract(self._regex_pattern, expand=False)

def weighted_rating(vote_counts, vote_average, q=0.6 ,f=1):
    """
    q는 평점을 부여하기 위한 최소 투표 횟수
    f는 적은 투표수에 대한 가중치
    """
    v = vote_counts
    m = v.quantile(q)
    R = vote_average
    C = np.mean(R) # Punishment for low votes

    score = ((v/(v+f*m) * R) +  (f*m/(v+f*m) * C))
    return score
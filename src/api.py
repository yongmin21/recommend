import requests
import json

class DataApi:
    
    def __init__(self, body):
        self._header = {
            'Content-Type' : 'application/json'
            }
        self._timeout = 5
        self._url = f"http://ec2-3-137-69-122.us-east-2.compute.amazonaws.com:8080/api/{body}"

    @property
    def url(self):
        return self._url

    def _CHECK_DATA_TYPE(self, data):
        if data['movieId'].dtype != str:
            data['movieId'] = data['movieId'].astype(str)

        elif data['year'].dtype != int:
            data['year'] = data['year'].astype(int)

        return data

    def CREATE_DATA(self, data):
        """
        서버에 데이터를 새로 생성하는 함수
        """ 
        
        response = requests.post(self._url,
                                headers = self._header,
                                data = data)
        response.raise_for_status()
        

    def READ_DATA(self, movieId=None):
        """
        서버에 데이터를 가져오는 함수
        """
        if movieId != None:
            response = requests.get(f"{self._url}/{movieId}",
                                    timeout = self._timeout,
                                    verify = False)

        response = requests.get(self._url)

        return response.json()

    def UPDATE_DATA(self, movieId, data):
        """
        서버에 데이터를 갱신하는 함수
        """
        response = requests.put(f"{self._url}/{movieId}",
                                headers = self._header,
                                json = data,
                                verify = False)

        response.raise_for_status()

    def DELETE_DATA(self, movieId, data):
        """
        서버에 데이터를 삭제하는 함수
        """
        response = requests.delete(f"{self._url}/{movieId}",
                                   headers = self._header,
                                   json = data)
        print(response.status_code)


import pandas as pd
import numpy as np

'''
loadData 함수
이 함수는 train,test 데이터를 불러오고 데이터셋을 리턴해줍니다.
'''

def loadData(trainDataPath:str,testDataPath:str):
    totalList = [trainDataPath,testDataPath]
    result = [] # [train, test]
    for dataPath in totalList:
        df = pd.read_csv(dataPath)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['year'] = pd.DatetimeIndex(df['datetime']).year
        df['month'] = pd.DatetimeIndex(df['datetime']).month

        result.append(df)

    return result
'''
fillnaBehind 함수
이 함수는 nan이 있는 데이터를 주변에 있는 
데이터의 평균으로 바꾸어줍니다.
'''
def fillnaBehind(Data:pd.DataFrame):
    nullBoolean = Data[['구미 혁신도시배수지 유출유량 적산차']].isnull()
    nullData = Data[nullBoolean].index
    for i in nullData:
        Data.loc[[i,'구미 혁신도시배수지 유출유량 적산차']] = \
            (Data.loc[[i-1,'구미 혁신도시배수지 유출유량 적산차']] +
              Data.loc[[i+1,'구미 혁신도시배수지 유출유량 적산차']])//2

'''
outlierDataremove 함수
이 함수는 이상치 값(값이 너무 높은 값만)만 제거하고
'''

def outlierDataToNan(Data:pd.DataFrame,thres = 3.0):
    
    q1=Data['구미 혁신도시배수지 유출유량 적산차'].quantile(0.25)
    q2=Data['구미 혁신도시배수지 유출유량 적산차'].quantile(0.5)
    q3=Data['구미 혁신도시배수지 유출유량 적산차'].quantile(0.75)

    iqr = q3-q1

    remove = Data['구미 혁신도시배수지 유출유량 적산차']> q3+thres*iqr
    Data = Data[remove] = np.NaN

    Data[Data['구미 혁신도시배수지 유출유량 적산차']<0] = np.NaN
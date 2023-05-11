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
    nullBoolean = Data['구미 혁신도시배수지 유출유량 적산차'].isnull()
    nullData = Data[nullBoolean].index
    # print(nullData)
    for i in nullData:
        # if Data.loc[i-1,'구미 혁신도시배수지 유출유량 적산차'] == np.NaN:
        #     Data.loc[i,'구미 혁신도시배수지 유출유량 적산차'] = Data.loc[i+1,'구미 혁신도시배수지 유출유량 적산차']
        if Data.loc[i+1,'구미 혁신도시배수지 유출유량 적산차'] == np.NaN:
            Data.loc[i,'구미 혁신도시배수지 유출유량 적산차'] = Data.loc[i-1,'구미 혁신도시배수지 유출유량 적산차']
        else:
            Data.loc[i,'구미 혁신도시배수지 유출유량 적산차'] = \
                (Data.loc[i-1,'구미 혁신도시배수지 유출유량 적산차'] +
                Data.loc[i+1,'구미 혁신도시배수지 유출유량 적산차'])//2.0
    
    print(nullData)
    Data = Data.fillna(method='pad')

    print(Data.isnull().sum())

    return Data

'''
fillZero 함수
이 함수는 0가 연속적인 것들을 채우고 평균으로 바꿉니다.
'''
def fillZero(Data:pd.DataFrame):
    zeroIndex = Data[Data['구미 혁신도시배수지 유출유량 적산차']==0].index
    # prev = zeroIndex[0]
    indexlist = []
    
    for i in zeroIndex:
        if len(indexlist) == 0:
            indexlist.append(i)
            prev = zeroIndex[0]
        elif prev + 1 == i: # 연속적이냐?
            sum+=Data.loc[i]['구미 혁신도시배수지 유출유량 적산차']
            
            indexlist.append(i)
        else: # 연속이 아닐때
            pass

'''
outlierDataToNan 함수
이 함수는 이상치 값(값이 너무 높은 값만)만 NaN으로
'''

def outlierDataToNan(Data:pd.DataFrame,thres = 3.0):
    # 이 부분은 조심히 다뤄야 함.(fect. 교수)
    q1=Data['구미 혁신도시배수지 유출유량 적산차'].quantile(0.25)
    q2=Data['구미 혁신도시배수지 유출유량 적산차'].quantile(0.5)
    q3=Data['구미 혁신도시배수지 유출유량 적산차'].quantile(0.75)

    iqr = q3-q1

    remove = Data['구미 혁신도시배수지 유출유량 적산차']> q3+thres*iqr
    Data[remove] = np.NaN

    Data[Data['구미 혁신도시배수지 유출유량 적산차']<0] = np.NaN

    return Data



'''
XDataToXAndYSeq(Data,step)
학습하

'''
def XDataToXAndYSeq(Data:pd.DataFrame,step = 24):
    
    X = Data['구미 혁신도시배수지 유출유량 적산차'].to_numpy()
    
    XSeq = []
    YSeq = []
    
    startIndex = step + 1

    for i,j in enumerate(range(startIndex,len(X))):
        XSeq.append(X[i:j-1])
        YSeq.append(X[j])

    
    return np.array(XSeq), np.array(YSeq).reshape((-1,1))
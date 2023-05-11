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
    nullBoolean = Data['구미 혁신도시배수지 유출유량 적산차'] == 0
    nullData = Data[nullBoolean].index
    # print(nullData)
    for i in nullData:
        # if Data.loc[i-1,'구미 혁신도시배수지 유출유량 적산차'] == np.NaN:
        #     Data.loc[i,'구미 혁신도시배수지 유출유량 적산차'] = Data.loc[i+1,'구미 혁신도시배수지 유출유량 적산차']
        if Data.loc[i+1,'구미 혁신도시배수지 유출유량 적산차'] == 0:
            Data.loc[i,'구미 혁신도시배수지 유출유량 적산차'] = Data.loc[i-1,'구미 혁신도시배수지 유출유량 적산차']
        else:
            Data.loc[i,'구미 혁신도시배수지 유출유량 적산차'] = \
                (Data.loc[i-1,'구미 혁신도시배수지 유출유량 적산차'] +
                Data.loc[i+1,'구미 혁신도시배수지 유출유량 적산차'])//2.0
    
    print(nullData)
    

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
            
        elif prev + 1 == i: # 연속적이냐?
            indexlist.append(i)
            
        else: # 연속이 아닐때
            total = Data.loc[prev+1]['구미 혁신도시배수지 유출유량 적산차']
            print(total)
            indexlist.append(prev+1)
            cnt = len(indexlist)
            Data.loc[np.array(indexlist)] = total / cnt
            indexlist.clear()
        prev = i

    print(Data[Data['구미 혁신도시배수지 유출유량 적산차']==0].index)
    return Data
'''
fillprevValue(Data:pd.Dataframe)
na값들은 이전의 데이터로 덮어 씌우는 함수입니다.
'''
def fillprevValue(Data:pd.DataFrame):
    
    return Data.fillna(method='pad')

'''
outlierDataToNan 함수
이 함수는 이상치 값(값이 너무 높은 값만)만 NaN으로

수정 5/11 : 너무 수치가 높은 값들은 데이터 형태에 따라서 부적합하다는 판단으로 극대값은 전처리 안함
단, 0보다 작은 값과 Nan 값은 다 0으로 처리
'''

def outlierDataToNan(Data:pd.DataFrame,low:bool = False,high:bool = False,thres = 3.0):

    if high:
        # 이 부분은 조심히 다뤄야 함.(fect. 교수)
        q1=Data['구미 혁신도시배수지 유출유량 적산차'].quantile(0.25)
        q2=Data['구미 혁신도시배수지 유출유량 적산차'].quantile(0.5)
        q3=Data['구미 혁신도시배수지 유출유량 적산차'].quantile(0.75)

        iqr = q3-q1

        remove = Data['구미 혁신도시배수지 유출유량 적산차']> q3+thres*iqr
        Data[remove] = Data['구미 혁신도시배수지 유출유량 적산차'].quantile(0.9)

    # Data[Data['구미 혁신도시배수지 유출유량 적산차']<0] = np.NaN

    if low:
        Data[Data['구미 혁신도시배수지 유출유량 적산차']<=0]= 0
        Data[Data['구미 혁신도시배수지 유출유량 적산차']<np.NaN] = 0

    return Data
'''
XDataToXAndYSeq(Data,step)
시계열 데이터 X,Y 변수로 변환해주는 값입니다, 기본 step은 24로 지정되어 있습니다.

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
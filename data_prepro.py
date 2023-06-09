import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from sklearn.preprocessing import StandardScaler


'''
loadData 함수
이 함수는 train,test 데이터를 불러오고 데이터셋을 리턴해줍니다.
'''

def loadData(trainDataPath:str,testDataPath:str,trainYearRange1 = 2017,trainYearRange2 = 2020):
    totalList = [trainDataPath,testDataPath]
    result = [] # [train, test]
    for dataPath in totalList:
        df = pd.read_csv(dataPath)
        print(df.info())
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['year'] = pd.DatetimeIndex(df['datetime']).year
        df['month'] = pd.DatetimeIndex(df['datetime']).month

        result.append(df)

    result[0] = result[0][result[0]['year']>=trainYearRange1]
    result[0] = result[0][result[0]['year']<=trainYearRange2]

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
        if Data.loc[i-1,'구미 혁신도시배수지 유출유량 적산차']>0 and  Data.loc[i+1,'구미 혁신도시배수지 유출유량 적산차']>0:
            Data.loc[i,'구미 혁신도시배수지 유출유량 적산차'] = \
                (Data.loc[i-1,'구미 혁신도시배수지 유출유량 적산차'] +
                Data.loc[i+1,'구미 혁신도시배수지 유출유량 적산차'])//2.0
        else:
            continue
    print(nullData)
    
    Data.reset_index(inplace=True)
    print(Data.isnull().sum())

    return Data

'''
fillZero 함수
이 함수는 0가 연속적인 것들을 채우고 평균으로 바꿉니다.
'''
def fillZero(Data:pd.DataFrame):
    zeroIndex = Data[Data['구미 혁신도시배수지 유출유량 적산차']==0].index.tolist()
    print(zeroIndex)
    splitList = continuedNumbersplit(zeroIndex)
    for i in splitList:
        changeNumber = i[-1]+1 / len(i)
        Data.loc[i[0]:i[-1],'구미 혁신도시배수지 유출유량 적산차'] = changeNumber
    # prev = zeroIndex[0]
    # indexlist = []
    
    # for i in zeroIndex:
    #    if len(indexlist) == 0:
    #        indexlist.append(i)
    #        
    #    elif prev + 1 == i: # 연속적이냐?
    #        indexlist.append(i)
            
    #    else: # 연속이 아닐때
    #        total = Data.loc[prev+1]['구미 혁신도시배수지 유출유량 적산차']
    #        print(total)
    #        indexlist.append(prev+1)
    #        cnt = len(indexlist)
    #        Data.loc[np.array(indexlist)] = total / cnt
    #        indexlist.clear()
    #    prev = i
    Data.reset_index(inplace=True)
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
        Data[Data['구미 혁신도시배수지 유출유량 적산차']<=0]= 0.0
        
    Data[Data['구미 혁신도시배수지 유출유량 적산차'] == np.NaN] = 0.0
    return Data
'''
XDataToXAndYSeq(Data,step)
시계열 데이터 X,Y 변수로 변환해주는 값입니다, 기본 step은 24로 지정되어 있습니다.

'''
def XDataToXAndYSeq(Data:pd.DataFrame,step = 24,output=1):
    
    X = Data['구미 혁신도시배수지 유출유량 적산차'].to_numpy()
    
    XSeq = []
    YSeq = []
    
    # startIndex = step

    # for i,j in enumerate(range(startIndex,len(X)-output)):
    #     XSeq.append(X[i:j])
    #     YSeq.append(X[j:j+output])

    for i in range(step,len(X)-output):
        print(i)
        XSeq.append(X[i-step:i])
        YSeq.append(X[i:i+output])

    
    
    return np.array(XSeq).reshape(-1,1,step), np.array(YSeq).reshape((-1,output))


'''
continuedNumberSplit(df:pd.DataFrame)
연속적인 숫자들끼리 묶는 함수 입니다.'''

def continuedNumbersplit(index:list):

    packet = []

    tmp = []

    v = index.pop(0)

    tmp.append(v)

    print(v)

    while(len(index)>0):
        vv = index.pop(0)
        print(vv)
        if v+1 == vv:
            tmp.append(vv)
            v = vv
        else:
            packet.append(tmp)
            tmp = []
            tmp.append(vv)
            v = vv

    packet.append(tmp)

    return packet

'''
showValueGrap(df):
이 함수는 적산차의 값을 그래프로 출력하는 함수입니다.

'''
def showValueGrap(df:pd.DataFrame,range1,range2):
    plt.figure(figsize=(50,10))
    for i in range(len(df)):
        
        Y = df[i]['구미 혁신도시배수지 유출유량 적산차'].to_numpy()
        X = np.arange(1,len(Y)+1)
        
        
        plt.plot(X[range1:range2],Y[range1:range2])

    
'''
linearInterpolation(df):
이 함수는 선형보간법을 이용하여 결측치를 극복하는 데이터입니다.
https://rfriend.tistory.com/682
'''
def linearInterpolation(df:pd.DataFrame):
    
    Y = df['구미 혁신도시배수지 유출유량 적산차'].to_numpy()
    X = np.arange(1,len(Y)+1)

    y_new_linear = interpolate.interp1d(X,Y,kind='linear')
    y_new = y_new_linear(X)

    df['구미 혁신도시배수지 유출유량 적산차'] = y_new

    return df

def triple_exponential_smoothing(X,L,α,β,γ,ϕ):

	def sig_ϕ(ϕ,m):
		return np.sum(np.array([np.power(ϕ,i) for i in range(m+1)]))

	C, S, B, F = (np.zeros( X.shape[0] ) for i in range(4))
	S[0], F[0] = X[0], X[0]
	B[0] = np.mean( X[L:2*L] - X[:L] ) / L
	m = 12
	sig_ϕ = sig_ϕ(ϕ,m)
	for t in range(1, X.shape[0]):
		S[t] = α * (X[t] - C[t % L]) + (1 - α) * (S[t-1] + ϕ * B[t-1])
		B[t] = β * (S[t] - S[t-1]) + (1-β) * ϕ * B[t-1]
		C[t % L] = γ * (X[t] - S[t]) + (1 - γ) * C[t % L]
		F[t] = S[t] + sig_ϕ * B[t] + C[t % L]
	return S

def reScale(X,reScale:StandardScaler = None):
    value = X['구미 혁신도시배수지 유출유량 적산차'].to_numpy().reshape(-1,1)
    if reScale is None:
        reScale = StandardScaler()
        reScale.fit(value)
    X['구미 혁신도시배수지 유출유량 적산차'] = reScale.transform(value).reshape(-1,)
    return [X,reScale]


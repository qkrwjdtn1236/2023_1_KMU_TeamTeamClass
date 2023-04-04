
import pandas as pd
import numpy as np

'''
dataMean 함수
년월별 평균을 구합니다.
'''

def dataMean(Data:pd.DataFrame,rangeYear1:int,rangeYear2:int):
    historymonth = []
    years = list(range(rangeYear1,rangeYear2+1))
    for i in years:
        print(i,'년')
        dummy = Data[Data['year'] == i]
        print(f'{i}년 평균',np.mean(dummy[['구미 혁신도시배수지 유출유량 적산차']].to_numpy()))
        monthMeanData = []
        for j in range(1,13):
            filter = dummy[dummy['month'] == j]
            monthAvg = np.mean(filter[['구미 혁신도시배수지 유출유량 적산차']].to_numpy())
            print(f'{j}월 :',monthAvg)
            monthMeanData.append(monthAvg)
        print()

        historymonth.append(monthMeanData)
    
    return years,historymonth # x => years, y => historymonthUsage

def showMonthHistoryGrap(year:list,historymonth:list):
    import matplotlib.pyplot as plt

    for i,j in enumerate(year):
        plt.plot(range(1,13),historymonth[i],label=str(year[i]))

    plt.legend()
    plt.show()
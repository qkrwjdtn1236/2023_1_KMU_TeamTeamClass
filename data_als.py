
import pandas as pd
import numpy as np

def dataMean(Data:pd.DataFrame,rangeYear1:int,rangeYear2:int):
    historymonth = []
    years = range(rangeYear1,rangeYear2+1)
    for i in years:
        print(i,'년')
        dummy = Data[Data['year'] == i]
        print(f'{i}년 평균',np.mean(dummy[['구미 혁신도시배수지 유출유량 적산차']].to_numpy()))

        for j in range(1,13):
            filter = dummy[dummy['month'] == j]
            monthAvg = np.mean(filter[['구미 혁신도시배수지 유출유량 적산차']].to_numpy())
            print(f'{j}월 :',monthAvg)
            historymonth.append(monthAvg)
            str(years)
        print()
    

    
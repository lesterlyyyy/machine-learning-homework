import pandas as pd
import warnings
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] = 'SimHei'
"""
读入数据
"""
def read_data():

    warnings.filterwarnings('ignore')
    data= pd.read_excel('./2023年模式识别与机器学习数据集汇总.xls')
    """
    数据预处理
    """
    color=set(data['喜欢颜色'])
    color.remove(np.nan)
    color.remove('rgb')
    color_dic ={c:i for i, c in enumerate(color)}
    """
    删除编号、籍贯
    """
    colume= [i for i in  data.columns.values if i not in ['编号','籍贯']]
    data=data[colume]

    data['喜欢颜色']=[color_dic[c] if c in color_dic.keys() else np.nan for c in  data['喜欢颜色'] ]
    last_three_data=data.iloc[:,-3:]

    last_three_data= last_three_data.fillna(value=0)

    data=data.iloc[:,:-3]

    my_imputer=SimpleImputer()
    data_imputed=my_imputer.fit_transform(data)
    data=pd.DataFrame(data_imputed,columns=data.columns)
    data=pd.concat([data,last_three_data],axis=1,)

    data_array=np.array(data,dtype=np.float)
    train_data=data_array[:,[1,2,4,5]]#身高、体重
    return train_data

def normalize(train_data):
    min,max=np.min(train_data,axis=0,keepdims=True),np.max(train_data,axis=0,keepdims=True)
    re=(train_data-min)/(max-min)
    return re,min,max

def unnormalize(data,min,max):
    return data*(max-min)+min

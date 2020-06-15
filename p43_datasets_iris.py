from sklearn import datasets
from pandas import DataFrame
import pandas as pd 

x_data = datasets.load_iris().data
y_data = datasets.load_iris().target
print('x_data from datasets: \n', x_data)
print('y_data from datasets: \n', y_data)

x_data = DataFrame(x_data, columns=['花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度'])
pd.set_option('display.unicode.east_asian_width', True) # 设置列名对齐
print('x_data add index: \n', x_data)

x_data['类别'] = y_data
print('x_data add a column: \n', x_data)
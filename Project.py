import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def df_filtered_product(dataframe, prod_num):
    df_prod = dataframe[dataframe['Products'] == 'Product ' + str(prod_num)]
    return df_prod


def preprocessing(df, product_number):
    useless_columns = ['Customers', 'Category', 'Segment', 'Regione', 'Provincia', 'Channel']
    for column in useless_columns:
        df = df.drop(column, axis=1)
    df = df_filtered_product(df, product_number)             # Choose the number of the product
    df = df.groupby(['Data Rif']).sum().reset_index()
    date_range = pd.date_range('2017-01-02', '2019-03-31', freq='D').to_series()
    week_num = len(date_range) // 7
    index = 0

    sales = []
    for week in range(0, week_num):
        STU = 0
        for day in range(0, 7):
            if index == len(df):
                break
            elif date_range[week*7 + day] == df['Data Rif'][index]:
                STU += df['Standard Units'][index]
                index += 1
        sales.append([date_range[week*7], STU])
    df_fin = pd.DataFrame(sales, columns=['Week', 'STU'])
    # df.set_index('Week Number', inplace=True)
    return df_fin

"""
def first_order_diff(dataset):
    STU_diff = np.dataset['Standard Units per Week']
    dataset['Standard Units per Week_diff'] = np.diff(dataset['Standard Units per Week'])
    dataset = dataset.dropna()
    return dataset
    
"""

def stationarity_test(df):
    result = adfuller(df.STU.dropna())
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')


dataset = pd.read_excel(r"C:\Users\Enrico\Google Drive\DATA MINING\CHALLENGE FATER\serie_tamponi.xlsx")
df = dataset.copy()                     # Copy the original dataset in a new identical one

df_clean = preprocessing(df, 3)
print(df_clean.head())
stationarity_test(df_clean)

plt.plot(df_clean['Week'], df_clean['STU'])
plt.show()

plot_acf(df_clean.STU)
plt.show()

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


def df_filtered_product(dataframe, prod_num):
    df_prod = dataframe[dataframe['Products'] == 'Product ' + str(prod_num)]
    return df_prod


def preprocessing(df):
    useless_columns = ['Customers', 'Category', 'Segment', 'Regione', 'Provincia', 'Channel']
    for column in useless_columns:
        df = df.drop(column, axis=1)
    df = df_filtered_product(df, 7)             # Choose the number of the product
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
    df_fin = pd.DataFrame(sales, columns=['Week', 'Standard Units per Week'])
    # df.set_index('Week Number', inplace=True)
    return df_fin


dataset = pd.read_excel(r"C:\Users\Enrico\Google Drive\DATA MINING\CHALLENGE FATER\serie_tamponi.xlsx")
df = dataset.copy()                     # Copy the original dataset in a new identical one

df_clean = preprocessing(df)


#pd.plotting.register_matplotlib_converters()
plt.plot(df_clean['Week'], df_clean['Standard Units per Week'])
plt.show()

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, acf
from statsmodels.tsa.arima_model import ARIMA

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


def stationarity_test(df, differenced):
    """
    We are going to perform Augmented Dicker-Fuller Test.
    It works as an hypothesis test, with
    H0 = Series is stationary
    H1 = Series is not stationary
    result of ADF-Test are ['ADF Statistics', 'p-value', '#Lags Used', 'Number of observations used']
    """
    if differenced:

        result = adfuller(df['Sales after differencing'])
        print(f'ADF Statistic: {result[0]}')
        print(f'p-value: {result[1]}')
        if result[1] <= 0.05:
            print("Evidence against the null-hypothesis, series look stationary!")
        else:
            print("Weak evidence against the null-hypothesis, showing that the series is likely to be non-stationary")
    else:
        result = adfuller(df['STU'])
        print(f'ADF Statistic: {result[0]}')
        print(f'p-value: {result[1]}')
        if result[1] <= 0.05:
            print("Evidence against the null-hypothesis, series look stationary!")
        else:
            print("Weak evidence against the null-hypothesis, showing that the series is likely to be non-stationary!")


def differencing(df):
    """ Differencing at first order """

    df['Sales after differencing'] = df['STU'] - df['STU'].shift(1)
    return df


def seasonal_differencing(df):
    """ Differencing at first order in case of seasonality """

    df['Sales after differencing'] = df['STU'] - df['STU'].shift(12)
    return df


def plotting_data(dataframe):
    plt.plot(dataframe['Sales after differencing'])
    plt.show()


def plotting_autocorr(dataframe):
    plot_acf(dataframe['Sales after differencing'].iloc[1:], lags=40)
    plt.show()


def plotting_part_autocorr(dataframe):
    plot_pacf(dataframe['Sales after differencing'].iloc[1:], lags=40)
    plt.show()


def splitting_df(dataframe):
    dataframe = dataframe.dropna()
    dataframe = dataframe.drop('STU', axis=1)
    # dataframe = dataframe.reset_index()
    train_set = dataframe.iloc[:87]
    test_set = dataframe.iloc[87:]
    print(train_set)
    print(test_set)
    return train_set, test_set


# -------- STARTING MAIN -----------
dataset = pd.read_excel(r"C:\Users\Enrico\Google Drive\DATA MINING\CHALLENGE FATER\serie_tamponi.xlsx")
df = dataset.copy()                     # Copy the original dataset in a new identical one

df_clean = preprocessing(df, 3)         # Choose the product number


stationarity_test(df_clean, differenced=False)                  # Stationarity test before differencing

df_clean = differencing(df_clean)
# df_clean = seasonal_differencing(df_clean)    # If data are seasonal




stationarity_test(df_clean.dropna(), differenced=True)          # Stationarity test after differencing

plotting_data(df_clean)         # Plotting the series after differencing


plotting_autocorr(df_clean)                 # Used for finding q in MA(q)
plotting_part_autocorr(df_clean)            # Used for finding p in AR(p)
# plot_pacf(df_clean)

training_set, test_set = splitting_df(df_clean)

model = ARIMA(training_set, order=(2,1,1))
#model = ARIMA(df_clean['Sales after differencing'].dropna(), order=(2,1,1))
model_fit = model.fit()

print(model_fit.summary())
df_clean['Forecast'] = model_fit.predict(start=80, typ='levels', dynamic=True)
df_clean[['Sales after differencing', 'Forecast']].plot(figsize=(12,8))
plt.show()


residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
plt.show()
residuals.plot(kind='kde')
plt.show()
print(residuals.describe())

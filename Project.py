import pandas as pd
import pmdarima as pm
import warnings
warnings.filterwarnings("ignore")


import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error


def df_filtered_product(dataframe, prod_num):
    """
    Filter all dataset for a certain product
    """
    df_prod = dataframe[dataframe['Products'] == 'Product ' + str(prod_num)]
    return df_prod


def preprocessing(df, product_number):
    """
    Cleaning data grouping values by product per week. Removing 'Estero' and creating the data structure for ARIMA model
    """
    useless_columns = ['Customers', 'Category', 'Segment', 'Regione', 'Provincia', 'Channel']
    df = df.drop(df[df.Provincia == '**'].index)             # Removing 'Estero'
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
    df_fin.Week = pd.to_datetime(df_fin.Week)
    df_fin.set_index('Week', inplace=True)
    return df_fin


def stationarity_test(df, differenced):
    """
    We are going to perform Augmented Dickey-Fuller Test.
    It works as an hypothesis test, with
    H0 = Series is stationary
    H1 = Series is not stationary
    result of ADF-Test are ['ADF Statistics', 'p-value', '#Lags Used', 'Number of observations used']
    """
    if differenced:

        result = adfuller(df['STU'])
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
            return True
        else:
            print("Weak evidence against the null-hypothesis, showing that the series is likely to be non-stationary!")
            return False


def plotting(dataframe, prod_num):
    """
    Plotting time series and differenced time series for each product
    """
    fig, axs = plt.subplots(2, sharex=True)
    axs[0].plot(dataframe['STU'])
    axs[1].plot(dataframe['STU'].diff().dropna())
    axs[0].set_title("Time Series of Product" + f"_{prod_num}")
    axs[1].set_title("Differenced Time Series of Product" + f"_{prod_num}")
    plt.savefig("Time Series of Product" + f"_{prod_num}" + ".pdf")


def plotting_autocorr(dataframe):
    """
    Plotting autocorrelation for data in order to find q for MA(q) at sight
    """
    plot_acf(dataframe['STU'].iloc[1:], lags=40)
    plt.show()


def plotting_part_autocorr(dataframe):
    """
    Plotting partial autocorrelation for data in order to find p for AR(p) at sight
    """
    plot_pacf(dataframe['STU'].iloc[1:], lags=40)
    plt.show()


def splitting_df(dataframe):
    """
    Splitting dataframe into training and test set for selecting the model
    """
    dataframe = dataframe.dropna()
    index = 100
    train_set = dataframe.iloc[:index]
    test_set = dataframe.iloc[index:]
    return train_set, test_set, dataframe


def rolling_forecast_ARIMA(train, test, p, d, q):
    """
    Rolling forecast ARIMA is a way to perform ARIMA forecast for (n+1) element given n elements of training set.
    After that, it takes one element from test set and add it to training set stepwise.
    """
    test = list(test['STU'])
    history = [x for x in train['STU']]
    predictions = []
    for t in range(len(test)):
        model = ARIMA(history, order=(p, d, q))
        model_fit = model.fit(disp=0)  # Avoid printing ARIMA stats
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
    error = mean_squared_error(test, predictions)
    return error


def forecast_ARIMA(train, test, p, d, q, model_selection, prod_num):
    """
    Make predictions with ARIMA model, and plotting both training set and forecast, or training set and test set in case of model selection
    """
    model = ARIMA(train['STU'].dropna(), order=(p, d, q))
    model_fit = model.fit(disp=0)
    if model_selection:
        forecast, st_errs, conf_int = model_fit.forecast(len(test), alpha=0.05)
    else:
        predictions_range = pd.date_range('2019-04-01', '2019-09-29', freq="W-MON").to_series()

        df_predictions = pd.DataFrame(predictions_range, columns=["Week"])

        forecast, st_errs, conf_int = model_fit.forecast(26, alpha=0.05)
        df_predictions['Sales_Prediction'] = forecast
        df_predictions['Week'] = pd.to_datetime(df_predictions["Week"])
        df_predictions.set_index("Week", inplace=True)

        plt.figure(figsize=(12, 5))
        plt.plot(train['STU'], label='training')
        plt.plot(df_predictions['Sales_Prediction'], label='forecast')
        plt.title('Forecast vs Actuals')
        plt.legend(loc='upper left', fontsize=8)
        plt.savefig("Product" + f"_{prod_num}" + ".pdf")
    if model_selection:
        MSE = mean_squared_error(test, forecast)
        return MSE
    return df_predictions


def ARIMA_model_selection(value):
    """
    For a more accurate forecast, auto_arima function within pmdarima is provided but not used for our porpouse.
    auto_arima function is able to select the best tuning parameters for ARIMA model, according to BIC and AIC.
    """
    model = pm.auto_arima(value, start_p=1, start_q=1,
                          test='adf',  # use adftest to find optimal 'd'
                          max_p=3, max_q=3,  # maximum p and q
                          m=1,  # frequency of series
                          d=None,  # let model determine 'd'
                          seasonal=False,  # No Seasonality
                          start_P=0,
                          D=0,
                          trace=True,
                          error_action='ignore',
                          suppress_warnings=True,
                          stepwise=True)
    print(model.summary())


def best_prediction(train, test, differenced, prod_num):
    """
    This function tests all possible parameters for ARIMA model for each product, and returns the tuning parameters that minimize MSE
    """
    MSE_target = 500000                 # Choosing a wide MSE in order to include all possible results
    best_MSE = MSE_target
    p_value = 0
    q_value = 0
    if differenced:
        d_value = 1
    else:
        d_value = 0
    for q in range(7):
        for p in range(7):
            # print(f"p_value: {p}, q_value: {q}")
            try:
                mod_sel = True
                MSE = forecast_ARIMA(train, test, p, d_value, q, mod_sel, prod_num)
                if MSE < best_MSE:
                    p_value = p
                    q_value = q
                    best_MSE = MSE
            except:
                pass
    print(f"Fitting an ARIMA model for Product n°{prod_num} with\n{p_value} as AR(p)\n{q_value} as MA(q)\n{d_value} as I(d)\nTest MSE = {best_MSE}")
    return p_value, q_value, d_value


# -------- STARTING MAIN -----------
dataset = pd.read_excel("serie_tamponi.xlsx")       # Loading data
df = dataset.copy()                                 # Copy the original dataset in a new identical one


for product in range(1, 23):                        # Iterate over all products
    df_clean = preprocessing(df, product)

    diff = False

    if not stationarity_test(df_clean, differenced=False):      # Stationarity test before differencing
        diff = True                                             # Time series must be stationary

    # plotting_autocorr(df_clean)                 # Used for finding q in MA(q)
    # plotting_part_autocorr(df_clean)            # Used for finding p in AR(p)

    training_set, test_set, df_clean = splitting_df(df_clean)
    plotting(df_clean, product)

    p, q, d = best_prediction(training_set, test_set, diff, product)

    try:
        predictions = forecast_ARIMA(df_clean, test_set, p, d, q, False, product)
        predictions.to_excel("Product" + f"_{product}" + ".xlsx")
    except:
        print("Unable to do forecast, some Error raised!")
        pass

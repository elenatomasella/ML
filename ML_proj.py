#Master's Degree in Data Science and Management
#Course: Machine Learning
#Students: Gian Lorenzo Marchioni () - David Paquette () - Elena Tomasella (781321)

# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import missingno as msno
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller
#split train and test
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA
import plotly.graph_objects as go
from pandas import DateOffset
from sklearn.metrics import mean_squared_error, confusion_matrix,mean_absolute_error
import math
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pandas.plotting import autocorrelation_plot
import plotly.graph_objs as go
import pandas_ta as ta
import yfinance as yf
import keras.models, keras.layers
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense


def volumes(dfs, dfs_names, nasdaq_df, amazon_df):
    # volumes
    fig, axs = plt.subplots(3, 2, figsize=(20, 10))
    for i in range(3):
        for j in range(2):
            plot_index = i * 2 + j
            dfs[plot_index].plot(x='Date', y='Volume', ax=axs[i, j], color='seagreen')
            axs[i, j].set_title(f'Volume for {dfs_names[plot_index]}')
            axs[i, j].set_xlabel('Date')
            axs[i, j].set_ylabel('Volume')
    plt.gca().set_facecolor('none')  # Transparent background
    plt.gca().patch.set_alpha(0.0)
    plt.tight_layout()
    # Specifica il nome del file e imposta il background trasparente
    plt.show()

    # adjust the volume outlier for nasdaq
    nasdaq_df.loc[nasdaq_df['Volume'] == nasdaq_df.Volume.max(), 'Volume'] = 11634422000
    # volume of nasdaq is not zero only from June 2006 on
    # volume of cac is not zero only from January 1998 on

    #volume for amzn
    # adjust the volume before 2022-06-06 split
    amazon_df.loc[amazon_df['Date'] <= pd.to_datetime('2022-06-06'), 'Volume'] = amazon_df.loc[amazon_df['Date'] <= pd.to_datetime('2022-06-06'), 'Volume'] * 20

    # assuming volumes were also incorrect for the periods where Open, High and Low were adjusted, we also adjust volumes
    amazon_df.loc[amazon_df['Date'] <= pd.to_datetime('1999-08-29'), 'Volume'] = amazon_df.loc[amazon_df['Date'] <= pd.to_datetime('1999-08-29'), 'Volume'] * 2
    amazon_df.loc[amazon_df['Date'] <= pd.to_datetime('1999-01-03'), 'Volume'] = amazon_df.loc[amazon_df['Date'] <= pd.to_datetime('1999-01-03'), 'Volume'] * 3
    amazon_df.loc[amazon_df['Date'] <= pd.to_datetime('1998-05-31'), 'Volume'] = amazon_df.loc[amazon_df['Date'] <= pd.to_datetime('1998-05-31'), 'Volume'] * 2

    # scaler
    scaler = MinMaxScaler()
    # standardization of the prices
    for df in dfs:
        df.loc[df['Volume'] > 0, ['Volume']] = scaler.fit_transform(df.loc[df['Volume'] > 0, ['Volume']])

def prices(dfs, dfs_names,amazon_df, ibm_df, nasdaq_df, cac_df, sp500_df):
    # prices: open, high, close, low BEFORE preprocessing
    fig, axs = plt.subplots(3, 2, figsize=(20, 10))
    for i in range(3):
        for j in range(2):
            plot_index = i * 2 + j
            dfs[plot_index].plot(x='Date', y='Open', ax=axs[i, j], color='purple')
            dfs[plot_index].plot(x='Date', y='High', ax=axs[i, j], color='green')
            dfs[plot_index].plot(x='Date', y='Low', ax=axs[i, j], color='red')
            dfs[plot_index].plot(x='Date', y='Close', ax=axs[i, j], color='blue')
            axs[i, j].set_title(f'{dfs_names[plot_index]}')
            axs[i, j].set_xlabel('Date')
    plt.tight_layout()
    plt.show()

    # replacing the value for the lowest (nonsense) value for Cac on 2022-09-18
    cac_df.loc[cac_df['Low'] == cac_df['Low'].min(), 'Low'] = 6052.42

    # fillna cac
    cac_df.loc[cac_df['Date'] == pd.to_datetime('2023-05-14'), 'Low'] = 7354.54
    cac_df.loc[cac_df['Date'] == pd.to_datetime('2023-05-14'), 'High'] = 7523.56
    cac_df.loc[cac_df['Date'] == pd.to_datetime('2023-05-14'), 'Close'] = 7491.96
    cac_df.loc[cac_df['Date'] == pd.to_datetime('2023-05-14'), 'Open'] = 7443.38

    cac_df.loc[cac_df['Date'] == pd.to_datetime('2023-05-21'), 'Low'] = 7197.40
    cac_df.loc[cac_df['Date'] == pd.to_datetime('2023-05-21'), 'High'] = 7493.38
    cac_df.loc[cac_df['Date'] == pd.to_datetime('2023-05-21'), 'Close'] = 7319.18
    cac_df.loc[cac_df['Date'] == pd.to_datetime('2023-05-21'), 'Open'] = 7467.93

    # fillna nasdaq
    nasdaq_df.loc[nasdaq_df['Date'] == pd.to_datetime('2023-05-14'), 'Low'] = 12174.06
    nasdaq_df.loc[nasdaq_df['Date'] == pd.to_datetime('2023-05-14'), 'High'] = 12364.65
    nasdaq_df.loc[nasdaq_df['Date'] == pd.to_datetime('2023-05-14'), 'Close'] = 12284.74
    nasdaq_df.loc[nasdaq_df['Date'] == pd.to_datetime('2023-05-14'), 'Open'] = 12231.68
    nasdaq_df.loc[nasdaq_df['Date'] == pd.to_datetime('2023-05-21'), 'Low'] = 12263.35
    nasdaq_df.loc[nasdaq_df['Date'] == pd.to_datetime('2023-05-21'), 'High'] = 12731.73
    nasdaq_df.loc[nasdaq_df['Date'] == pd.to_datetime('2023-05-21'), 'Close'] = 12657.90
    nasdaq_df.loc[nasdaq_df['Date'] == pd.to_datetime('2023-05-21'), 'Open'] = 12301.17

    # fillna sp500
    sp500_df.loc[sp500_df['Date'] == pd.to_datetime('2023-05-14'), 'Low'] = 4109.86
    sp500_df.loc[sp500_df['Date'] == pd.to_datetime('2023-05-14'), 'High'] = 4212.91
    sp500_df.loc[sp500_df['Date'] == pd.to_datetime('2023-05-14'), 'Close'] = 4191.98
    sp500_df.loc[sp500_df['Date'] == pd.to_datetime('2023-05-14'), 'Open'] = 4126.65
    sp500_df.loc[sp500_df['Date'] == pd.to_datetime('2023-05-21'), 'Low'] = 4103.98
    sp500_df.loc[sp500_df['Date'] == pd.to_datetime('2023-05-21'), 'High'] = 4212.87
    sp500_df.loc[sp500_df['Date'] == pd.to_datetime('2023-05-21'), 'Close'] = 4205.45
    sp500_df.loc[sp500_df['Date'] == pd.to_datetime('2023-05-21'), 'Open'] = 4190.78

    #adjust amazon
    # adjust all past prices for the 2022-06-06 split
    amazon_df.loc[amazon_df['Date'] <= pd.to_datetime('2022-06-06'), ['Open', 'High', 'Low', 'Close']] = amazon_df.loc[amazon_df['Date'] <= pd.to_datetime('2022-06-06'), ['Open','High','Low','Close']] / 20

    # adjust problematic section where low is higher than close
    amazon_df.loc[amazon_df['Date'] <= pd.to_datetime('1999-08-29'), ['Open', 'High', 'Low']] = amazon_df.loc[amazon_df['Date'] <= pd.to_datetime('1999-08-29'), ['Open', 'High', 'Low']] / 2
    amazon_df.loc[amazon_df['Date'] <= pd.to_datetime('1999-01-03'), ['Open', 'High', 'Low']] = amazon_df.loc[amazon_df['Date'] <= pd.to_datetime('1999-01-03'), ['Open', 'High', 'Low']] / 3
    amazon_df.loc[amazon_df['Date'] <= pd.to_datetime('1998-05-31'), ['Open', 'High', 'Low']] = amazon_df.loc[amazon_df['Date'] <= pd.to_datetime('1998-05-31'), ['Open', 'High', 'Low']] / 2

    #adjust ibm
    ibm_df_subset = ibm_df[ibm_df['Close'] < ibm_df['Low']]
    ibm_df_subset.loc[ibm_df_subset['Date'] <= pd.to_datetime('2021-11-04'), ['Open', 'High', 'Low']] = ibm_df_subset.loc[ibm_df_subset['Date'] <= pd.to_datetime('2021-11-04'), ['Open', 'High', 'Low']] / 1.046
    # adjust for the 1999-05-27 split, plus the Open and High at 1999-05-30, which are an anomaly in the dataset
    ibm_df_subset.loc[ibm_df_subset['Date'] <= pd.to_datetime('1999-05-27'), ['Open', 'High', 'Low']] =ibm_df_subset.loc[ibm_df_subset['Date'] <= pd.to_datetime('1999-05-27'), ['Open', 'High', 'Low']] / 2
    ibm_df_subset.loc[ibm_df_subset['Date'] == pd.to_datetime('1999-05-30'), ['Open', 'High']] = ibm_df_subset.loc[ibm_df_subset['Date'] == pd.to_datetime('1999-05-30'), ['Open','High']] / 2
    # adjust for the 1997-05-28 split, plus the Open and High at 1997-06-01, which are an anomaly in the dataset
    ibm_df_subset.loc[ibm_df_subset['Date'] <= pd.to_datetime('1997-05-28'), ['Open', 'High', 'Low']] =ibm_df_subset.loc[ibm_df_subset['Date'] <= pd.to_datetime('1997-05-28'), ['Open', 'High', 'Low']] / 2
    ibm_df_subset.loc[ibm_df_subset['Date'] == pd.to_datetime('1997-06-01'), ['Open', 'High']] = ibm_df_subset.loc[ibm_df_subset['Date'] == pd.to_datetime('1997-06-01'), ['Open','High']] / 2

    ibm_df.loc[ibm_df['Close'] < ibm_df['Low']] = ibm_df_subset


    # prices AFTER the cleaning of the Close prices
    fig, axs = plt.subplots(3, 2, figsize=(20, 10))
    for i in range(3):
        for j in range(2):
            plot_index = i * 2 + j
            dfs[plot_index].plot(x='Date', y='Open', ax=axs[i, j], color='purple')
            dfs[plot_index].plot(x='Date', y='High', ax=axs[i, j], color='green')
            dfs[plot_index].plot(x='Date', y='Low', ax=axs[i, j], color='red')
            dfs[plot_index].plot(x='Date', y='Close', ax=axs[i, j], color='blue')
            axs[i, j].set_title(f'{dfs_names[plot_index]}')
            axs[i, j].set_xlabel('Date')

    plt.tight_layout()
    plt.show()

    # scaler
    scaler = StandardScaler()
    # standardization of the prices
    for df in dfs:
        df[['Close', 'Open', 'Low', 'High']] = scaler.fit_transform(df[['Close', 'Open', 'Low', 'High']])

# Calculate RSI (Relative Strength Index)
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Calculate MACD (Moving Average Convergence Divergence)
def calculate_macd(data, short_window=12, long_window=26):
    short_ema = data['Close'].ewm(span=short_window, min_periods=1).mean()
    long_ema = data['Close'].ewm(span=long_window, min_periods=1).mean()
    macd = short_ema - long_ema
    signal_line = macd.ewm(span=9, min_periods=1).mean()
    return macd, signal_line

#Augmented Dickey-Fuller (ADF) test on each time series provided in the dfs list.
#The ADF test is a statistical test used to determine whether a unit root is present in a time series dataset.
#A unit root indicates that the series is non-stationary, meaning it has a trend or seasonality that needs to be removed for accurate analysis.
#  Define a function adf_check that takes a time series as input and performs the ADF test on it.
def adf_check(time_series):
    result = adfuller(time_series)
    print('Augmented Dickey-Fuller Test:')
    labels = ['ADF Test Statistic','p-value','#Number of Lags Used','Number of Observations Used']
    for value,label in zip(result,labels):
        print(label+' : '+str(value) )

    if result[1] <= 0.05:
        #Checks if the p-value from the ADF test is less than or equal to 0.05, which is a common significance level.
        #If so, it indicates evidence against the null hypothesis of a unit root being present.
        print(f"Reject the null hypothesis. Data has no unit root and is stationary.")
    else:
        #If the p-value is greater than 0.05, the null hypothesis cannot be rejected.
        print(f"Fail to reject the null hypothesis. Time series has a unit root, indicating it is non-stationary.")


def ARIMA_first(dfs, dfs_names, ibm_df, amazon_df, cac_df, sp500_df, microsoft_df, nasdaq_df):
    for df in dfs:
        # Calculate moving averages (e.g., 10-day and 50-day)
        df['MA10'] = df['Close'].rolling(window=10).mean()
        df['MA40'] = df['Close'].rolling(window=40).mean()
        df['MACD'], df['Signal_Line'] = calculate_macd(df)
        df['RSI'] = calculate_rsi(df)
    for df, name in zip(dfs, dfs_names):
        fig, (ax1, ax2, ax3) = plt.subplots(3)
        ax1.plot(df.Close);
        ax1.set_title(f'Original Series {name}');
        ax1.axes.xaxis.set_visible(False)
        # 1st Differencing
        ax2.plot(df.Close.diff());
        ax2.set_title(f'1st Order Differencing {name}');
        ax2.axes.xaxis.set_visible(False)
        # 2nd Differencing
        ax3.plot(df.Close.diff().diff());
        ax3.set_title(f'2nd Order Differencing {name}')
        plt.show()
        # --> to get standardized data, we use the log on the close prices and check it with adf_check
        # correzione esponenziale di ibm con il logaritmo
        df['Log Close'] = np.log(df['Close'] + 2)
        df['Log Close First Difference'] = df['Log Close'] - df['Log Close'].shift(1)
        adf_check(df['Log Close First Difference'].dropna())
        df['Log Close Second Difference'] = df['Log Close First Difference'] - df[
            'Log Close First Difference'].shift(1)
        adf_check(df['Log Close Second Difference'].dropna())
        df['Log Seasonal Difference'] = df['Log Close'] - df['Log Close'].shift(52)
        adf_check(df['Log Seasonal Difference'].dropna())
        df['Seasonal First Difference'] = df['Log Close First Difference'] - df[
            'Log Close First Difference'].shift(54)
        adf_check(df['Seasonal First Difference'].dropna())
        # selection of p and q
        # Plot PACF
        plt.style.use('default')
        plot_pacf(df['Log Close First Difference'].dropna().dropna(),
                  title=f'{name} - Log Close First Difference PACF', color='#008080')
        plt.grid(True, linestyle='--', linewidth=0.5, color='gray')
        plt.savefig(f'{name}-PACF.png', transparent=True)
        plt.show()
        # Plot ACF
        plt.style.use('default')
        plot_acf(df['Log Close First Difference'].dropna(),
                 title=f'{name} - Log Close First Difference ACF', color='#008080')
        plt.grid(True, linestyle='--', linewidth=0.5, color='gray')
        plt.savefig(f'{name}-ACF.png', transparent=True)
        plt.show()

    for df, name in zip(dfs,dfs_names):
        results_list = []
        for perc in [0.8,0.9]: #trying two different % for the test set as suggested
            train_size_df = int(len(df)*perc)
            df_train, df_test = df[:train_size_df], df[train_size_df:]
            df_test.set_index('Date', inplace=True)
            df_train.set_index('Date', inplace=True)

            model = ARIMA(df_train['Log Close'], order=(2, 2, 1)) #CUSTOMIZE THE ORDER IN THE FOR CYCLE!!
            results = model.fit()
            print(results.summary())

            # Get the residuals
            residuals = pd.DataFrame(results.resid)
            residuals.plot()  # suggesting that they still be some trend information not captured by the model?
            plt.title(f'{name} -lineplot of the residual errors with {perc} as trainset')
            plt.show()
            residuals.plot(kind='kde')
            plt.title(f'{name} - density plot of the residual error values with {perc} as trainset')
            print(residuals.describe())

            # forecast
            fc = results.forecast(len(df_test), alpha=0.05)
            df_test['forecast'] = fc
            plt.figure(figsize=(12, 5), dpi=100)
            plt.plot(df_train['Log Close'], color='green', label='training')
            plt.plot(df_test['Log Close'], color='blue', label=f'{name} - Actual Log Close Price with {perc} as trainset')
            plt.plot(df_test['forecast'], color='orange', label=f'{name}- Predicted Close Price with {perc} as trainset')
            plt.title(f'{name} Close Price Prediction vs Actual with {perc} as trainset')
            plt.xlabel('Date')
            plt.ylabel('Close Price')
            plt.legend(loc='upper left', fontsize=8)
            plt.show()

            fig = go.Figure()
            # Add traces for 'Predicted Close' and 'Close'
            fig.add_trace(go.Scatter(x=df_test.index, y=df_test['forecast'], mode='lines', name='Predicted Log Close',
                                     line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=df_test.index, y=df_test['Log Close'], mode='lines', name='Actual Log Close',
                                     line=dict(color='red')))
            # Update layout for better zooming
            fig.update_layout(
                title=f'Predicted vs Actual Log Close Prices for {name} with {perc} as trainset',
                xaxis=dict(
                    title='Date',
                    rangeselector=dict(
                        buttons=list([
                            dict(count=1, label='1m', step='month', stepmode='backward'),
                            dict(count=6, label='6m', step='month', stepmode='backward'),
                            dict(count=1, label='YTD', step='year', stepmode='todate'),
                            dict(count=1, label='1y', step='year', stepmode='backward'),
                            dict(step='all')
                        ])
                    ),
                    rangeslider=dict(
                        visible=True
                    ),
                    type='date'
                ),
                yaxis=dict(title='Close Price'),
            )

            fig.show()

            mse = mean_squared_error(df_test['Log Close'], fc)
            print(f'MSE for {name} with {perc} as trainset: ' + str(mse))
            mae = mean_squared_error(df_test['Log Close'], fc)
            print(f'MAE for {name} with {perc} as trainset: ' + str(mae))
            rmse = math.sqrt(mean_squared_error(df_test['Log Close'], fc))
            print(f'RMSE for {name} with {perc} as trainset: ' + str(rmse))
            mape = np.mean(np.abs(fc - df_test['Log Close']) / np.abs(df_test['Log Close']))
            print(f'MAPE for {name} with {perc} as trainset: ' + str(mape))

            results_list.append({'Modello': name, 'Percentuale di addestramento': perc,
                                 'MSE': mse, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape})

        results_df = pd.DataFrame(results_list)
        return f'comparison for {name}: ', results_df


def ARIMA_prediction(df):
        for perc in [0.9]: #we decided for 0.9
            train_size_df = int(len(df)*perc)
            df_train, df_test = df.iloc[:train_size_df,:], df.iloc[train_size_df:,:]
            df.Date = pd.to_datetime(df.Date)
            #df_test.set_index('Date', inplace=True)
            #df_train.set_index('Date', inplace=True)
            model = ARIMA(df_train['Close'], order=(2, 2, 1)) #CUSTOMIZE THE ORDER IN THE FOR CYCLE!!
            results = model.fit()
        return results.forecast(steps=len(df_test))


#hurst coefficient of the time series
def hurst(ts):

    n = len(ts)
    max_window = int(n / 2)
    window_sizes = range(10, max_window)
    R_S = np.zeros(len(window_sizes))

    for i, window_size in enumerate(window_sizes):
        window_ranges = np.arange(0, n, window_size)
        R, S = 0, 0
        for j in window_ranges:
            if j + window_size < n:
                # Calculate the range and standard deviation
                window = ts[j:j+window_size]
                R += np.max(window) - np.min(window)
                S += np.std(window)

        # Calculate the rescaled range
        R_S[i] = R / S

    # Fit a linear line to log-log plot
    fit = np.polyfit(np.log(window_sizes), np.log(R_S), 1)

    # Hurst exponent is the slope of the line
    hurst_exponent = fit[0]

    return hurst_exponent

def hurst_coeff(dataset):
    ts = dataset['Close']
    print("Hurst Exponent for Close:", hurst(ts))

    dataset['log return'] = np.log(dataset['Close'].shift(-1) / dataset['Close'])
    ts1 = dataset['log return']
    print("Hurst Exponent for Log return:", hurst(ts1))

    dataset['return'] = dataset['Close'].shift(-1) / dataset['Close']
    ts2 = dataset['return']
    print("Hurst Exponent for return:", hurst(ts2))

def first_LSTM_arch(dataset):
    def first_model(X_train, p_dropout=0.2):
        model = Sequential()
        model.add(
            LSTM(units=50,
                 return_sequences=True,
                 input_shape=(X_train.shape[1], 1))
        )
        model.add(Dropout(p_dropout))
        model.add(LSTM(units=50,
                       return_sequences=True))
        model.add(Dropout(p_dropout))

        model.add(Dense(units=1))
        return model

    splitlimit = int(len(dataset) * 0.8)
    dataset_train_set = dataset.loc[:splitlimit, :]
    #dataset_test_set = dataset.loc[splitlimit:, :]

    for window in [5,10,15]: #trying different length for the window
        X_train = []
        y_train = []
        df_days = dataset_train_set.shape[0]
        for x in range(window, df_days):
            X_train.append(dataset.loc[x - window:x, 'Close'])
            y_train.append(dataset.loc[x, 'Close'])

        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        X_test = []
        y_test = []
        for x in range(df_days, dataset.shape[0]):
            X_test.append(dataset.loc[x - window:x, 'Close'])
            y_test.append(dataset.loc[x, 'Close'])
        X_test, y_test = np.array(X_test), np.array(y_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        # training
        # definition
        model = first_model(X_train)
        display(model.summary())
        model.compile(optimizer='adam', loss='mean_squared_error')
        for epoch in [10,20,30,40,50]:
            history = model.fit(X_train, y_train, epochs=epoch, batch_size=30, validation_data=(X_test, y_test))
            y_pred = model.predict(X_test)
            # Plot the predicted values
            plt.style.use('default')
            plt.plot(y_test, label='True Values', color='#2F4F4F')
            plt.plot(y_pred[:, 0], label='Predicted Values', color='#FF6600')
            plt.grid(True, linestyle='--', linewidth=0.5, color='gray')
            plt.title(f'True vs Predicted Values with {epoch} epochs and {window} as window')
            plt.xlabel('Index')
            plt.ylabel('Close Price')
            plt.legend()
            plt.show()
            # Extract the MSE values from the history object
            mse_values = history.history['loss']
            validation_values = history.history['val_loss']
            # Plot the trend of MSE with respect to epochs
            plt.plot(range(1, len(mse_values) + 1), mse_values, label='MSE', color='#008080')
            plt.plot(range(1, len(validation_values) + 1), validation_values, label='Validation Test', color='#800080')
            plt.title(f'Mean Squared Error (MSE) Trend with {epoch} epochs and {window} as window')
            plt.xlabel('Epochs')
            plt.ylabel('MSE')
            plt.legend()
            plt.show()


def second_LSTM_arch(dataset): # #SECONDA ARCHITETTURA: inserisco anche gli indicatori in X_train input.
    def LSTM_second(X_train, p_dropout=0.2):
        model = Sequential()
        model.add(
            LSTM(units=50,
                 return_sequences=True,
                 input_shape=(X_train.shape[1], 5))
        )
        model.add(Dropout(p_dropout))
        model.add(LSTM(units=50,
                       return_sequences=True))
        model.add(Dropout(p_dropout))

        model.add(Dense(units=1))
        return model

    splitlimit = int(len(dataset) * 0.8)
    dataset_train_set = dataset.loc[:splitlimit, :]
    #dataset_test_set = dataset.loc[splitlimit:, :]

    for window in [5,10,15]: #trying different length for the window
        X_train = []
        y_train = []
        df_days = dataset_train_set.shape[0]
        for x in range(window, df_days):
            X_train.append(dataset.loc[x - window:x, ['Close', 'RSI', 'EMAF', 'EMAM', 'EMAS']].values)
            y_train.append(dataset.loc[x, 'Close'])

        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        X_test = []
        y_test = []
        for x in range(df_days, dataset.shape[0]):
            X_test.append(dataset.loc[x - window:x, ['Close', 'RSI', 'EMAF', 'EMAM', 'EMAS']].values)
            y_test.append(dataset.loc[x, 'Close'])
        X_test, y_test = np.array(X_test), np.array(y_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        # training
        # definition
        model = LSTM_second(X_train)
        display(model.summary())
        model.compile(optimizer='adam', loss='mean_squared_error')
        for epoch in [10,20,30,40,50]:
            history = model.fit(X_train, y_train, epochs=epoch, batch_size=30, validation_data=(X_test, y_test))
            y_pred = model.predict(X_test)
            # Plot the predicted values
            plt.style.use('default')
            plt.plot(y_test, label='True Values', color='#2F4F4F')
            plt.plot(y_pred[:, 0], label='Predicted Values', color='#FF6600')
            plt.grid(True, linestyle='--', linewidth=0.5, color='gray')
            plt.title(f'True vs Predicted Values with {epoch} epochs and {window} as window')
            plt.xlabel('Index')
            plt.ylabel('Close Price')
            plt.legend()
            plt.show()
            # Extract the MSE values from the history object
            mse_values = history.history['loss']
            validation_values = history.history['val_loss']
            # Plot the trend of MSE with respect to epochs
            plt.plot(range(1, len(mse_values) + 1), mse_values, label='MSE', color='#008080')
            plt.plot(range(1, len(validation_values) + 1), validation_values, label='Validation Test', color='#800080')
            plt.title(f'Mean Squared Error (MSE) Trend with {epoch} epochs and {window} as window')
            plt.xlabel('Epochs')
            plt.ylabel('MSE')
            plt.legend()
            plt.show()

def third_LSTM_arch(dataset): #QUARTA ARCHITETTURA: con indicatori,  e più layers.
    def LSTM_third(X_train, p_dropout=0.2):
        model = Sequential()
        model.add(
            LSTM(units=50,
                 return_sequences=True,
                 input_shape=(X_train.shape[1], 5))
        )
        model.add(Dropout(p_dropout))
        model.add(LSTM(units=50,
                       return_sequences=True))
        model.add(Dropout(p_dropout))

        model.add(Dropout(p_dropout))
        model.add(LSTM(units=50,
                       return_sequences=True))
        model.add(Dropout(p_dropout))

        model.add(Dropout(p_dropout))
        model.add(LSTM(units=50,
                       return_sequences=True))
        model.add(Dropout(p_dropout))

        model.add(Dropout(p_dropout))
        model.add(LSTM(units=50,
                       return_sequences=True))
        model.add(Dropout(p_dropout))

        model.add(Dense(units=1))
        return model

    dataset.dropna(inplace=True)
    splitlimit = int(len(dataset) * 0.8)
    dataset_train_set = dataset.loc[:splitlimit, :]
    # dataset_test_set = dataset.loc[splitlimit:, :]
    for window in [5, 10, 15]:  # trying different length for the window
        X_train = []
        y_train = []
        df_days = dataset_train_set.shape[0]
        for x in range(window, df_days):
            X_train.append(dataset.loc[x - window:x, ['Close', 'RSI', 'EMAF', 'EMAM', 'EMAS']].values)
            y_train.append(dataset.loc[x, 'Close'])

        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        X_test = []
        y_test = []
        for x in range(df_days, dataset.shape[0]):
            X_test.append(dataset.loc[x - window:x, ['Close', 'RSI', 'EMAF', 'EMAM', 'EMAS']].values)
            y_test.append(dataset.loc[x, 'Close'])
        X_test, y_test = np.array(X_test), np.array(y_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        # training
        # definition
        model = LSTM_third(X_train)
        display(model.summary())
        model.compile(optimizer='adam', loss='mean_squared_error')
        for epoch in [10, 20, 30, 40, 50]:
            history = model.fit(X_train, y_train, epochs=epoch, batch_size=30, validation_data=(X_test, y_test))
            y_pred = model.predict(X_test)
            # Plot the predicted values
            plt.style.use('default')
            plt.plot(y_test, label='True Values', color='#2F4F4F')
            plt.plot(y_pred[:, 0], label='Predicted Values', color='#FF6600')
            plt.grid(True, linestyle='--', linewidth=0.5, color='gray')
            plt.title(f'True vs Predicted Values with {epoch} epochs and {window} as window')
            plt.xlabel('Index')
            plt.ylabel('Close Price')
            plt.legend()
            plt.show()
            # Extract the MSE values from the history object
            mse_values = history.history['loss']
            validation_values = history.history['val_loss']
            # Plot the trend of MSE with respect to epochs
            plt.plot(range(1, len(mse_values) + 1), mse_values, label='MSE', color='#008080')
            plt.plot(range(1, len(validation_values) + 1), validation_values, label='Validation Test', color='#800080')
            plt.title(f'Mean Squared Error (MSE) Trend with {epoch} epochs and {window} as window')
            plt.xlabel('Epochs')
            plt.ylabel('MSE')
            plt.legend()
            plt.show()

def fifth_LSTM_arch(dataset):
    def LSTM_fifth(X_train, p_dropout=0.2):
        model = Sequential()
        model.add(
            LSTM(units=10,
                 return_sequences=True,
                 input_shape=(X_train.shape[1], 5))
        )
        model.add(Dropout(p_dropout))
        model.add(LSTM(units=10,
                       return_sequences=True))
        model.add(Dropout(p_dropout))

        model.add(Dense(units=1))
        return model

    dataset.dropna(inplace=True)
    splitlimit = int(len(dataset) * 0.8)
    dataset_train_set = dataset.loc[:dataset.index[0] + splitlimit, :]  ####iloc

    X_train = []
    y_train = []
    df_days = dataset_train_set.shape[0]

    for x in range(dataset.index[0] + 10, df_days):  # prendo intanto una finestra di 10 settimane
        X_train.append(dataset.loc[x - 10:x - 1, ['Close', 'RSI', 'EMAF', 'EMAM', 'EMAS']].values)
        y_train.append(dataset.loc[x, 'Close'])

    X_train, y_train = np.array(X_train), np.array(y_train)

    X_test = []
    y_test = []

    for x in range(dataset.index[0] + df_days, dataset.shape[0]):  # prendo intanto una finestra di 10 settimane
        X_test.append(dataset.loc[x - 10:x - 1, ['Close', 'RSI', 'EMAF', 'EMAM', 'EMAS']].values)
        y_test.append(dataset.loc[x, 'Close'])

    X_test, y_test = np.array(X_test), np.array(y_test)  # non serve convertirlo ancora y_test è già pronto

    # definition
    model = LSTM_fifth(X_train)
    model.build()
    display(model.summary())
    model.compile(optimizer='adam',
                  loss='mean_squared_error')

    # training
    history = model.fit(X_train, y_train, epochs=30, batch_size=10, validation_data=(X_test, y_test))
    y_pred = model.predict(X_test)

    # Plot the predicted values
    plt.plot(y_test, label='True Values')
    plt.plot(y_pred[:, 0], label='Predicted Values')

    plt.title('True vs Predicted Values')
    plt.xlabel('Index')
    plt.ylabel('Close Price')
    plt.legend()
    plt.show()

    # Extract the MSE values from the history object
    mse_values = history.history['loss']
    validation_values = history.history['val_loss']

    # Plot the trend of MSE with respect to epochs
    plt.plot(range(1, len(mse_values) + 1), mse_values, label='MSE')
    plt.plot(range(1, len(validation_values) + 1), validation_values, label='Validation Test')
    plt.title('Mean Squared Error (MSE) Trend')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()
    plt.show()

def LSTM_logreturn_arch(dataset):
    def LSTM_model(X_train, p_dropout=0.2):
        model = Sequential()
        model.add(
            LSTM(units=50,
                 return_sequences=True,
                 input_shape=(X_train.shape[1], 5))
        )
        model.add(Dropout(p_dropout))
        model.add(LSTM(units=50,
                       return_sequences=True))
        model.add(Dropout(p_dropout))

        model.add(Dense(units=1))
        return model

    dataset.dropna(inplace=True)
    splitlimit = int(len(dataset) * 0.8)
    dataset_train_set = dataset.loc[:dataset.index[0] + splitlimit, :]  ####iloc

    for window in [5,10,15]:
        X_train = []
        y_train = []
        df_days = dataset_train_set.shape[0]

        for x in range(dataset_train_set.index[0] + window,df_days):
            X_train.append(dataset.loc[x - window:x, ['log return', 'RSI', 'EMAF', 'EMAM', 'EMAS']].values)
            y_train.append(dataset.loc[x, 'log return'])

        X_train, y_train = np.array(X_train), np.array(y_train)

        X_test = []
        y_test = []

        for x in range(dataset_train_set.index[0] + df_days, dataset_train_set.index[0]+dataset.shape[0]):
            X_test.append(dataset.loc[x - window:x, ['log return', 'RSI', 'EMAF', 'EMAM', 'EMAS']].values)
            y_test.append(dataset.loc[x, 'log return'])

        X_test, y_test = np.array(X_test), np.array(y_test)  # non serve convertirlo ancora y_test è già pronto

        # definition
        model = LSTM_model(X_train)
        display(model.summary())
        model.compile(optimizer='adam',
                      loss='mean_squared_error')

        for epoch in [10,20,30]:
            # training
            history = model.fit(X_train, y_train, epochs=epoch, batch_size=10, validation_data=(X_test, y_test))
            y_pred = model.predict(X_test)

            # Plot the predicted values
            plt.plot(y_test, label='True Values')
            plt.plot(y_pred[:, 0], label='Predicted Values')

            plt.title(f'True vs Predicted Values with {epoch} epochs and {window} as window')
            plt.xlabel('Index')
            plt.ylabel('log return close')
            plt.legend()
            plt.show()

            # Extract the MSE values from the history object
            mse_values = history.history['loss']
            validation_values = history.history['val_loss']

            # Plot the trend of MSE with respect to epochs
            plt.plot(range(1, len(mse_values) + 1), mse_values, label='MSE')
            plt.plot(range(1, len(validation_values) + 1), validation_values, label='Validation Test')
            plt.title(f'Mean Squared Error (MSE) Trend {epoch} epochs and {window} as window')
            plt.xlabel('Epochs')
            plt.ylabel('MSE')
            plt.legend()
            plt.show()


def LSTM_first(dataset):
    dataset['RSI'] = ta.rsi(dataset.Close, length=10)
    dataset['EMAF'] = ta.ema(dataset.Close, length=20)
    dataset['EMAM'] = ta.ema(dataset.Close, length=100)
    dataset['EMAS'] = ta.ema(dataset.Close, length=150)
    first_LSTM_arch(dataset)
    #second_LSTM_arch(dataset)
    third_LSTM_arch(dataset)
    #fifth_LSTM_arch(dataset)


def LSTM_prediction(dataset):
    def LSTM_third(X_train, p_dropout=0.2):
        model = Sequential()
        model.add(
            LSTM(units=50,
                 return_sequences=True,
                 input_shape=(X_train.shape[1], 5))
        )
        model.add(Dropout(p_dropout))
        model.add(LSTM(units=50,
                       return_sequences=True))
        model.add(Dropout(p_dropout))

        model.add(Dropout(p_dropout))
        model.add(LSTM(units=50,
                       return_sequences=True))
        model.add(Dropout(p_dropout))

        model.add(Dropout(p_dropout))
        model.add(LSTM(units=50,
                       return_sequences=True))
        model.add(Dropout(p_dropout))

        model.add(Dropout(p_dropout))
        model.add(LSTM(units=50,
                       return_sequences=True))
        model.add(Dropout(p_dropout))

        model.add(Dense(units=1))
        return model
    splitlimit = int(len(dataset) * 0.9)
    dataset_train_set = dataset.loc[:dataset.index[0]+splitlimit, :]

    X_train = []
    y_train = []
    df_days = dataset_train_set.shape[0]
    for x in range(15, len(dataset_train_set)):
        X_train.append(dataset.iloc[x - 15:x, [4,6,7,8,9]].values)
        y_train.append(dataset.iloc[x,4])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 5))

    X_test = []
    y_test = []
    for x in range(dataset_train_set.index[-1], dataset.index[-1]+1):
        X_test.append(dataset.loc[x - 15:x-1, ['Close','RSI','EMAF','EMAM','EMAS']].values)
        y_test.append(dataset.loc[x, 'Close'])
    X_test, y_test = np.array(X_test), np.array(y_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 5))

    # training
    # definition
    model = LSTM_third(X_train)
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=10, batch_size=30, validation_data=(X_test, y_test))
    return model.predict(X_test)[:,0,0]

def next_week_market_behaviour(dataset):
    dataset['real good choice'] = [0 for x in list(range(dataset.index[0], dataset.index[-1]+1))]
    dataset['next_week_return'] = dataset.Close.shift(1) / dataset.Close - 1 #attenzione!
    for x in list(range(dataset.index[0], dataset.index[-1]+1)):
        if dataset.loc[x,'next_week_return']<-0.5:
            dataset.loc[x,'real good choice'] = -1
        elif dataset.loc[x,'next_week_return']>0.5:
            dataset.loc[x,'real good choice'] = 1


def AI_trading_NN(dataset,dataset_test): ##modello decisionale sulle previsioni
    dataset.dropna(inplace=True)
    dataset_test['pred Close'] = (ARIMA_prediction(dataset)+LSTM_prediction(dataset)[:len(ARIMA_prediction(dataset))+1])/2 #attenzione indici
    dataset_test['next_week_pred_return'] = dataset_test['pred Close'].shift(1) / dataset_test['pred Close'] - 1
    dataset_test['model choice'] = [0 for x in range(dataset_test.index[0], dataset_test.index[-1]+1)]
    for x in range(dataset_test.index[0], dataset_test.index[-1]+1):
        if dataset_test.loc[x, 'next_week_pred_return'] < -0.5:
            dataset_test.loc[x,'model choice'] = -1
        elif dataset_test.loc[x, 'next_week_pred_return'] > 0.5:
            dataset_test.loc[x,'model choice'] = 1

def AI_trading_NN(dataset,dataset_test):
    from scipy.optimize import minimize
    def costo_ottimizzato(pesi, ARIMA_pred, LSTM_pred, y_true):
        predizione_combinata = pesi[0] * ARIMA_pred + pesi[1] * LSTM_pred
        return np.mean((predizione_combinata - dataset_test.Close) ** 2)

    dataset.dropna(inplace=True)
    # Ottimizzazione dei pesi
    pesi_iniziali = [0.5, 0.5]  # Pesi iniziali
    risultato = minimize(costo_ottimizzato, pesi_iniziali, args=(
        ARIMA_prediction(dataset), LSTM_prediction(dataset)[:len(ARIMA_prediction(dataset)) + 1], dataset_test.Close),
                         method='Nelder-Mead')
    pesi_ottimali = risultato.x

    # Applicazione dei pesi ottimali
    ARIMA_pred = ARIMA_prediction(dataset)
    LSTM_pred = LSTM_prediction(dataset)[:len(ARIMA_pred) + 1]
    predizione_combinata = pesi_ottimali[0] * ARIMA_pred + pesi_ottimali[1] * LSTM_pred

    dataset_test['pred Close'] = predizione_combinata
    dataset_test['next_week_pred_return'] = dataset_test['pred Close'].shift(1) / dataset_test['pred Close'] - 1
    dataset_test['model choice'] = [0 for x in range(dataset_test.index[0], dataset_test.index[-1] + 1)]
    for x in range(dataset_test.index[0], dataset_test.index[-1] + 1):
        if dataset_test.loc[x, 'next_week_pred_return'] < -0.5:
            dataset_test.loc[x, 'model choice'] = -1
        elif dataset_test.loc[x, 'next_week_pred_return'] > 0.5:
            dataset_test.loc[x, 'model choice'] = 1


def __main__():

    #import datasets
    amazon_df = pd.read_csv('/Users/elenatomasella/Documents/LUISS/machine learning/euklid/Weekly series/Amazon_weekly.csv', sep=',', header=0)
    cac_df = pd.read_csv('/Users/elenatomasella/Documents/LUISS/machine learning/euklid/Weekly series/Cac_weekly.csv', sep=',', header=0)
    ibm_df = pd.read_csv('/Users/elenatomasella/Documents/LUISS/machine learning/euklid/Weekly series/IBM_weekly.csv', sep=',', header=0)
    microsoft_df = pd.read_csv('/Users/elenatomasella/Documents/LUISS/machine learning/euklid/Weekly series/Microsoft_weekly.csv', sep=',', header=0)
    nasdaq_df = pd.read_csv('/Users/elenatomasella/Documents/LUISS/machine learning/euklid/Weekly series/Nasdaq_weekly.csv', sep=',', header=0)
    sp500_df = pd.read_csv('/Users/elenatomasella/Documents/LUISS/machine learning/euklid/Weekly series/Sp500_weekly.csv', sep=',', header=0)
    dfs = [amazon_df, cac_df, ibm_df, microsoft_df, nasdaq_df, sp500_df]
    dfs_names = ['Amazon', 'Cac', 'IBM', 'Microsoft', 'Nasdaq', 'sp500']


    # conversion of Date values into datetime object
    for dataset in dfs:
        dataset['Date'] = pd.to_datetime(dataset['Date'], format='%Y-%m-%d')
        dataset['RSI'] = ta.rsi(dataset.Close, length=10)
        dataset['EMAF'] = ta.ema(dataset.Close, length=20)
        dataset['EMAM'] = ta.ema(dataset.Close, length=100)
        dataset['EMAS'] = ta.ema(dataset.Close, length=150)
        #dataset['log return'] = np.log(dataset['Close'].shift(-1) / dataset['Close'])

    volumes(dfs, dfs_names, nasdaq_df, amazon_df)
    prices(dfs, dfs_names, amazon_df, ibm_df, nasdaq_df, cac_df, sp500_df)

    #ARIMA_first(dfs,dfs_names, ibm_df, amazon_df, cac_df, sp500_df, microsoft_df, nasdaq_df)
    #consideration: as we can see, taking a validation set of the 0.9 has increased significantly
    #the accuracy of the model
    #for dataset in dfs:
        #LSTM_first(dataset)
        #LSTM_logreturn_arch(dataset)

    for dataset in dfs:
        next_week_market_behaviour(dataset)

    amazon_df_test = amazon_df[int(0.9 * amazon_df.shape[0]):]
    cac_df_test = cac_df[int(0.9 * cac_df.shape[0]):]
    ibm_df_test =ibm_df[int(0.9 * ibm_df.shape[0]):]
    microsoft_df_test = microsoft_df[int(0.9 * microsoft_df.shape[0]):]
    nasdaq_df_test = nasdaq_df[int(0.9 * nasdaq_df.shape[0]):]
    sp500_df_test = sp500_df[int(0.9 * sp500_df.shape[0]):]


    #amazon
    AI_trading_NN(amazon_df, amazon_df_test)
    print('Amazon: \n',confusion_matrix(amazon_df_test['real good choice'], amazon_df_test['model choice'], labels=[1, 0, -1]))

    # cac
    AI_trading_NN(cac_df, cac_df_test)
    print('cac: \n',
          confusion_matrix(cac_df_test['real good choice'], cac_df_test['model choice'], labels=[1, 0, -1]))

    # ibm
    AI_trading_NN(ibm_df, ibm_df_test)
    print('ibm: \n',
          confusion_matrix(ibm_df_test['real good choice'], ibm_df_test['model choice'], labels=[1, 0, -1]))

    # microsoft
    AI_trading_NN(microsoft_df, microsoft_df_test)
    print('microsoft: \n',
          confusion_matrix(microsoft_df_test['real good choice'], microsoft_df_test['model choice'], labels=[1, 0, -1]))

    #nasdaq
    AI_trading_NN(nasdaq_df, nasdaq_df_test)
    print('nasdaq: \n',
          confusion_matrix(nasdaq_df_test['real good choice'], nasdaq_df_test['model choice'], labels=[1, 0, -1]))

    # sp500
    AI_trading_NN(sp500_df, sp500_df_test)
    print('sp500: \n',
          confusion_matrix(sp500_df_test['real good choice'], sp500_df_test['model choice'], labels=[1, 0, -1]))


def main():
    __main__()

main()

#codice di brutta copia per cose che ho inserito nei cicli for
def drafts():
    # arima
    ##microsoft ARIMA


    train_size_microsoft = int(len(microsoft_df) * 0.8)
    microsoft_train, microsoft_test = microsoft_df[:train_size_microsoft], microsoft_df[train_size_microsoft:]

    microsoft_test.set_index('Date', inplace=True)
    microsoft_train.set_index('Date', inplace=True)
    # MICROSOFT
    model = ARIMA(microsoft_train['Log Close'], order=(1, 2, 1))  # ORDERS: TO BE DECIDED ON THE ANALYSIS ABOVE.
    ##add the seasonal orders too!!

    results = model.fit()
    print(results.summary())
    # Get the residuals
    residuals = pd.DataFrame(results.resid)
    residuals.plot()
    plt.title('lineplot of the residual errors')
    # suggesting that they still be some trend information not captured by the model
    plt.show()
    residuals.plot(kind='kde')
    plt.title('density plotof the residual error values')
    print(residuals.describe())

    # forecast
    fc = results.forecast(314, alpha=0.05)
    microsoft_test['forecast'] = fc
    plt.figure(figsize=(12, 5), dpi=100)
    plt.style.use('default')

    plt.plot(microsoft_train['Log Close'], color='#3498db', label='training')
    plt.plot(microsoft_test['Log Close'], color='#2ecc71', label='Actual Log Close Price')
    plt.plot(microsoft_test['forecast'], color='#e74c3c', label='Predicted Log Close Price')
    plt.title('Microsoft Close Price Actual vs Prediction')
    plt.grid(True, linestyle='--', linewidth=0.5, color='gray')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend(loc='upper left', fontsize=8)
    plt.savefig('microsoft pred arima.png')
    plt.show()

    fig = go.Figure()

    # Add traces for 'Predicted Close' and 'Close'
    fig.add_trace(
        go.Scatter(x=microsoft_test.index, y=microsoft_test['forecast'], mode='lines', name='Predicted Log Close',
                   line=dict(color='blue')))
    fig.add_trace(
        go.Scatter(x=microsoft_test.index, y=microsoft_test['Log Close'], mode='lines', name='Actual Log Close',
                   line=dict(color='red')))

    # Update layout for better zooming
    fig.update_layout(
        title='Predicted vs Actual Log Close Prices for Microsoft',
        xaxis=dict(
            title='Date',
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label='1m', step='month', stepmode='backward'),
                    dict(count=6, label='6m', step='month', stepmode='backward'),
                    dict(count=1, label='YTD', step='year', stepmode='todate'),
                    dict(count=1, label='1y', step='year', stepmode='backward'),
                    dict(step='all')
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type='date'
        ),
        yaxis=dict(title='Close Price'),
    )

    fig.show()

    mse = mean_squared_error(microsoft_test['Log Close'], fc)
    print('MSE: ' + str(mse))
    mae = mean_squared_error(microsoft_test['Log Close'], fc)
    print('MAE: ' + str(mae))
    rmse = math.sqrt(mean_squared_error(microsoft_test['Log Close'], fc))
    print('RMSE: ' + str(rmse))
    mape = np.mean(np.abs(fc - microsoft_test['Log Close']) / np.abs(microsoft_test['Log Close']))
    print('MAPE: ' + str(mape))

    #amazon

    train_size_amazon = int(len(amazon_df) * 0.8)  # 80% for training
    amazon_train, amazon_test = amazon_df[:train_size_amazon], amazon_df[train_size_amazon:]
    amazon_test.set_index('Date', inplace=True)
    amazon_train.set_index('Date', inplace=True)

    # AMAZON -->> problemi!
    model = ARIMA(amazon_train['Log Close'], order=(3, 3, 1))  # ORDERS: TO BE DECIDED ON THE ANALYSIS ABOVE.
    ##add the seasonal orders too!!

    results = model.fit()
    print(results.summary())

    # Get the residuals
    residuals = pd.DataFrame(results.resid)
    residuals.plot()  # suggesting that they still be some trend information not captured by the model?
    plt.title('lineplot of the residual errors')
    plt.show()
    residuals.plot(kind='kde')
    plt.title('density plotof the residual error values')
    print(residuals.describe())

    # forecast
    fc = results.forecast(280, alpha=0.05)
    amazon_test['forecast'] = fc
    plt.figure(figsize=(12, 5), dpi=100)
    plt.plot(amazon_train['Log Close'], color='green', label='training')
    plt.plot(amazon_test['Log Close'], color='blue', label='Actual Log Close Price')
    plt.plot(amazon_test['forecast'], color='orange', label='Predicted Close Price')
    plt.title('Amazon Close Price Prediction vs Actual')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend(loc='upper left', fontsize=8)
    plt.show()

    fig = go.Figure()

    # Add traces for 'Predicted Close' and 'Close'
    fig.add_trace(go.Scatter(x=amazon_test.index, y=amazon_test['forecast'], mode='lines', name='Predicted Log Close',
                             line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=amazon_test.index, y=amazon_test['Log Close'], mode='lines', name='Actual Log Close',
                             line=dict(color='red')))

    # Update layout for better zooming
    fig.update_layout(
        title='Predicted vs Actual Log Close Prices for Amazon',
        xaxis=dict(
            title='Date',
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label='1m', step='month', stepmode='backward'),
                    dict(count=6, label='6m', step='month', stepmode='backward'),
                    dict(count=1, label='YTD', step='year', stepmode='todate'),
                    dict(count=1, label='1y', step='year', stepmode='backward'),
                    dict(step='all')
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type='date'
        ),
        yaxis=dict(title='Close Price'),
    )

    fig.show()

    mse = mean_squared_error(amazon_test['Log Close'], fc)
    print('MSE: ' + str(mse))
    mae = mean_squared_error(amazon_test['Log Close'], fc)
    print('MAE: ' + str(mae))
    rmse = math.sqrt(mean_squared_error(amazon_test['Log Close'], fc))
    print('RMSE: ' + str(rmse))
    mape = np.mean(np.abs(fc - amazon_test['Log Close']) / np.abs(amazon_test['Log Close']))
    print('MAPE: ' + str(mape))

    #nasdaq
    # selection of d parameter for Nasdaq
    # as the Close are not stationary, we should consider d such as they would become so.
    # Original Series

    train_size_nasdaq = int(len(sp500_df) * 0.8)  # 80% for training
    nasdaq_train, nasdaq_test = nasdaq_df[:train_size_nasdaq], nasdaq_df[train_size_nasdaq:]
    nasdaq_test.set_index('Date', inplace=True)
    nasdaq_train.set_index('Date', inplace=True)
    # nasdaq
    model = ARIMA(nasdaq_train['Close'], order=(2, 2, 1))  # ORDERS: TO BE DECIDED ON THE ANALYSIS ABOVE.
    ##add the seasonal orders too!!

    results = model.fit()
    print(results.summary())

    nasdaq_df['Predicted Close'] = results.predict(start=nasdaq_train.index[0], end=nasdaq_train.index[-1],
                                                   typ='levels')

    # Get the residuals
    residuals = pd.DataFrame(results.resid)
    residuals.plot()  # lineplot of the residual errors, suggesting that they still be some trend information not captured by the model?
    plt.show()
    residuals.plot(kind='kde')  # density plotof the residual error values
    print(residuals.describe())
    fc = results.forecast(314, alpha=0.05)
    nasdaq_test['forecast'] = fc
    plt.figure(figsize=(12, 5), dpi=100)
    plt.plot(nasdaq_train['Log Close'], color='green', label='training')
    plt.plot(nasdaq_test['Log Close'], color='blue', label='Actual Log Close Price')
    plt.plot(nasdaq_test['forecast'], color='orange', label='Predicted Close Price')
    plt.title('HCL Close Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Actual Close Price')
    plt.legend(loc='upper left', fontsize=8)
    plt.show()
    # Get the residuals
    residuals = pd.DataFrame(results.resid)
    residuals.plot()
    print(residuals.describe())
    nasdaq_test['Residual'] = nasdaq_test['Log Close'].iloc[2:] - nasdaq_test['Log Close'].iloc[2:]

    fig = go.Figure()

    # Add traces for 'Predicted Close' and 'Close'
    fig.add_trace(go.Scatter(x=nasdaq_test.index, y=nasdaq_test['forecast'], mode='lines', name='Predicted Log Close',
                             line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=nasdaq_test.index, y=nasdaq_test['Log Close'], mode='lines', name='Actual Log Close',
                             line=dict(color='red')))

    # Update layout for better zooming
    fig.update_layout(
        title='Predicted vs Actual Log Close Prices for nasdaq',
        xaxis=dict(
            title='Date',
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label='1m', step='month', stepmode='backward'),
                    dict(count=6, label='6m', step='month', stepmode='backward'),
                    dict(count=1, label='YTD', step='year', stepmode='todate'),
                    dict(count=1, label='1y', step='year', stepmode='backward'),
                    dict(step='all')
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type='date'
        ),
        yaxis=dict(title='Close Price'),
    )

    fig.show()

    mse = mean_squared_error(nasdaq_test['Log Close'], fc)
    print('MSE: ' + str(mse))
    mae = mean_squared_error(nasdaq_test['Log Close'], fc)
    print('MAE: ' + str(mae))
    rmse = math.sqrt(mean_squared_error(nasdaq_test['Log Close'], fc))
    print('RMSE: ' + str(rmse))
    mape = np.mean(np.abs(fc - nasdaq_test['Log Close']) / np.abs(nasdaq_test['Log Close']))
    print('MAPE: ' + str(mape))


    # sp500

    train_size_sp500 = int(len(sp500_df) * 0.8)  # 80% for training
    sp500_train, sp500_test = sp500_df[:train_size_sp500], sp500_df[train_size_sp500:]
    sp500_test.set_index('Date', inplace=True)
    sp500_train.set_index('Date', inplace=True)
    model = ARIMA(sp500_train['Close'], order=(4, 1, 1))  # ORDERS: TO BE DECIDED ON THE ANALYSIS ABOVE.
    ##add the seasonal orders too!!

    results = model.fit()
    print(results.summary())

    # Get the residuals
    residuals = pd.DataFrame(results.resid)
    residuals.plot()  # lineplot of the residual errors, suggesting that they still be some trend information not captured by the model?
    plt.show()
    residuals.plot(kind='kde')  # density plotof the residual error values
    print(residuals.describe())

    # forecast
    fc = results.forecast(314, alpha=0.05)
    sp500_test['forecast'] = fc
    plt.figure(figsize=(12, 5), dpi=100)
    plt.plot(sp500_train['Log Close'], color='green', label='training')
    plt.plot(sp500_test['Log Close'], color='blue', label='Actual Log Close Price')
    plt.plot(sp500_test['forecast'], color='orange', label='Predicted Close Price')
    plt.title('sp500 Close Price Prediction vs Actual')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend(loc='upper left', fontsize=8)
    plt.show()

    fig = go.Figure()

    # Add traces for 'Predicted Close' and 'Close'
    fig.add_trace(go.Scatter(x=sp500_test.index, y=sp500_test['forecast'], mode='lines', name='Predicted Log Close',
                             line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=sp500_test.index, y=sp500_test['Log Close'], mode='lines', name='Actual Log Close',
                             line=dict(color='red')))

    # Update layout for better zooming
    fig.update_layout(
        title='Predicted vs Actual Log Close Prices for sp500',
        xaxis=dict(
            title='Date',
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label='1m', step='month', stepmode='backward'),
                    dict(count=6, label='6m', step='month', stepmode='backward'),
                    dict(count=1, label='YTD', step='year', stepmode='todate'),
                    dict(count=1, label='1y', step='year', stepmode='backward'),
                    dict(step='all')
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type='date'
        ),
        yaxis=dict(title='Close Price'),
    )

    fig.show()

    mse = mean_squared_error(sp500_test['Log Close'], fc)
    print('MSE: ' + str(mse))
    mae = mean_squared_error(sp500_test['Log Close'], fc)
    print('MAE: ' + str(mae))
    rmse = math.sqrt(mean_squared_error(sp500_test['Log Close'], fc))
    print('RMSE: ' + str(rmse))
    mape = np.mean(np.abs(fc - sp500_test['Log Close']) / np.abs(sp500_test['Log Close']))
    print('MAPE: ' + str(mape))

    #cac
    train_size_cac = int(len(cac_df) * 0.8)  # 80% for training
    cac_train, cac_test = cac_df[:train_size_cac], cac_df[train_size_cac:]
    cac_test.set_index('Date', inplace=True)
    cac_train.set_index('Date', inplace=True)

    model = ARIMA(cac_train['Close'], order=(1, 1, 2))  # ORDERS: TO BE DECIDED ON THE ANALYSIS ABOVE.
    ##add the seasonal orders too!!

    results = model.fit()
    print(results.summary())

    # Get the residuals
    residuals = pd.DataFrame(results.resid)
    residuals.plot()  # lineplot of the residual errors, suggesting that they still be some trend information not captured by the model?
    plt.show()
    residuals.plot(kind='kde')  # density plotof the residual error values
    print(residuals.describe())
    fc = results.forecast(314, alpha=0.05)
    cac_test['forecast'] = fc
    plt.figure(figsize=(12, 5), dpi=100)
    plt.plot(cac_train['Log Close'], color='green', label='training')
    plt.plot(cac_test['Log Close'], color='blue', label='Actual Log Close Price')
    plt.plot(cac_test['forecast'], color='orange', label='Predicted Close Price')
    plt.title('cac Close Price Prediction vs Actual')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend(loc='upper left', fontsize=8)
    plt.show()

    fig = go.Figure()

    # Add traces for 'Predicted Close' and 'Close'
    fig.add_trace(go.Scatter(x=cac_test.index, y=cac_test['forecast'], mode='lines', name='Predicted Log Close',
                             line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=cac_test.index, y=cac_test['Log Close'], mode='lines', name='Actual Log Close',
                             line=dict(color='red')))

    # Update layout for better zooming
    fig.update_layout(
        title='Predicted vs Actual Log Close Prices for Cac',
        xaxis=dict(
            title='Date',
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label='1m', step='month', stepmode='backward'),
                    dict(count=6, label='6m', step='month', stepmode='backward'),
                    dict(count=1, label='YTD', step='year', stepmode='todate'),
                    dict(count=1, label='1y', step='year', stepmode='backward'),
                    dict(step='all')
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type='date'
        ),
        yaxis=dict(title='Close Price'),
    )

    fig.show()

    mse = mean_squared_error(cac_test['Log Close'], fc)
    print('MSE: ' + str(mse))
    mae = mean_squared_error(cac_test['Log Close'], fc)
    print('MAE: ' + str(mae))
    rmse = math.sqrt(mean_squared_error(cac_test['Log Close'], fc))
    print('RMSE: ' + str(rmse))
    mape = np.mean(np.abs(fc - cac_test['Log Close']) / np.abs(cac_test['Log Close']))
    print('MAPE: ' + str(mape))



    #parte di first_LSTM

    # forecast
    dataset_test_set.loc[2:,'forecast'] = y_pred
    plt.figure(figsize=(12, 5), dpi=100)
    plt.plot(dataset_train_set['Log Close'], color='green', label='training')
    plt.plot(dataset_test_set['Log Close'], color='blue', label='Actual Log Close Price')
    plt.plot(dataset_test_set['forecast'], color='orange', label='Predicted Close Price')
    plt.title(f'{name} Close Price Prediction vs Actual')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend(loc='upper left', fontsize=8)
    plt.show()


    #altra parte del primo LSTM

    # un'altra prova, aumentando il numero di epoche ma mantenendo lo stesso modello
    # training
    history = model.fit(X_train, y_train, epochs=50, batch_size=30, validation_data=(X_test, y_test))
    y_pred = model.predict(X_test)

    # Plot the predicted values
    plt.style.use('default')
    plt.plot(y_test, label='True Values', color='#2F4F4F')
    plt.plot(y_pred[:, 0], label='Predicted Values', color='#FF6600')
    plt.grid(True, linestyle='--', linewidth=0.5, color='gray')
    plt.title('True vs Predicted Values -more epohcs')
    plt.xlabel('Index')
    plt.ylabel('Close Price')
    plt.legend()
    plt.show()

    # Extract the MSE values from the history object
    mse_values = history.history['loss']
    validation_values = history.history['val_loss']

    # Plot the trend of MSE with respect to epochs
    plt.plot(range(1, len(mse_values) + 1), mse_values, label='MSE')
    plt.plot(range(1, len(validation_values) + 1), validation_values, label='Validation Test')
    plt.title('Mean Squared Error (MSE) Trend')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()
    plt.show()

    # stesso modello, aumentando la finestra temporale a 10 settimane (50 epoche)
    X_train = []
    y_train = []
    df_days = dataset_train_set.shape[0]

    for x in range(10, df_days):  # prendo finestra di 10 settimane
        X_train.append(dataset.loc[x - 10:x, 'Close']) #it was x-1 for colab but i converted it here
        y_train.append(dataset.loc[x, 'Close'])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    X_test = []
    y_test = []
    for x in range(df_days, dataset.shape[0]):  # prendo intanto una finestra di 5 settimane
        X_test.append(dataset.loc[x - 10:x, 'Close'])  # anche qui ho tolto il -1
        y_test.append(dataset.loc[x, 'Close'])
    X_test, y_test = np.array(X_test), np.array(y_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


    # definition
    model = first_model(X_train)
    model.build()
    display(model.summary())
    model.compile(optimizer='adam',
                  loss='mean_squared_error')

    # training
    history = model.fit(X_train, y_train, epochs=50, batch_size=30, validation_data=(X_test, y_test))
    y_pred = model.predict(X_test)

    # Plot the predicted values
    plt.plot(dataset['Close'], label='True Values')
    plt.plot(y_pred[:, 0], label='Predicted Values')

    plt.title('True vs Predicted Values')
    plt.xlabel('Index')
    plt.ylabel('Close Price')
    plt.legend()
    plt.show()

    # Extract the MSE values from the history object
    mse_values = history.history['loss']
    validation_values = history.history['val_loss']

    # Plot the trend of MSE with respect to epochs
    plt.plot(range(1, len(mse_values) + 1), mse_values, label='MSE')
    plt.plot(range(1, len(validation_values) + 1), validation_values, label='Validation Test')
    plt.title('Mean Squared Error (MSE) Trend')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()
    plt.show()
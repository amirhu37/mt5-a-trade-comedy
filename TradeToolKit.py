"""
helo, Dear User..
I'm 'Trade Comedy' , I was Born as Money making bot, but now thigs have changed. 
todat I am a Colection of helping function and classes that can help 'OTHERS' to make a Money Making bot
here ,inside me, there are a candle-pattern class witch can find engulfing, doji and three's. 
I can help you find Supports and Resistance lines, Pivot Points, trend line.
I'm still a Kid, if you You want to help to grow see me at 'https://github.com/amirhu37/mt5-a-trade-comedy'

"""

from typing import List
import MetaTrader5 as mt5
import numpy as np
from sklearn import linear_model as ln
from datetime import datetime
from os import system
from os.path import exists

__name__ = "mtkit"

assert mt5.initialize() == True , "MetaTRader can't Run, check installation or network Connection"
ordr_dict = {'buy': 0, 'sell': 1}

time_perid = {
    "1m":      mt5.TIMEFRAME_M1,
    '3m':      mt5.TIMEFRAME_M3,
    '5m':      mt5.TIMEFRAME_M5,
    '15m':     mt5.TIMEFRAME_M15,
    '30m':     mt5.TIMEFRAME_M30,
    '1h':     mt5.TIMEFRAME_H1,
    '4h':    mt5.TIMEFRAME_H4,
    '1d':    mt5.TIMEFRAME_D1,
    '1w':  mt5.TIMEFRAME_W1, }


def Symbol_data(sym: str, time_frame: str, bar_range: int, method: str, Open_candle: bool = False):
    """
    excract SYMBOL datas such as:
    ----
    method: 't' or 'time' , 'o' or'open' , 'h' or 'high', 'l' or 'low', 'c' or 'close', 'v' or 'volume'

    open_candle : means should include current open candle or not
    """
    o = 0 if Open_candle == True else 1
    # if Open_candle == True : o = 0
    bars = mt5.copy_rates_from_pos(sym, time_perid[time_frame], o, bar_range)
    OHLC = {'t': 0, 'o': 1, 'h': 2, 'l': 3, 'c': 4, 'v': 5,
            'time': 0, 'open': 1, 'high': 2, 'low': 3, 'close': 4, 'volume': 5}
    temp_list = [i for i in bars]

    if method == 'all':

        data = [temp_list[i] for i in range(len(temp_list))]
    elif method in OHLC:
        data = [temp_list[i][OHLC[method]] for i in range(len(temp_list))]

    bars = np.array(data)
    return bars


def Symbol_info(s: str):
    """
    SYMBOLs can Have difrent pips and spreads.
    maybe I can help you with that.\nBTW rr is Risk/Reward Ratio
    -
    return: spread, point, r/r ratio
    """
    DATA: dict = mt5.symbol_info(s)._asdict()

    spread = DATA['spread'] 
    pip = DATA['point']


    return spread * pip , pip


class MovingAverages:
    """
        I can help you to Find Simple Moving Average, base on this methods:
        ----

        method: 'open' ,'high', 'low', 'close', 'volume'
        ----

        time_Frame: 1m, 5m, 15m, 30m, 1h, 4h
    """

    def __init__(self, symbol: str, time_frame: str,  window_size: int, ohlc: str):
        self.window = window_size
        self.ohlc = ohlc
        self.time_frame = time_frame
        self.symbol = symbol
        ohlc_dict = {"open": 1, 'high': 2, "low": 3, "close": 4, 'volume': 5}
        self.bars = Symbol_data(
            self.symbol, self.time_frame, self.window, self.ohlc)

    def Simple(self,):
        data = self.bars
        weights = np.repeat(1.0, self.window) / self.window
        return np.convolve(data, weights, 'valid')

    def Exponintial(self, alpha=0.5):
        data = self.bars
        weights = np.exp(np.linspace(-1., 0., self.window))
        weights /= weights.sum()
        a = np.convolve(data, weights, mode='full')[:len(data)]
        b = np.ones_like(data)
        b[:self.window] = data[:self.window]
        return a * alpha + b * (1 - alpha)

    @property
    def sma(self):
        return self.Simple()

    @property
    def ema(self):
        return self.Exponintial()


class Trend:
    def __init__(self,  symbol: str, time_frame: str, ma: int = 30) -> None:
        self.symbol = symbol
        self.time_frame = time_frame
        self.ma = ma

    def trend_fit(self,) -> float:
        "Nothing is more Important than Trend Line.\nI used Linear Regression for Find it"

        bars = mt5.copy_rates_from_pos(
            self.symbol, time_perid[self.time_frame], 1, self.ma)
        temp_list = [i for i in bars]
        highs = [temp_list[i][2] for i in range(len(temp_list))]
        lows = [temp_list[i][3] for i in range(len(temp_list))]

        self.high_1: float = max(highs)
        self.low_1: float = max(lows)

        self.high_2: float = min(highs)
        self.low_2: float = min(lows)

        highs = np.array(highs)
        lows = np.array(lows)

        end = (highs.shape[0])+1

        q = np.arange(1, end)

        q = q.reshape(-1, 1)

        reg = ln.LinearRegression()

        self.ln_closse = reg.fit(q, highs)
        self.ln_lows = reg.fit(q, lows)

        if round(self.ln_closse.coef_[0], 2) < 0:
            return round(self.ln_closse.coef_[0], 2), round(self.ln_closse.intercept_, 2)
        elif round(self.ln_closse.coef_[0], 2) > 0:
            return round(self.ln_lows.coef_[0], 2), round(self.ln_lows.intercept_, 2)

    def trend_line(self, lentgh: int):
        x = [i for i in range(lentgh)]
        s, st = self.trend_fit()
        line = [round(((i * s) + st), 2) for i in x]
        if s > 0:
            t_line = [round((i / 1.0005), 2) for i in line]
            r_line = [round((i + (self.high_1 - self.high_2)), 4)
                      for i in t_line]
            # 1.0025
        elif s < 0:
            t_line = [round((i * 1.0005), 2) for i in line]
            r_line = [round((i - (self.low_1 - self.low_2)), 4)
                      for i in t_line]
        return t_line, r_line


def pivot_point(symbol, time_frame,):
    high =  Symbol_data(symbol, time_frame, 1, 'h')
    low =  Symbol_data(symbol, time_frame, 1, 'l')
    close =  Symbol_data(symbol, time_frame, 1, 'c')
 

    pivot_point = (high + low + close) / 3
    support1 = (2 * pivot_point) - high
    support2 = pivot_point - (high - low)
    support3 = low - 2 * (high - pivot_point)
    resistance1 = (2 * pivot_point) - low
    resistance2 = pivot_point + (high - low)
    resistance3 = high + 2 * (pivot_point - low)
    
    return pivot_point , support1, support2, support3, resistance1, resistance2, resistance3

class Patterns:
    "I help you to find Some candle Patterns,\nSuch as 'engulfing', 'doji', 'threes' (soldiers/Raves)"
    # OHLC

    def __init__(self, symbol: str, time_frame: str, bar_range: int = 100) -> None:
        # OHLC
        self.symbol = symbol
        self.timeframe = time_frame
        self.candles_eng = mt5.copy_rates_from_pos(
            symbol, time_perid[time_frame], 1, 3)
        self.candles_doj = mt5.copy_rates_from_pos(
            symbol, time_perid[time_frame], 1, 4)
        self.candles_3 = mt5.copy_rates_from_pos(
            symbol, time_perid[time_frame], 1, 5)
        self.bars = mt5.copy_rates_from_pos(
            symbol, time_perid[time_frame], 1, bar_range)

    def engulfing(self) -> tuple:
        "I'll help you to find 'engulfing' pattern. Ascendig & Descendig"
        # Candle 1
        self.open_1: float = self.candles_eng[1][1]
        self.High_1: float = self.candles_eng[1][2]
        self.low_1: float = self.candles_eng[1][3]
        self.close_1: float = self.candles_eng[1][4]
        # Candle 2
        self.open_2: float = self.candles_eng[2][1]
        self.High_2: float = self.candles_eng[2][2]
        self.low_2: float = self.candles_eng[2][3]
        self.close_2: float = self.candles_eng[2][4]
        # Proces
        self.body_1: float = self.close_1 - self.open_1    # is it + or - ->  close - open
        self.body_2: float = self.close_2 - self.open_2    # is it + or - ->  close - open
        # Check Conditions
        if (self.body_1 < 0) and (self.body_2 > 0) and (abs(self.body_1) * 1.5 < self.body_2):
            return 'buy', 'Double Ascendig', self.open_2, self.low_1

        elif (self.body_1 > 0) and (self.body_2 < 0) and (self.body_1 * 1.5 < abs(self.body_2)):
            return 'sell', 'Double Descendig', self.High_1, self.open_2

        else:
            return 'No Position', 'No Pattern', None, None

    def doji(self) -> tuple:
        "I'll help you to find 'doji' pattern. Ascendig & Descendig"
        # # Candle 1
        self.open_1: float = self.candles_doj[1][1]
        self.High_1: float = self.candles_doj[1][2]
        self.low_1: float = self.candles_doj[1][3]
        self.close_1: float = self.candles_doj[1][4]
        # # Candle 2
        self.open_2: float = self.candles_doj[2][1]
        self.High_2: float = self.candles_doj[2][2]
        self.low_2: float = self.candles_doj[2][3]
        self.close_2: float = self.candles_doj[2][4]
        # # candle 3
        self.open_3: float = self.candles_doj[3][1]
        self.High_3: float = self.candles_doj[3][2]
        self.low_3: float = self.candles_doj[3][3]
        self.close_3: float = self.candles_doj[3][4]
        # Proces
        self.body_1: float = self.close_1 - self.open_1    # is it + or - ->  close - open
        self.body_2: float = self.close_2 - self.open_2    # is it + or - ->  close - open
        self.body_3: float = self.close_3 - self.open_3    # is it + or - ->  close - open
        self.z = abs(self.High_2 - self.open_2)
        self.y = abs(self.low_2 - self.close_2)
        self.x = abs(self.close_2 - self.open_2)
        # Conditios
        if (self.body_1 < 0.) and (self.body_3 > 0.) and (1.001 >= self.x >= 0.001):
            # (self.low_2 / self.open_2 == 1.00) and (0.8 < self.High_2 / self.close_2 > 0.85):
            return 'buy', 'Acending Doji', self.low_2, self.open_2

        elif (self.body_1 > 0.) and (self.body_3 < 0.) and (1.001 >= self.x >= 0.001):
            # (self.High_2 / self.close_2 == 1.00) and (0.9 < self.low_2 / self.open_2 < 0.99):
            return 'sell', 'Decending Doji', self.open_2, self.High_2

        else:
            return 'No Position', 'No Pattern', None, None

    def threes(self) -> tuple:
        """I'll help you to find 'Three soldires/Ravens' pattern.
        alittle advise: use me with help of my cusin 'trend', but how? find it yourself :) """
        # candle 0
        self.open_0: float = self.candles_3[1][1]
        self.close_0: float = self.candles_3[1][4]
        self.High_0: float = self.candles_3[1][2]
        self.low_0: float = self.candles_3[1][3]
        # Candle 1
        self.open_1: float = self.candles_3[2][1]
        self.close_1: float = self.candles_3[2][4]
        self.High_1: float = self.candles_3[2][2]
        self.low_1: float = self.candles_3[2][3]
        # # Candle 2
        self.open_2: float = self.candles_3[3][1]
        self.close_2: float = self.candles_3[3][4]
        self.High_2: float = self.candles_3[3][2]
        self.low_2: float = self.candles_3[3][3]
        # candle 3
        self.open_3: float = self.candles_3[4][1]
        self.close_3: float = self.candles_3[4][4]
        self.High_3: float = self.candles_3[4][2]
        self.low_3: float = self.candles_3[4][3]
        # Proces
        self.body_0: float = self.close_0 - self.open_0    # is it + or - ->  close - open
        self.body_1: float = self.close_1 - self.open_1    # is it + or - ->  close - open
        self.body_2: float = self.close_2 - self.open_2    # is it + or - ->  close - open
        self.body_3: float = self.close_3 - self.open_3    # is it + or - ->  close - open
        # Conditions
        if (self.body_0 < 0) and (self.body_1 > 0) and (self.body_2 > 0) and (self.body_3 > 0):
            return 'buy', 'Three Soldires', self.open_1, self.low_1

        elif (self.body_0 > 0) and (self.body_1 < 0) and (self.body_2 < 0) and (self.body_3 < 0):
            return 'sell', "three Ravens", self.open_1, self.High_2

        else:
            return 'No Position', 'No Pattern', None, None

# Support - Resistance Area


class SUPPORT_RESISTANCE:
    """
    I'm goint to find some Supports/Resistenc for you.
    Use 'result' Methods for Exact Numbers
    """

    def __init__(self, symbol: str, time_frame, bar_range: int = 100, n1: int = 3, n2: int = 2, l: int = 60) -> None:
        self.n1 = n1
        self.n2 = n2
        self.l = l
        self.bars = mt5.copy_rates_from_pos(
            symbol, time_perid[time_frame], 1, bar_range)
        temp_list = list()

        for i in self.bars:
            # Converting to tuple and then to array to fix an error.
            temp_list.append(list(i))
        self.bars = np.array(temp_list)

    def _support(self, l: int) -> int:
        "I'm giving you Supports lines, BUt for Find Real Lines use 'reslut' method"
        self.l = l
        # n1 n2 before and after candle l
        for i in range((self.l - self.n1 + 1), self.l + 1):
            if (self.bars[:, 3][i] > self.bars[:, 3][i-1]):
                # Compare 2 Lows  to eachother

                return 0
        for i in range(self.l + 1, self.l + self.n2+1):
            # Compare 2 Lows  to eachother:
            if (self.bars[:, 3][i] < self.bars[:, 3][i-1]):
                return 0
        return 1
    # OHLC

    def _resistance(self, l: int) -> int:
        "It's time to find Resistance lines, BUt for Find Real Lines use 'reslut' method"
        self.l = l
        # n1 n2 before and after candle l
        for i in range(self.l - self.n1 + 1, self.l + 1):
            if (self.bars[:, 2][i] < self.bars[:, 2][i-1]):
                return 0
        for i in range(self.l + 1, self.l + self.n2 + 1):
            if (self.bars[:, 2][i] > self.bars[:, 2][i-1]):
                return 0
        return 1

    def result(self):
        "I'm here to show you S/R Lins Values"
        S: list[float] = []
        R: list[float] = []
        for row in range(3, (len(self.bars) - self.n2)):
            if self._support(row):
                S.append(self.bars[:, 3][row])

            if self._resistance(row):
                R.append(self.bars[:, 2][row])
        return S, R

# Some indicator I found in TradingView


def chandelier_exit(symbol: str, time_frame: str, period: int, multiplier: float,  method: str):
    """
    somthing

    Keyword arguments:
    argument -- description
    Return: BUY : 1, SELL: 0
    """

    bars = Symbol_data(symbol, time_frame, period, method)
    close = Symbol_data(symbol, time_frame, 2, 'close')
    atr = multiplier * Average_True_Range(symbol, time_frame, period)
    long_stop = bars.max() - atr
    short_stop = bars.min() + atr

    flag1 = 1
    flag2 = 1

    if close[-1] < short_stop:
        flag1 = 1

        if bars[-1] < long_stop:
            flag2 = -1
        else:
            flag2 = 1
    else:
        flag1 = -1

    # BUY : 1, SELL: 0
    if flag1 == 1 and flag2 == -1:
        return 1
    elif flag1 == -1 and flag2 == 1:
        return 0

    else:
        return None

# ATR


def Average_True_Range(symbol: str, time_frame: str,  period: int) -> list:
    high = Symbol_data(symbol, time_frame, period, "h")
    low = Symbol_data(symbol, time_frame, period, "l")
    close = Symbol_data(symbol, time_frame, period, "c")

    true_range = list()
    # OHLC
    high_low = high - low
    high_close = np.abs(high - close)
    low_close = np.abs(low - close)

    for i, j, t in list(zip(high_low, high_close, low_close)):
        true_range.append(np.max((i, j, t)))

    # CalCulate AVERAGE
    atr = np.mean(np.roll(true_range[:], period))
    return round(atr, 2)





def slope(symbol: str, time_frame: str, methode : str, window: int) -> list:
    data = Symbol_data(symbol, time_frame, window, methode)

    x = np.arange(len(data))
    y = np.array(data)
    slope, _ = np.polyfit(x, y, 1)
    return slope



def relative_strength_index(symbol: str, time_frame: str, time_window: int, method: str) -> np.ndarray:
    prices = Symbol_data(symbol,  time_frame, time_window, method)
    n = time_window
    deltas = np.diff(prices)
    seed = deltas[:n+1]
    up = seed[seed >= 0].sum()/n
    down = -seed[seed < 0].sum()/n
    try:
        rs = up/down
        rsi = np.zeros_like(prices)
        rsi[:n] = 100. - 100./(1.+rs)
    except:
        rs = 0
        rsi = np.zeros_like(prices)
        rsi[:n] = 100. - 100./(1.+rs)

    for i in range(n, len(prices)):
        delta = deltas[i-1]  # The diff is 1 shorter

        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up*(n-1) + upval)/n
        down = (down*(n-1) + downval)/n

        rs = up/down
        rsi[i] = 100. - 100./(1.+rs)

    return rsi / 100


def stochastic_oscillator(symbol: str, time_frame: str, time_window: int, method: str):
    prices = Symbol_data(symbol,  time_frame, time_window, method)
    n = time_window
    high = np.zeros_like(prices)
    low = np.zeros_like(prices)
    close = np.zeros_like(prices)
    volume = np.zeros_like(prices)
    high[:n] = prices[:n]
    low[:n] = prices[:n]
    close[:n] = prices[:n]
    volume[:n] = prices[:n]
    for i in range(n, len(prices)):
        high[i] = np.max(prices[i-n+1:i+1])
        low[i] = np.min(prices[i-n+1:i+1])
        close[i] = prices[i]
        volume[i] = volume[i-1] + prices[i] - prices[i-1]
        return close, high, low, volume


def bollinger_bands(symbol: str, time_frame: str, time_window: int, method: str):
    prices = Symbol_data(symbol,  time_frame, time_window, method)
    n = time_window
    deltas = np.diff(prices)
    seed = deltas[:n+1]
    up = seed[seed >= 0].sum()/n
    down = -seed[seed < 0].sum()/n
    rs = up/down
    rsi = np.zeros_like(prices)
    rsi[:n] = 100. - 100./(1.+rs)
    for i in range(n, len(prices)):
        delta = deltas[i-1]  # The diff is 1 shorter
        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta
            up = (up*(n-1) + upval)/n
            down = (down*(n-1) + downval)/n
            rs = up/down
            rsi[i] = 100. - 100./(1.+rs)
            return rsi


def moving_average(symbol: str, time_frame: str, time_window: int, method: str):
    prices = Symbol_data(symbol,  time_frame, time_window, method)
    n = time_window
    return np.convolve(prices, np.ones((n,))/n, mode='valid')



def macd(symbol: str, time_frame: str, fast: int, slow: int,  method: str):
    slow_ma = moving_average(symbol, time_frame, slow, method)
    fast_ma = moving_average(symbol, time_frame, fast, method)

    return fast_ma - slow_ma



def exponential_moving_average(symbol: str, time_frame: str, time_window: int, method: str):
    prices = Symbol_data(symbol,  time_frame, time_window, method)
    n = time_window
    return np.convolve(prices, np.ones((n,))/np.power(n, 2), mode='valid')


def moving_standard_deviation(symbol: str, time_frame: str, time_window: int, method: str):
    prices = Symbol_data(symbol,  time_frame, time_window, method)
    n = time_window
    return np.convolve(prices, np.ones((n,))/n, mode='valid')


def ichimoku(symbol: str, time_frame: int, bar_range: int, conversion: int = 9, base: int = 26, b: int = 52):
    """
    Nothing Yet...
    """
    bars = mt5.copy_rates_from_pos(
        symbol, time_perid[time_frame], 0, bar_range)
    # bars = Symbol_data(symbol, time_frame, bar_range, 'open')
    temp_list = []
    for i in bars:
        # Converting to tuple and then to array to fix an error.
        temp_list.append(list(i))

    temp_list = np.array(temp_list)
    # OHLC
    teken_sen = convs_line = [None for _ in range(conversion)]
    kijin_sen = base_line = [None for _ in range(base)]
    senko_A = span_A = [None for _ in range(base)]
    senko_B = span_B = [None for _ in range(b)]
    Date = [None for _ in range(b)]

    for i in range(len(temp_list)):
        try:
            period_9 = ((max(temp_list[i - conversion: i+1, 2]) +
                        min(temp_list[i - conversion: i + 1, 3]))/2)
            convs_line.append(period_9)

            period_26 = (
                (max(temp_list[i - base: i+1, 2]) + min(temp_list[i - base: i + 1, 3]))/2)
            base_line.append(period_26)

            period_52 = (
                (max(temp_list[i - b: i+1, 2]) + min(temp_list[i - b: i + 1, 3]))/2)
            span_B.append(period_52)

            date = temp_list[i - conversion:i + 1, 0][-1]
            Date.append(date)

        except:
            pass

    for i, j in list(zip(convs_line, base_line)):
        try:
            span_A.append(((i+j)/2))
        except:
            pass

    return Date, convs_line, base_line, span_A, span_B


def Donchian(symbol: str, time_frame: str, bar_range: int, length: int = 20):
    """
    Nothing Yet...
    """
    bars = mt5.copy_rates_from_pos(
        symbol, time_perid[time_frame], 1, bar_range)
    # bars = Symbol_data(symbol, time_frame, bar_range, 'close')

    temp_list = []
    uppers = []
    lowers = []
    Date = []
    for i in bars:
        # Converting to tuple and then to array to fix an error.
        temp_list.append(list(i))

    temp_list = np.array(temp_list)
    # temp_list = bars
    # print(len(temp_list))
    # OHLC

    for i in range(len(temp_list)):
        try:
            upper = max(temp_list[i - length:i + 1, 2])
            lower = min(temp_list[i - length:i + 1, 3])
            date = temp_list[i - length:i + 1, 0][-1]

            uppers.append(upper)
            lowers.append(lower)
            Date.append(date)
        except:
            pass

    for _ in range(length):
        uppers.insert(0, uppers[0])
        lowers.insert(0, lowers[0])
        Date.insert(0, Date[0])

    base = [((i+j)/2) for i, j in list(zip(uppers, lowers))]

    return Date, uppers, base, lowers


def momentum(symbol: str, time_frame: str, window: int, method: str):
    price = Symbol_data(symbol, time_frame, window,  method)
    shifted_price = np.roll(price, 100)
    momentum = (price - shifted_price) / shifted_price * 100
    return momentum[-1]

def amount_sl(sym,  type, percent_equity = 1):
    pe = percent_equity / 100
    sym = sym.upper()
    type_ = type.lower()
    Buyyers : float = mt5.symbol_info_tick(sym)._asdict()['bid'] # Buyyers
    Sellers : float = mt5.symbol_info_tick(sym)._asdict()['ask'] # Sellers
    equity = mt5.account_info()._asdict()["equity"] 
    amount = equity * pe
    spread, pip = Symbol_info(sym)
    spread *= pip
    # print(amount)
    if type_ == 'buy':

        sl = Buyyers - amount  - spread

    elif type_ == 'sell':
        sl = Sellers + amount + spread
    else:
        raise KeyError('KeyWord UnRecugnized !')
    # print(Buyyers, sl, amount)
    return sl



def amount_tp(sym, type, percent_equity = 1):
    pe = percent_equity / 100
    sym = sym.upper()
    type = type.lower()
    Buyyers : float = mt5.symbol_info_tick(sym)._asdict()['bid'] # Buyyers
    Sellers : float = mt5.symbol_info_tick(sym)._asdict()['ask'] # Sellers
    equity = mt5.account_info()._asdict()["equity"]
    spread, pip = Symbol_info(sym)
    amount = equity * pe

    spread *= pip
    if type == 'buy':
        tp = Buyyers + amount 

    elif type == 'sell':
        tp = Sellers - amount 
    else:
        raise KeyError('KeyWord UnRecugnized !')
    
    return tp

def candle_sl(sym,  type, time_frame, candle_count = 1):

    sym = sym.upper()
    type_ = type.lower()

    highs = Symbol_data(sym, time_frame , candle_count, 'h')
    lows = Symbol_data(sym, time_frame , candle_count, 'l')


    spread, pip = Symbol_info(sym)
    spread *= pip
    # print(amount)
    if type_ == 'buy':

        sl = np.mean(lows)  - spread

    elif type_ == 'sell':
        sl = np.mean(highs) + spread
    else:
        raise KeyError('Sl KeyWord UnRecugnized !')

    return sl



def candle_tp(sym, type, time_frame, candle_count = 1):

    sym = sym.upper()
    type = type.lower()
    sym = sym.upper()
    type_ = type.lower()

    highs = Symbol_data(sym, time_frame , candle_count, 'h')
    lows = Symbol_data(sym, time_frame , candle_count, 'l')

    spread, pip = Symbol_info(sym)


    spread *= pip
    if type == 'buy':
        tp = np.mean(highs) + spread 

    elif type == 'sell':
        tp = np.mean(lows) - spread 
    else:
        raise KeyError('KeyWord UnRecugnized !')
    
    return tp


def update_sl(symbol, ):
    symbol = symbol.upper()
    if mt5.positions_get(symbol=symbol):
        try:
            sl_old = (mt5.positions_get(symbol=symbol)[0]._asdict())['sl']
            tp_old = (mt5.positions_get(symbol=symbol)[0]._asdict())['tp']
            ticket = (mt5.positions_get(symbol=symbol)[0]._asdict())['ticket']
            pos_type = (mt5.positions_get(symbol=symbol)[0]._asdict())['type']
            price_now = (mt5.positions_get(symbol=symbol)
                         [0]._asdict())['price_current']
            open_price = (mt5.positions_get(symbol=symbol)
                          [0]._asdict())['price_open']

            spread, pip = Symbol_info(symbol)
            diff = abs(sl_old - price_now)/ 2
            # print(diff)
            # sl_new = sl_old
            if pos_type == 1:  # SELL
                sl_new = sl_old - (diff + spread)  

            if pos_type == 0:  # BUY
                sl_new = sl_old + diff + spread
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": ticket,
                "symbol": symbol,
                "sl": sl_new,
                "tp": tp_old}
            result_SL = mt5.order_send(request)
            comment = result_SL._asdict()['comment']

            return comment
        except:
            return 'update faild.'


def update_tp(symbol):
    symbol = symbol.upper()
    if mt5.positions_get(symbol=symbol):
        try:
            tp_now = (mt5.positions_get(symbol=symbol)[0]._asdict())['tp']
            sl_old = (mt5.positions_get(symbol=symbol)[0]._asdict())['sl']
            ticket = (mt5.positions_get(symbol=symbol)[0]._asdict())['ticket']
            pos_type = (mt5.positions_get(symbol=symbol)[0]._asdict())['type']
            price_now = (mt5.positions_get(symbol=symbol)
                         [0]._asdict())['price_current']
            open_price = (mt5.positions_get(symbol=symbol)
                          [0]._asdict())['price_open']

            spread, pip = Symbol_info(symbol)
            diff = abs(price_now - open_price)/2

            tp_new = tp_now
            if pos_type == 1:  # SELL
                tp_new = tp_now - diff

            if pos_type == 0:  # BUY
                tp_new = tp_now + diff 
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": ticket,
                "symbol": symbol,
                "sl": sl_old,
                "tp": tp_new}
            result_SL = mt5.order_send(request)
            comment = result_SL._asdict()['comment']

            print("New TP", tp_new)
            return comment
        except:
            return 'update faild.'




def market_order(*, symbol: str, time_frame: str,
                 volume: float, order_type: str, deviation: int = 1, 
                 sl: float = "step" ,
                 tp: float = "step" ,
                 tp_rate: float = 1,
                 sl_rate: float = 1,

                 comment: str = "") -> List[dict]:
    """
    I'm responsiable to Open Deals
    -------
    if 'sl' or 'tp' is equal to -1.0, automated 'tp' and 'sl' will generate base on your 'RATIO' value.

    default value is '0.0' which means NO 'sl' or 'tp' is set.

    when you put sl and tp on auto generate: 
    `step`:
        tp_rate : 
            how much $ you want to bet on `take profit`
        sl_rate : 
            how much $ you want to bet on `sell stop`
    `step`:
        `sl_rate`:
            what precente of your euidity you want to bet for stopLoss
        `tp_rate`:
            what precente of your euidity you want to bet for takeprofit
    """

    tick = mt5.symbol_info_tick(symbol)

    SPREAD, POINT = Symbol_info(symbol)
    # POINT *= POINT
    order_dict = {'buy': 0, 'sell': 1}
    price_dict = {'buy': tick.ask, 'sell': tick.bid}
    SPREAD *= POINT
    POINT *= 100

    buy_sl = {
        sl : sl,
        "amount": amount_sl(symbol, order_type, sl_rate),
        "step"   : price_dict[order_type]  - (POINT* sl_rate)  - SPREAD,
        "atr"   : price_dict[order_type] - Average_True_Range(symbol , time_frame, sl_rate),
        "candle" : candle_sl(symbol, order_type, time_frame, sl_rate)
    }
    buy_tp = {
        tp : tp,
        "amount":  amount_tp(symbol, order_type, tp_rate),
        "step": price_dict[order_type]  + (POINT* tp_rate)  + SPREAD,
        "atr" :  price_dict[order_type] + Average_True_Range(symbol , time_frame, tp_rate),
        "candle" : candle_tp(symbol, order_type, time_frame, tp_rate)
    }
    
    sell_sl = {
        sl : sl,
        "amount": amount_sl(symbol, order_type, sl_rate),
        "step": price_dict[order_type] + SPREAD + (POINT * sl_rate),
        "atr" : price_dict[order_type] + Average_True_Range(symbol , time_frame, sl_rate),
        "candle" : candle_sl(symbol, order_type, time_frame, sl_rate)

    }

    sell_tp = {
        tp : tp,
        "amount": amount_tp(symbol,  order_type, tp_rate),
        "step":price_dict[order_type] - (tp_rate * POINT) + SPREAD,
        "atr"  : price_dict[order_type] - Average_True_Range(symbol , time_frame, tp_rate),
        "candle" : candle_tp(symbol, order_type, time_frame, tp_rate)
    }
    # BUY
    if order_type == 'buy' :
        sl = buy_sl[sl]
        tp = buy_tp[tp]

    # SELL
    elif order_type == 'sell':
        sl = sell_sl[sl]
        tp = sell_tp[tp]


    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": order_dict[order_type],
        "price": price_dict[order_type],
        "sl": sl ,
        "tp": tp ,
        "deviation": deviation,
        "magic": 379,
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK, }

    try:
        order_result: dict = mt5.order_send(request)._asdict()
        request_request: dict = (order_result['request'])._asdict()
                
        return request_request,  order_result
    except:
        return mt5.last_error(), None


def close_order(ticket, deviation: int = 20) -> dict:
    """
    I'm responsiable to Close Deals
    --
    function to close an order base don ticket id
    """
    positions = mt5.positions_get(ticket=ticket)

    for pos in positions:
        tick = mt5.symbol_info_tick(pos.symbol)
        # 0 represents buy, 1 represents sell - inverting order_type to close the position
        type_dict = {0: 1, 1: 0}
        price_dict = {0: tick.ask, 1: tick.bid}

        if pos.ticket == ticket:
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "position": pos.ticket,
                "symbol": pos.symbol,
                "volume": pos.volume,
                "type": type_dict[pos.type],
                "price": price_dict[pos.type],
                "deviation": deviation,
                "magic": 100,
                "comment": "python close order",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            order_result = mt5.order_send(request)

            return order_result

    return 'Ticket does not exist'


def close_all_by_symbol(symbol):
    tick = mt5.symbol_info_tick(symbol)
    position = mt5.positions_get(symbol=symbol)

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "position": position[0][0],
        "symbol": symbol,
        "volume": position[0][9],
        "type": mt5.ORDER_TYPE_BUY if position[0][5] == 1 else mt5.ORDER_TYPE_SELL,
        "price": tick.ask if position[0][5] == 1 else tick.bid,
        "deviation": 20,
        "magic": 101,
        "comment": "python close order",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,
    }
    res = mt5.order_send(request)
    return res


def close_all_by_ticket(ticket):
    position = mt5.positions_get(ticket=ticket)
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "position": position[0][0],
        "symbol": position[0][2],
        "volume": position[0][9],
        "type": mt5.ORDER_TYPE_BUY if position[0][5] == 1 else mt5.ORDER_TYPE_SELL,
        "price": position[0][4],
        "deviation": 20,
        "magic": 101,
        "comment": "python close order",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,
    }
    res = mt5.order_send(request)
    return res


def candel_color(sym,time_frame):
    Open = Symbol_data(sym=sym, time_frame=time_frame, bar_range= 1 ,method= 'o')[0]
    Close = Symbol_data(sym=sym, time_frame=time_frame, bar_range= 1 ,method= 'c')[0]
    if Open > Close:
        return -1 # Red
    elif Open < Close: # Green
        return 1
    else: #Doji
        return 0

def fair_value_gap( sym,time_frame ): 
    high = Symbol_data(sym=sym, time_frame=time_frame, bar_range= 1 ,method= 'h')[0]
    low = Symbol_data(sym=sym, time_frame=time_frame, bar_range= 1 ,method= 'h')[0]
    if not high >= low:
        return low - high
    elif not low <= high:
        return high - low
    else: 
        return None


def kill_mt5():
    mt5.shut_down()
######### END ############



"""
helo, Dear User..
I'm 'Trade Comedy' , I was Born as Money making bot, but now thigs have changed. 
todat I am a Colection of helping function and classes that can help 'OTHERS' to make a Money Making bot
here ,inside me, there are a candle-pattern class witch can find engulfing, doji and three's. 
I can help you find Supports and Resistance lines, Pivot Points, trend line.
I'm still a Kid, if you You want to help to grow see me at 'https://github.com/amirhu37/mt5-a-trade-comedy'

"""

import MetaTrader5 as mt5 
import numpy as np
from sklearn import linear_model as ln
from datetime import datetime
from os import system
from os.path import exists


mt5.initialize()
ordr_dict = {'buy': 0, 'sell': 1}

time_perid = {
    "1m" :      mt5.TIMEFRAME_M1,
    '5m' :      mt5.TIMEFRAME_M5,
    '15m':     mt5.TIMEFRAME_M15,
    '30m':     mt5.TIMEFRAME_M30,
    '1h' :     mt5.TIMEFRAME_H1,
    '4h' :    mt5.TIMEFRAME_H4,
    '1d' :    mt5.TIMEFRAME_D1,
    '1w' :  mt5.TIMEFRAME_W1 }


class static_list:
    def __init__(self, index: int) -> None:
        self.x = [None for _ in range(index)]
    def add(self,idx, value):
        try:
            self.x.insert(idx, value )
            self.x.pop(value)
            return self.x
        except:
            raise ValueError ("add faild. index out of reserved places")
    def delete(self, value):
        try:
            self.x.pop(value)
            return self.x
        except:
            raise ValueError ("delete failde, maybe value does not exsist")




def Symbol_data(sym: str, time_frame: str, bar_range: int, method: str , Open_candle: bool = False):
    """
    excract SYMBOL datas such as:
    ----
    method: 't' or 'time' , 'o' or'open' , 'h' or 'high', 'l' or 'low', 'c' or 'close', 'v' or 'volume'

    open_candle : means should include current open candle or not
    """
    o = 1
    if Open_candle == True : o = 0
    bars = mt5.copy_rates_from_pos(sym, time_perid[time_frame], o, bar_range)
    OHLC = {'t': 0 , 'o': 1 , 'h': 2, 'l': 3, 'c': 4, 'v':5,
        'time': 0 , 'open': 1 , 'high': 2, 'low': 3, 'close': 4, 'volume':5}
    temp_list = [i for i in bars] 
    
    if method == 'all':
         
        data = [temp_list[i] for i in range(len(temp_list)) ]
    elif method in OHLC:
        data = [temp_list[i][ OHLC[method]  ] for i in range(len(temp_list)) ]
    
    bars = np.array(data)
    return bars


def Symbol_info(s: str, rr: int = 1):
    """
    SYMBOLs can Have difrent pips and spreads.
    maybe I can help you with that.\nBTW rr is Risk/Reward Ratio
    -
    return: spread, point, r/r ratio
    """
    DATA: dict = mt5.symbol_info(s)._asdict()

    spread = DATA['spread']

    syms = {
        'BTCUSD': [spread/100, point := 100, rr],

        'XAUUSD':  [spread/100, point := 1, rr],
        'XAUUSDc':  [spread/100, point := 1, rr],
        'XAUUSDb': [spread/100, point := 1, rr],

        'ETHUSD':   [spread/10, point := 1, rr],

        'EURUSD':  [spread/10_000, point := .001, rr],
        'EURUSDc': [spread/10_000, point := .001, rr],
        'EURUSDb': [spread/10_000, point := .001, rr],

        'GBPUSD':  [spread/10_000, point := .001, rr],
        'GBPUSDc': [spread/10_000, point := .001, rr],
        'GBPUSDb': [spread/10_000, point := .001, rr],

        'USDCAD':  [spread/10_000, point := .001, rr],
        'USDCADc': [spread/10_000, point := .001, rr],
        'USDCADb': [spread/10_000, point := .001, rr],

        'USDJPY':  [spread/10, point := .01, rr],
        'USDJPYc': [spread/10, point := .01, rr],
        'USDJPYb': [spread/10, point := .01, rr],

    }

    return syms[s]


def moving_average(symbol: str, time_frame: str, bar_range: int , period: int , method: str ) -> list:
    """
    I can help you to Find Simple Moving Average, base on this methods:
    ----
    
    method: 'Open' ,'high', 'low', 'close', 'volume'
    ----
    
    time_Frame: 1m, 5m, 15m, 30m, 1h, 4h
    """
    method_dict = {"open": 1, 'high': 2, "low":3, "close": 4, 'volume': 5 }
    # barss = Symbol_data(symbol, time_frame, bar_range, method)
    barss = mt5.copy_rates_from_pos(symbol, time_perid[time_frame],  0 , bar_range)
    
    temp_list = list()
    movings_avg = list()

    for i in barss:
        # Converting to tuple and then to array to fix an error.
        temp_list.append(list(i))
       
    bars = np.array(temp_list)
      
    # print(len(bars))
    for i in range(0 , len(temp_list), period):
        try:
            # print(i, i+period)
            lasts = bars[i     : i + period , method_dict[method] ]  # SMA base on 4: CLOSE , 1: OPEN
            # print(len(lasts))
            ma =  np.mean(lasts)  

            movings_avg.append(round(ma,2))
            
        except:
            pass    
    
    # print(f"mving avg {period}: ", movings_avg)

    return movings_avg

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
    atr = multiplier * Average_True_Range(symbol,time_frame, period)
    long_stop = bars.max() - atr
    short_stop = bars.min() + atr
    
    
    flag1 = 1
    flag2 = 1

    if close[-1] < short_stop:
        flag1 = 1

        if bars[-1] <  long_stop :
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

#ATR
def Average_True_Range(symbol: str, time_frame: str , period: int ) -> list:
    bars = mt5.copy_rates_from_pos(symbol, time_perid[time_frame],  0 , period  )
    # bars = Symbol_data(symbol, time_frame, period , 'close' )
    temp_list = list()
    true_range = list()

    for i in bars:
        # Converting to tuple and then to array to fix an error.
        temp_list.append(list(i))  
    bars = np.array(temp_list)
    
    # OHLC
    high_low = bars[:,2] - bars[:,3]
    high_close = np.abs(bars[:,2] - bars[:,4])
    low_close = np.abs(bars[:,3] - bars[:,4])

    for i,j,t in list(zip(high_low, high_close, low_close )):
        true_range.append( np.max((i,j,t)) )

    # CalCulate AVERAGE
    atr = np.mean(np.roll(true_range[:], period ))
    return round(atr,2)

#RSI
def relative_strength_index(symbol: str, time_frame: str , period: int, method: str ):
    bar_range = period *2 
    OHLC = { 'open': 1 , 'high': 2, 'low': 3, 'close': 4, 'volume':5}
    bars = mt5.copy_rates_from_pos(symbol, time_perid[time_frame], 0, bar_range)
    
    temp_list = list()

    for i in bars:
        # Converting to tuple and then to array to fix an error.
        temp_list.append(list(i))  
    bars = np.array(temp_list)

    RSI = list() 
    avg_gain = list()
    avg_loss = list()

    # OHLC
    for i in range(0, len(bars), period):
        try:
            close_delta = np.diff(bars[ i :i +period, OHLC[method] ],1)

            gain   =        np.clip(close_delta, a_min = 0.0,    a_max = None)
            loss =  np.abs( np.clip(close_delta, a_min = None, a_max = 0.0)  )


            ma_gain = np.roll(gain, period).mean()
            ma_loss = np.roll(loss, period ).mean()
            avg_gain.append(ma_gain)
            avg_loss.append(ma_loss)
        except:
            pass

    for i in range(len(avg_gain)):
        try:
            wms_avg_gain = (avg_gain[i-1] * (period -1 ) + avg_gain[i] ) / period
            wms_avg_loss = (avg_loss[i-1] * (period -1 ) + avg_loss[i] ) / period

            rs = wms_avg_gain / wms_avg_loss

            rsi = 100 - ( 100 / (1 + rs) )

            RSI.append(round(rsi, 2) )
        except:
            pass
        
    return RSI




def journal(Symbol: str, vol: int, tf: int, ma: int, pos: str,  rr_Ration: int, price: float, sl: float, tp: float, comment: str, pattern: str) -> dict:
    """
    Journal your Trade Activity is a Big Deal,
    I'm a part of this High Duty
    """

    system("if exist Journal\ (echo None ) else (mkdir Journal\)")
    if not exists(f'Journal\{str(datetime.now())[:10]}.csv'):

        with open(f'Journal\{str(datetime.now())[:10]}.csv', 'a') as j:
            for k in journal.keys():
                j.writelines(f"{k},")
            j.writelines("\n")

    journal = {"date": str(datetime.now()),
               "Symbol": Symbol,
               "volume": vol,
               'Time-frame': tf,
               "Moving-Average": ma,
               "position": pos,
               "R/R Ratio": rr_Ration,
               "price":     f"{price: .2f}",
               "sl":        f"{sl: .2f}",
               "tp":        f"{tp: .2f}",
               "comment": comment,
               "pattern-type": pattern,
               }
    return journal

class Trend:
    def __init__(self,  symbol: str, time_frame: str, ma: int = 30) -> None:
        self.symbol = symbol
        self.time_frame = time_frame
        self.ma = ma
    def trend_fit(self,) -> float:
        "Nothing is more Important than Trend Line.\nI used Linear Regression for Find it"

        bars = mt5.copy_rates_from_pos(self.symbol, time_perid[self.time_frame], 1, self.ma)
        temp_list = [i for i in bars]
        highs = [temp_list[i][2] for i in range(len(temp_list))]
        lows = [temp_list[i][3] for i in range(len(temp_list))]

        
        self.high_1 : float = max(highs)
        self.low_1: float = max(lows)
        
        self.high_2 : float = min(highs)
        self.low_2: float = min(lows)
        

        highs = np.array(highs)
        lows =  np.array(lows)

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


    def trend_line(self, lentgh: int ):
        x = [i for i in range(lentgh)]
        s, st = self.trend_fit()
        line = [round(((i * s) + st), 2) for i in x]
        if s > 0:
            t_line = [round ((i / 1.0005) , 2) for i in line]
            r_line = [round (( i + (self.high_1 - self.high_2) ) , 4) for i in t_line]
            # 1.0025
        elif s < 0 : 
            t_line = [round((i * 1.0005), 2) for i in line]
            r_line = [round(( i -  (self.low_1 -  self.low_2) ), 4) for i in t_line]
        return t_line, r_line

class PIVOT:
    """
    Pivot Points can be Important.
    but a little bit of advice; use it for higher Time Frames
    Use this methos for support/Resistance of Pivot Points; 'resistaces_PP' ,  'supports_PP' ,'result'
    """

    def __init__(self, SYM: str, time_frame: str = '1h', bar_range: int = 2) -> None:
        self.candles = mt5.copy_rates_from_pos(SYM, time_perid[time_frame], 1, bar_range)
        # self.open_1: float = self.candles   [1][1] # OPEN
        self.High_1: float = self.candles[1][2]  # HIGH
        self.low_1: float = self.candles[1][3]  # LOW
        self.close_1: float = self.candles[1][4]   # CLOSE

        self.PP_list: list[float] = list()
        self.lows: list[float] = list()
        self.highs: list[float] = list()

        self.PP: float = (self.High_1 + self.low_1 + self.close_1) / 3

    def resistaces_PP(self, ):
        "Pivot Points Have 3 Resistance Lines, I will Find it For You\nBUT for exact Numbers Use 'result' Methods"
        r1: float = (2 * self.PP) - self.low_1
        r2: float = self.PP + (self.High_1 - self.low_1)
        r3: float = self.High_1 + 2*(self.PP - self.low_1)

        return round(r1, 2), round(r2, 2), round(r3, 2)

    def supports_PP(self,):
        "Pivot Points Have 3 Support Lines, I will Find it For You\nBUT for exact Numbers Use 'result' Methods"
        s1: float = (2 * self.PP) - self.High_1
        s2: float = self.PP - (self.High_1 - self.low_1)
        s3: float = self.low_1 - 2 * (self.High_1 - self.PP)

        return round(s1, 2), round(s2, 2), round(s3, 2)

    def result(self,):
        "Exact Numbers Are Here"
        s1 = self.sup_PP()
        r1 = self.resis_PP()

        return s1, r1, round(self.PP, 2)

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


class SUPPORT_RESISTANCE:
    """
    I'm goint to find some Supports/Resistenc for you.
    Use 'result' Methods for Exact Numbers
    """
    def __init__(self, symbol: str, time_frame, bar_range: int = 100, n1: int = 3, n2: int = 2, l: int = 60) -> None:
        self.n1 = n1
        self.n2 = n2
        self.l = l
        self.bars = mt5.copy_rates_from_pos(symbol, time_perid[time_frame], 1, bar_range)
        temp_list = list()

        for i in self.bars:
            # Converting to tuple and then to array to fix an error.
            temp_list.append(list(i))
        self.bars = np.array(temp_list)

    def support(self, l: int) -> int:
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

    def resistance(self, l: int) -> int:
        "It's time to find Resistance lines, BUt for Find Real Lines use 'reslut' method"
        self.l = l
        # n1 n2 before and after candle l
        for i in range(self.l - self.n1 + 1, self.l + 1):
            if (self.bars[:, 2][i] < self.bars[:, 2][i-1]):
                return 0
        for i in range(self.l + 1, self.l + self.n2 + 1):
            if(self.bars[:, 2][i] > self.bars[:, 2][i-1]):
                return 0
        return 1

    def result(self):
        "I'm here to show you S/R Lins Values"
        S: list[float] = []
        R: list[float] = []
        for row in range(3, (len(self.bars) - self.n2)):
            if self.support(row):
                S.append(self.bars[:, 3][row])

            if self.resistance(row):
                R.append(self.bars[:, 2][row])
        return S, R


def ichimoku(symbol: str, time_frame: int, bar_range : int ,conversion: int = 9 , base: int = 26 ,b : int = 52 ):
    """
    Nothing Yet...
    """
    bars = mt5.copy_rates_from_pos(symbol, time_perid[time_frame], 0, bar_range)
    # bars = Symbol_data(symbol, time_frame, bar_range, 'open')
    temp_list =  []
    for i in bars:
        # Converting to tuple and then to array to fix an error.
        temp_list.append(list(i))

    temp_list = np.array(temp_list)
    #OHLC
    teken_sen = convs_line= [None for _ in range(conversion)]
    kijin_sen = base_line = [None for _ in range(base)]
    senko_A = span_A =      [None for _ in range(base)]
    senko_B = span_B =      [None for _ in range(b)]
    Date =                  [None for _ in range(b)]



    for i in range(len(temp_list)):
        try:
            period_9  =  ( ( max( temp_list  [i - conversion : i+1 , 2] ) + min( temp_list[i - conversion: i +1 , 3]) )/2)
            convs_line.append(period_9)
            
            period_26 =  ( ( max( temp_list  [i - base       : i+1 , 2] ) + min( temp_list[i - base      : i +1 , 3]) )/2)
            base_line.append(period_26)           
            
            period_52 =  ( ( max(  temp_list [i - b          : i+1 , 2] ) + min( temp_list[i - b         : i +1 , 3]) )/2)
            span_B.append(period_52)

            date = temp_list[i - conversion :i + 1 , 0][-1]
            Date.append(date)

        except:
            pass

    for i,j in list(zip(convs_line, base_line)):
        try:
            span_A.append(((i+j)/2))
        except :
            pass


    return Date, convs_line, base_line, span_A, span_B


def Donchian(symbol:str, time_frame: str, bar_range: int, length: int = 20):
    """
    Nothing Yet...
    """
    bars = mt5.copy_rates_from_pos(symbol, time_perid[time_frame], 1, bar_range)
    # bars = Symbol_data(symbol, time_frame, bar_range, 'close')

    temp_list =  []
    uppers = []
    lowers = []
    Date =[]
    for i in bars:
        # Converting to tuple and then to array to fix an error.
        temp_list.append(list(i))

    temp_list = np.array(temp_list)
    # temp_list = bars
    # print(len(temp_list))
    #OHLC

    for i in range(len(temp_list)):
        try:
            upper = max( temp_list[i - length :i + 1 , 2] )  
            lower = min( temp_list[i - length :i + 1 , 3] ) 
            date = temp_list[i - length :i + 1 , 0][-1]

            uppers.append(upper)
            lowers.append(lower)
            Date.append(date)
        except:
            pass



    for _ in range(length):
        uppers.insert(0, uppers[0])
        lowers.insert(0, lowers[0])
        Date.insert(0, Date[0])

    base = [((i+j)/2) for i,j in list(zip(uppers, lowers))]

 
    return Date, uppers, base, lowers


def market_order(*,symbol: str, volume: float, order_type: str, deviation: int, sl: float =0.0 , tp: float = 0.0, RATIO: int = 1 ) -> dict:
    """
    I'm responsiable to Open Deals
    -------
    if 'sl' or 'tp' is equal to -1.0, automated 'tp' and 'sl' will generate base on your 'RATIO' value.
    
    default value is '0.0' which means NO 'sl' or 'tp' is set.
    
    when you put sl and tp on auto generate: REMEMBER to set RATIO : R/R ration.
    
    RATIO is base on 1 to ... for example: default is 1 to 1 
    """
    tick = mt5.symbol_info_tick(symbol)

    SPREAD, POINT, _ = Symbol_info(symbol)
    print('spread:', SPREAD)
    order_dict = {'buy': 0, 'sell': 1}
    price_dict = {'buy': tick.ask, 'sell': tick.bid}


    # BUY
    if order_type == 'buy':

        
        if sl == -1.0:
            sl = price_dict['buy'] - SPREAD - POINT
        if tp == -1.0:
            tp = price_dict['buy'] + RATIO*POINT + SPREAD

        print('sl rul; ', sl)
        print('price: ', price_dict['buy'])
        print('tp rul: ', tp)
        print('pr - tp ', f"{tp - price_dict['buy'] : .2f}")
        print('sl - tp ', f"{price_dict['buy'] - sl : .2f}")
    
    # SELL
    elif order_type == 'sell':  
        if sl == -1.0:
            sl = price_dict['sell'] + SPREAD + POINT
        if tp == -1.0:
            tp = price_dict['sell'] - RATIO*POINT - SPREAD



        print('sl rul; ', sl)
        print('price: ', f"{price_dict['sell']: .4f}")
        print('tp rul: ', tp)
        print('pr - tp ', price_dict['sell'] - tp)
        print('sl - tp ', price_dict['sell'] - sl)
        
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": order_dict[order_type],
        "price": price_dict[order_type],
        "sl": sl,
        "tp": tp,
        "deviation": deviation,
        "magic": 100,
        "comment": f"{str(datetime.now())[:19]}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC, }

    order_result: dict = mt5.order_send(request)._asdict()
    request_result: dict = (order_result['request'])._asdict()
    print("comment:", request_result['comment'])
    return order_result, request_result



def close_order(*,ticket, deviation: int) -> dict:
    """
    I'm responsiable to Close Deals
    --
    function to close an order base don ticket id
    """
    positions = mt5.positions_get()

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
            print(order_result)

            return order_result

    return 'Ticket does not exist'


######### END ############

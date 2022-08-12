"""
helo, Dear User..
I'm 'Trade Comedy' , I was Born as Money making bot, but now thigs have changed. 
todat I am a Colection of helping function and classes that can help 'OTHERS' to make a Money Making bot
here ,inside me, there are a candle-pattern class witch can find engulfing, doji and three's. 
I can help you find Supports and Resistance lines, Pivot Points, trend line.
I'm still a Kid, if you You want to help to grow see me at 'https://github.com/amirhu37/mt5-a-trade-comedy'

"""
import pandas as pd
import MetaTrader5 as mt5 
import numpy as np
from sklearn import linear_model as ln
from datetime import datetime
from os import system
from os.path import exists


# print(__doc__)
mt5.initialize()
ordr_dict = {'buy': 0, 'sell': 1}

time_perid = {
    1:      mt5.TIMEFRAME_M1,
    5:      mt5.TIMEFRAME_M5,
    15:     mt5.TIMEFRAME_M15,
    30:     mt5.TIMEFRAME_M30,
    60:     mt5.TIMEFRAME_H1,
    240:    mt5.TIMEFRAME_H4,
    360:    mt5.TIMEFRAME_D1,
    10080:  mt5.TIMEFRAME_W1
}


def Symbol_data(sym: str, tf: int, ma: int, ohlc: str, Open_candle: bool = False, o: int = 1):
    if Open_candle == True : o = 0
    bars = mt5.copy_rates_from_pos(sym, time_perid[tf], o, ma)
    OHLC = {'t': 0 , 'o': 1 , 'h': 2, 'l': 3, 'c': 4}
    temp_list = [i for i in bars] 
    
    if ohlc == 'all':
         
        data = [temp_list[i] for i in range(len(temp_list)) ]
    elif ohlc in OHLC:
        data = [temp_list[i][ OHLC[ohlc]  ] for i in range(len(temp_list)) ]
    return data

def Symbol_info(s: str, rr: int = 1):
    """SYMBOLs can Have difrent pips and spreads.
    maybe I can help you with that.\nBTW rr is Risk/Reward Ratio"""
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


def moving_average(symbol: str, TIME_FRAME: int, bar_period: int , period: int, method: str ) -> list:
    """
    I can help you to Find Simple Moving Average, base on this methods:
    ----
    
    method: 'Open' ,'high', 'low', 'close'
    ----
    
    time_Frame: 1,5,15,30,60
    """
    method_dict = {"open": 1, 'high': 2, "low":3, "close": 4 }
    bars = mt5.copy_rates_from_pos(symbol, time_perid[TIME_FRAME], 1, bar_period)
    temp_list = list()
    movings_avg = list()

    for i in bars:
        # Converting to tuple and then to array to fix an error.
        temp_list.append(list(i))
        last_close = bars[-1][ method_dict[method] ]  # SMA base on 4: CLOSE , 1: OPEN
        bars = np.array(temp_list)
        sma =  np.mean(bars[:, 1])  # 4: CLOSE , 1: OPEN Columns
    
    for i in range(0,len(temp_list), period):
        lasts = temp_list[i:i+10][ method_dict[method] ]  # SMA base on 4: CLOSE , 1: OPEN
        # print('lasts: ',lasts[:][1])
        ma =  np.mean(lasts[:][1])  # 4: CLOSE , 1: OPEN Columns
        movings_avg.append(ma)
    
    return movings_avg


def journal(Symbol: str, vol: int, tf: int, ma: int, pos: str,  rr_Ration: int, price: float, sl: float, tp: float, comment: str, pattern: str) -> dict:
    """Journal your Trade Activity is a Big Deal,
    I'm a part of this High Duty"""

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
    def __init__(self,  symbol: str, tf: int, ma: int = 30) -> None:
        self.symbol = symbol
        self.tf = tf
        self.ma = ma
    def trend_fit(self,) -> float:
        "Nothing is more Important than Trend Line.\nI used Linear Regression for Find it"

        bars = mt5.copy_rates_from_pos(self.symbol, time_perid[self.tf], 1, self.ma)
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
    "Pivot Points can be Important,\nbut a little bit of advice; use it for higher Time Frames\nUse this methos for support/Resistance of Pivot Points; 'resistaces_PP' ,  'supports_PP' ,'result'"

    def __init__(self, SYM: str, TF: int = 240, peride: int = 2) -> None:
        self.candles = mt5.copy_rates_from_pos(SYM, time_perid[TF], 1, peride)
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

    def __init__(self, symbol, time_frame, period) -> None:
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
            symbol, time_perid[time_frame], 1, period)

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
    "I'm goint to find some Supports/Resistenc for you.\nUse 'result' Methods for Exact Numbers"

    def __init__(self, symbol: str, timeframe, period, n1: int, n2: int, l: int = 60) -> None:
        self.n1 = n1
        self.n2 = n2
        self.l = l
        self.bars = mt5.copy_rates_from_pos(symbol, timeframe, 0, period)
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


def ichimoku(symbol: str, time_frame: int, bar_period : int ,conversion: int = 9 , base: int = 26 ,b : int = 52 ):
    
    bars = mt5.copy_rates_from_pos(symbol, time_perid[time_frame], 1, bar_period)

    temp_list =  []
    
    kijin_sen = base_line = []
    teken_sen = convs_line= []
    senko_A = span_A = []
    senko_B = span_B = []

    for i in bars:
        # Converting to tuple and then to array to fix an error.
        temp_list.append(list(i))

    
    temp_list = np.array(temp_list)
    #OHLC
    n = None
    high_period_9 = [max( temp_list[i:i+conversion, 2] ) for i in range(0, len(temp_list),conversion) ]
    low_period_9 = [min( temp_list[i:i+conversion, 3])  for i in range(0, len(temp_list),conversion) ]
    for _ in range(len(high_period_9)):
        high_period_9.insert(0, n)

    for _ in range(len(low_period_9)):
        low_period_9.insert(0, n)
         
    high_period_26 = [max( temp_list[i:i+base , 2] ) for i in range(0, len(temp_list),base) ]
    low_period_26 = [min( temp_list[i:i+base  , 3])  for i in range(0, len(temp_list),base) ]
    
    for _ in range(len(high_period_26)):
        high_period_26.insert(0, n)
    for _ in range(len(low_period_26)):
        low_period_26.insert(0, n)

    high_period_52 = [max( temp_list[i:i+b , 2] ) for i in range(0, len(temp_list),b) ]
    low_period_52 = [min( temp_list[i:i+b  , 3])  for i in range(0, len(temp_list),b) ]

    for _ in range(len(high_period_52)):
        high_period_52.insert(0, n)
    for _ in range(len(low_period_52)):
        low_period_52.insert(0, n)

    for i,j in list(zip(high_period_9, low_period_9)):
        try:

            convs_line.append(((i+j)/2))
        except TypeError:
            pass

    for i,j in list(zip(high_period_26, low_period_26)):
        try:

            base_line.append(((i+j)/2))
        except TypeError:
            pass

    for i,j in list(zip(convs_line, base_line)):
        try:

            span_A.append(((i+j)/2))
        except TypeError:
            pass

    for i,j in list(zip(high_period_52, low_period_52)):
        try:

            span_B.append(((i+j)/2))
        except TypeError:
            pass

    


    return convs_line, base_line, span_A, span_B

    

def market_order(*,symbol: str, volume: float, order_type: str, deviation: int,SPREAD: float, POINT: int, RATIO: int) -> dict:
    "I'm responsiable to Open Deals"
    tick = mt5.symbol_info_tick(symbol)

    #  = spread, point, rr_Ratio
    print('spread:', SPREAD)
    order_dict = {'buy': 0, 'sell': 1}
    price_dict = {'buy': tick.ask, 'sell': tick.bid}

    sl_rull = 0.0
    tp_rull = 0.0

    # BUY
    if order_type == 'buy':

        sl_rull = price_dict['buy'] - SPREAD - POINT
        tp_rull = price_dict['buy'] + RATIO*POINT + SPREAD

        print('sl rul; ', sl_rull)
        print('price: ', price_dict['buy'])
        print('tp rul: ', tp_rull)
        print('pr - tp ', f"{tp_rull - price_dict['buy'] : .2f}")
        print('sl - tp ', f"{price_dict['buy'] - sl_rull : .2f}")
    
    # SELL
    elif order_type == 'sell':  

        sl_rull = price_dict['sell'] + SPREAD + POINT
        tp_rull = price_dict['sell'] - RATIO*POINT - SPREAD

        print('sl rul; ', sl_rull)
        print('price: ', f"{price_dict['sell']: .4f}")
        print('tp rul: ', tp_rull)
        print('pr - tp ', price_dict['sell'] - tp_rull)
        print('sl - tp ', price_dict['sell'] - sl_rull)
        
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": order_dict[order_type],
        "price": price_dict[order_type],
        "sl": sl_rull,
        "tp": tp_rull,
        "deviation": deviation,
        "magic": 100,
        "comment": f"{str(datetime.now())[:19]}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC, }

    order_result: dict = mt5.order_send(request)._asdict()
    request_result: dict = (order_result['request'])._asdict()
    print("comment:", request_result['comment'])
    return order_result, request_result


# function to close an order base don ticket id
def close_order(*,ticket, deviation: int) -> dict:
    "I'm responsiable to Close Deals"
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

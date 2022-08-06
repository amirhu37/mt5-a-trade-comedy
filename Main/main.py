"""
helo, Dear User..
I'm 'Trade Comedy' , I was Born as Money making bot, but now thigs have changed. 
todat I am a Colection of helping function and classes that can help 'OTHERS' to make a Money Making bot
here ,inside me, there are a candle-pattern class witch can find engulfing, doji and three's. 
I can help you find Supports and Resistance lines, Pivot Points, trend line.
I'm still a Kid, if you You want to help to grow see me at 'https://github.com/amirhu37/mt5-a-trade-comedy'

"""

from MetaTrader5 import initialize,TIMEFRAME_M1, TIMEFRAME_M5, TIMEFRAME_M15, TIMEFRAME_M30, TIMEFRAME_H1, TIMEFRAME_H4, TIMEFRAME_D1, TIMEFRAME_W1, symbol_info, copy_rates_from_pos, symbol_info_tick, order_send, TRADE_ACTION_DEAL, ORDER_TIME_GTC, ORDER_FILLING_IOC, positions_get
from numpy import array, mean, arange, polyfit, poly1d 
from sklearn import linear_model as ln
from datetime import datetime
from os import system
from os.path import exists


# print(__doc__)
initialize()
ordr_dict = {'buy': 0, 'sell': 1}

time_perid = {
    1:      TIMEFRAME_M1,
    5:      TIMEFRAME_M5,
    15:     TIMEFRAME_M15,
    30:     TIMEFRAME_M30,
    60:     TIMEFRAME_H1,
    240:   TIMEFRAME_H4,
    360:   TIMEFRAME_D1,
    10080:  TIMEFRAME_W1
}


def Symbol_data(sym: str, tf: int, ma: int, ohlc: str, Open: bool = False):
    o = 1
    if Open : o = 0
    bars = copy_rates_from_pos(sym, time_perid[tf], o, ma)
    OHLC = {'o': 1 , 'h': 2, 'l': 3, 'c': 4}
    temp_list = [i for i in bars] 
    data = [temp_list[i][ OHLC[ohlc]  ] for i in range(len(temp_list)) ]
    return data

def Symbol_info(s: str, rr: int = 1):
    """SYMBOLs can Have difrent pips and spreads.
    maybe I can help you with that.\nBTW rr is Risk/Reward Ratio"""
    DATA: dict = symbol_info(s)._asdict()

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


def sma(symbol: str, TIME_FRAME: int, period: int) -> str:
    "I can help you to Find Simple Moving Average, base on 'Open's"
    bars = copy_rates_from_pos(symbol, time_perid[TIME_FRAME], 1, period)
    temp_list = list()
    for i in bars:
        # Converting to tuple and then to array to fix an error.
        temp_list.append(list(i))
        last_close = bars[-1][1]  # SMA base on 4: CLOSE , 1: OPEN
        bars = array(temp_list)
        sma = mean(bars[:, 1])  # 4: CLOSE , 1: OPEN Columns

    direction = 'flat'
    if last_close > sma:
        direction = 'buy'
    elif last_close < sma:
        direction = 'sell'

    return last_close, sma, direction


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

class Trend_reg:
    def __init__(self,  symbol: str, tf: int, ma: int = 20) -> None:
        self.symbol = symbol
        self.tf = tf
        self.ma = ma
    def trend_reg(self,) -> float:
        "Nothing is more Important than Trend Line.\nI used Linear Regression for Find it"

        bars = copy_rates_from_pos(self.symbol, time_perid[self.tf], 1, self.ma)
        temp_list = [i for i in bars]
        closes = [temp_list[i][4] for i in range(len(temp_list))]
        opens = [temp_list[i][1] for i in range(len(temp_list))]

        closes = array(closes)
        opens = array(opens)

        end = (closes.shape[0])+1

        q = arange(1, end)

        q = q.reshape(-1, 1)

        reg = ln.LinearRegression()

        self.ln_closse = reg.fit(q, closes)
        self.ln_opens = reg.fit(q, opens)

        if round(self.ln_closse.coef_[0], 2) < 0:
            return round(self.ln_closse.coef_[0], 2), round(self.ln_closse.intercept_, 2)
        elif round(self.ln_closse.coef_[0], 2) > 0:
            return round(self.ln_opens.coef_[0], 2), round(self.ln_opens.intercept_, 2)


    def trend_line(self,):
        x = [i for i in range(self.ma)]
        s, st = self.trend_reg()
        t_line = [round(((i * s) + st), 2) for i in x]
        # I have reasons to do this :>
        if s > 0:
            t_line = [i / 1.0025 for i in t_line]
        elif s < 0 : 
            t_line = [i * 1.0025 for i in t_line]
        return t_line



class Trend_np:
    def __init__(self,  symbol: str, tf: int, ma: int = 20) -> None:
        self.symbol = symbol
        self.tf = tf
        self.ma = ma
    def trend_np(self,):
        bars = copy_rates_from_pos(self.symbol, time_perid[self.tf], 0, self.ma)

        temp_list = [i for i in bars]

        closes = [temp_list[i][1] for i in range(len(temp_list)) ]

        xs = [i for i in range(len(closes))]

        closes = array(closes)
        z = polyfit (xs, closes, 1)
        self.p = poly1d (z)
        return round(self.p.coef[0] , 2), round(self.p.coef[1], 2)

    def trend_line(self, ):
        s, st = self.trend_np()
        x = [i for i in range(self.ma)]
        t_line = [round(((i * s) + st), 2) for i in x]
        return t_line



class PIVOT:
    "Pivot Points can be Important,\nbut a little bit of advice; use it for higher Time Frames\nUse this methos for support/Resistance of Pivot Points; 'resistaces_PP' ,  'supports_PP' ,'result'"

    def __init__(self, SYM: str, TF: int = 240, peride: int = 2) -> None:
        self.candles = copy_rates_from_pos(SYM, time_perid[TF], 1, peride)
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
        self.candles_eng = copy_rates_from_pos(
            symbol, time_perid[time_frame], 1, 3)
        self.candles_doj = copy_rates_from_pos(
            symbol, time_perid[time_frame], 1, 4)
        self.candles_3 = copy_rates_from_pos(
            symbol, time_perid[time_frame], 1, 5)
        self.bars = copy_rates_from_pos(
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
        self.bars = copy_rates_from_pos(symbol, timeframe, 0, period)
        temp_list = list()

        for i in self.bars:
            # Converting to tuple and then to array to fix an error.
            temp_list.append(list(i))
        self.bars = array(temp_list)

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


def market_order(*,symbol: str, volume: float, order_type: str, deviation: int,SPREAD: float, POINT: int, RATIO: int) -> dict:
    "I'm responsiable to Open Deals"
    tick = symbol_info_tick(symbol)

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
        "action": TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": order_dict[order_type],
        "price": price_dict[order_type],
        "sl": sl_rull,
        "tp": tp_rull,
        "deviation": deviation,
        "magic": 100,
        "comment": f"{str(datetime.now())[:19]}",
        "type_time": ORDER_TIME_GTC,
        "type_filling": ORDER_FILLING_IOC, }

    order_result: dict = order_send(request)._asdict()
    request_result: dict = (order_result['request'])._asdict()
    print("comment:", request_result['comment'])
    return order_result, request_result


# function to close an order base don ticket id
def close_order(*,ticket, deviation: int) -> dict:
    "I'm responsiable to Close Deals"
    positions = positions_get()

    for pos in positions:
        tick = symbol_info_tick(pos.symbol)
        # 0 represents buy, 1 represents sell - inverting order_type to close the position
        type_dict = {0: 1, 1: 0}
        price_dict = {0: tick.ask, 1: tick.bid}

        if pos.ticket == ticket:
            request = {
                "action": TRADE_ACTION_DEAL,
                "position": pos.ticket,
                "symbol": pos.symbol,
                "volume": pos.volume,
                "type": type_dict[pos.type],
                "price": price_dict[pos.type],
                "deviation": deviation,
                "magic": 100,
                "comment": "python close order",
                "type_time": ORDER_TIME_GTC,
                "type_filling": ORDER_FILLING_IOC,
            }

            order_result = order_send(request)
            print(order_result)

            return order_result

    return 'Ticket does not exist'


######### END ############

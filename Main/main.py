from ast import literal_eval
from enum import auto
import enum
import time
import MetaTrader5 as mt5
from numpy import array, mean, arange
from sklearn import linear_model as ln
from datetime import datetime
from os import system
from os.path import exists

# MASTER BRANCH
print('Trade ON Trend')

SYMBOL = input('SYMBOL: ')

VOLUME = 0.02 #float(input('Volume(Lotage 0.01 ~ 10): '))

TIMEFRAME = 1 # int(input('Time Frame (1, 5, 15, 30, 60): '))

# int(input('Moving Average(simple): ')) #SMA deleted but functions are still exsit
SMA_PERIOD = 40

RR_RATIO = 2 #int(input('Risk/Reward Ratio (1 to ?): '))

DEVIATION = 20

MINUTE = 60

minutes = TIMEFRAME * MINUTE

mt5.initialize()

journal = {"date": auto(),  # datetime.now(),
           "Symbol": SYMBOL,
           "volume": VOLUME,
           'Time-frame': TIMEFRAME,
           "Moving-Average": SMA_PERIOD,
           "position": auto(),
           "R/R Ratio": RR_RATIO,  # patt,
           "price": auto(),  # req['price'],
           "sl": auto(),  # req['sl'],
           "tp": auto(),  # req['tp'],
           "comment": auto(),  # pos['commnet'],
           "pattern-type": auto(),  # patt_type
           }


time_perid = {
    1: mt5.TIMEFRAME_M1,
    5: mt5.TIMEFRAME_M5,
    15: mt5.TIMEFRAME_M15,
    30: mt5.TIMEFRAME_M30,
    60: mt5.TIMEFRAME_H1,
    240: mt5.TIMEFRAME_H4,
    360: mt5.TIMEFRAME_D1,
    10080: mt5.TIMEFRAME_W1
}


def symbol_info(s: str, rr: int = 1):

    SYMBOL_DATA = mt5.symbol_info(s)._asdict()

    spread = SYMBOL_DATA['spread']

    syms = {
        'BTCUSD': [spread/100, point := 100, rr],

        'XAUUSD':  [spread/100, point := 1, rr],
        'XAUUSDc':  [spread/100, point := 1, rr],
        'XAUUSDb': [spread/100, point := 1, rr],

        'ETHUSD':   [spread/100, point := 0, rr],

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


def updator(date: str, pos: str, pr: float, sl: float, tp: float, comment: str, pattern: str) -> dict:
    journal['date'] = date  # str(datetime.now())[:19]
    journal['position'] = pos  # results[0]
    journal['price'] = f"{pr: .2f}"  # f"{req['price']: .2f}"
    journal['sl'] = f'{sl: .2f}'  # f"{req['sl']: .2f}"
    journal['tp'] = f"{tp: .2f}"  # f"{req['tp']: .2f}"
    journal['comment'] = comment  # position['comment']
    journal['pattern-type'] = pattern  # f"{results[3]}"
    return journal


def trend(symbol: str, tf, sma: int = 20,) -> float:

    bars = mt5.copy_rates_from_pos(symbol, time_perid[tf], 1, sma)

    temp_list = list()

    xs = list()

    for i in bars:
        # Converting to tuple and then to array to fix an error.
        temp_list.append(list(i))
        last_close = bars[-1][4]  # SMA base on 4: CLOSE , 1: OPEN
        last_open = bars[-1][1]
        bars = array(temp_list)
        sma = mean(bars[:, 1])  # 4: CLOSE , 1: OPEN Columns
        xs.append(round(((last_close+last_open)/2), 2))

    xs = array(xs)

    end = (xs.shape[0])+1

    q = arange(1, end)

    q = q.reshape(-1, 1)

    reg = ln.LinearRegression()
    reg.fit(q, xs)

    slope_, STD = round(reg.coef_[0], 2), round(reg.intercept_, 2)

    x = [i for i in range(sma)]
    t_line = [ round(((i * slope_) + STD), 2 )  for i in x]
    return t_line



class S_R:
    def __init__(self, symbol: str, timeframe, period, n1: int, n2: int, l: int = 60) -> None:
        self.n1 = n1
        self.n2 = n2
        self.l = l
        self.bars = mt5.copy_rates_from_pos(symbol, timeframe, 0, period)
        temp_list = list()

        for i in self.bars:
            # Converting to tuple and then to array to fix an error.
            temp_list.append(list(i))
        self.bars = array(temp_list)

    def support(self, l: int) -> int:
        self.l = l
        # n1 n2 before and after candle l
        for i in range(self.l - self.n1 + 1, self.l + 1):
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
        S: list[float] = []
        R: list[float] = []
        for row in range(3, (len(self.bars) - self.n2)):
            if self.support(row):
                S.append(self.bars[:, 3][row])

            if self.resistance(row):
                R.append(self.bars[:, 2][row])
        return S, R


def market_order(symbol: str, volume: float, order_type: str, **kwargs) -> dict:
    tick = mt5.symbol_info_tick(symbol)

    SPREAD, POINT, RATIO = symbol_info(symbol, RR_RATIO)
    print('spread:', SPREAD)
    order_dict = {'buy': 0, 'sell': 1}
    price_dict = {'buy': tick.ask, 'sell': tick.bid}

    sl_rull = 0.0
    tp_rull = 0.0

    if order_type == 'buy':

        sl_rull = price_dict['buy'] - SPREAD - POINT
        tp_rull = price_dict['buy'] + (RATIO*POINT) + SPREAD

        print('sl rul; ', sl_rull)
        print('price: ', price_dict['buy'])
        print('tp rul: ', tp_rull)
        print('pr - tp ', f"{tp_rull - price_dict['buy'] : .2f}")
        print('sl - tp ', f"{price_dict['buy'] - sl_rull : .2f}")

    elif order_type == 'sell':  # SELL

        sl_rull = price_dict['sell'] + SPREAD + POINT
        tp_rull = price_dict['sell'] - (RATIO*POINT) - SPREAD

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
        "deviation": DEVIATION,
        "magic": 100,
        "comment": f"{str(datetime.now())[:19]}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC, }

    order_result: dict = mt5.order_send(request)._asdict()
    request_result: dict = (order_result['request'])._asdict()
    print("comment:", request_result['comment'])
    return order_result, request_result

# function to close an order base don ticket id


def close_order(ticket) -> dict:
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
                "deviation": DEVIATION,
                "magic": 100,
                "comment": "python close order",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            order_result = mt5.order_send(request)
            print(order_result)

            return order_result

    return 'Ticket does not exist'


if True:
    system("if exist Journal\ (echo None ) else (mkdir Journal\)")
    pass

if __name__ == '__main__':

    if not exists(f'Journal\{str(datetime.now())[:10]}.csv'):

        with open(f'Journal\{str(datetime.now())[:10]}.csv', 'a') as j:
            for k in journal.keys():
                j.writelines(f"{k},")
            j.writelines("\n")

    ordr_dict = {'buy': 0, 'sell': 1}

    while True:

        system('cls')

        print("\n")

        SUPPORTS, RESISTANCE = S_R(SYMBOL, TIMEFRAME,SMA_PERIOD,2,3,23).result()


        print("Symbol: ", SYMBOL, "\n\n")

        slope, std_ = trend(SYMBOL, TIMEFRAME, SMA_PERIOD).slop
        # line = trend(slope, std_, SMA_PERIOD)

        # OHLC
        open_now: float = (list(mt5.copy_rates_from_pos(
            SYMBOL, time_perid[TIMEFRAME], 1, 2))[-1][1]).item()  # 1 open , 4 close

        high_now: float = (list(mt5.copy_rates_from_pos(
                    SYMBOL, time_perid[TIMEFRAME], 1, 2))[-1][2]).item()  # 1 open , 4 close
            
        low_now: float = (list(mt5.copy_rates_from_pos(
                    SYMBOL, time_perid[TIMEFRAME], 0, 2))[-1][3]).item()  # 1 open , 4 close


        close_now: float = (list(mt5.copy_rates_from_pos(
            SYMBOL, time_perid[TIMEFRAME], 1, 2))[-1][4]).item()  # 1 open , 4 close
        # get price now and turn it from np float to a native float
        
        # print('Slop: ', slope)
        # print('std:' , std_)
        # print('t line: ', line)

        # # BUY
        # for  j in line:
        #     if not mt5.positions_total():
        #         if slope < 0  and close_now > j :

        #             market_order(SYMBOL, VOLUME,'buy')

        #             time.sleep( ((SMA_PERIOD+1)/3) * 60)

        # # Sell        
        # for  j in line:
        #     if not mt5.positions_total():
        #         if slope > 0  and close_now < j :

        #             market_order(SYMBOL, VOLUME,'sell')

        #             time.sleep( ((SMA_PERIOD+1)/3) * 60)



        time.sleep(1)
        # break
        system('cls')
######### END ############

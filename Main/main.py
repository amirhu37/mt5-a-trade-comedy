
import time
import MetaTrader5 as mt5
from numpy import array, mean, arange
from sklearn import linear_model as ln 
from datetime import datetime
from os import system
from os.path import exists


SYMBOL =  input('SYMBOL: ')
VOLUME =  float(input('Volume(Lotage 0.01 ~ 10): '))
TIMEFRAME =  int(input('Time Frame (1, 5, 15, 30, 60): '))
PERIOD = 10            # int(input('how many candles to candles: ')) #SMA deleted but functions are still exsit
RR_RATIO = int(input('Risk/Reward Ratio (1 to ?): '))
DEVIATION = 20
MINUTE = 60
minutes = TIMEFRAME * MINUTE

mt5.initialize()

journal = {"date": None,  # datetime.now(),
           "Symbol": SYMBOL,
           "volume": VOLUME,
           'Time-frame': TIMEFRAME,
           "Moving-Average": PERIOD,
           "position": None,
           "R/R Ratio": RR_RATIO,  # patt,
           "price": None,  # req['price'],
           "sl": None,  # req['sl'],
           "tp": None,  # req['tp'],
           "comment": None,  # pos['commnet'],
           "pattern-type": None,  # patt_type
           }

time_perid = {
    1: mt5.TIMEFRAME_M1,
    5: mt5.TIMEFRAME_M5,
    15: mt5.TIMEFRAME_M15,
    30: mt5.TIMEFRAME_M30,
    60: mt5.TIMEFRAME_H1,
    240 : mt5.TIMEFRAME_H4,
    360 : mt5.TIMEFRAME_D1,
    10080: mt5.TIMEFRAME_W1
}


def symbol_info(s: str, rr: int=1):

    SYMBOL_DATA = mt5.symbol_info(s)._asdict()

    spread = SYMBOL_DATA['spread']
    
    syms = {
        'BTCUSD': [spread/100, point:=100, rr],

        'XAUUSD' :  [spread/100 , point:= 1 , rr],
        'XAUUSDc':  [spread/100 , point:= 1 , rr],
        'XAUUSDb' : [spread/100 , point:= 1 , rr],

        'ETHUSD':   [spread/100 , point:= 0 , rr],

        'EURUSD' :  [spread/10_000 , point:= .001 , rr],
        'EURUSDc' : [spread/10_000 , point:= .001 , rr],
        'EURUSDb' : [spread/10_000 , point:= .001 , rr],

        'GBPUSD' :  [spread/10_000, point:= .001 , rr],
        'GBPUSDc' : [spread/10_000, point:= .001 , rr],
        'GBPUSDb' : [spread/10_000, point:= .001 , rr],

        'USDCAD' :  [spread/10_000, point:= .001 , rr],
        'USDCADc' : [spread/10_000, point:= .001 , rr],
        'USDCADb' : [spread/10_000, point:= .001 , rr],

        'USDJPY' :  [spread/10, point:= .01 , rr],
        'USDJPYc' : [spread/10, point:= .01 , rr],
        'USDJPYb' : [spread/10, point:= .01 , rr],

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

class PIVOT:

    def __init__(self, SYM: str, TF: int = 240, peride: int = 2 ) -> None:
        self.candles =  mt5.copy_rates_from_pos(SYM, time_perid[TF], 1, peride)
        # self.open_1: float = self.candles   [1][1] # OPEN
        self.High_1: float = self.candles   [1][2] # HIGH
        self.low_1: float =  self.candles   [1][3]  # LOW
        self.close_1: float = self.candles  [1][4]   # CLOSE 

        self.PP_list: list[float] = list()
        self.lows : list[float] = list()
        self.highs : list[float] = list()

        self.PP : float = (self.High_1 + self.low_1 + self.close_1) / 3


    def resis_PP(self, ):
        r1 : float  = (2 * self.PP) - self.low_1
        r2 : float = self.PP + (self.High_1 - self.low_1)
        r3 : float = self.High_1 + 2*(self.PP - self.low_1)
        
        return round(r1,2), round(r2,2), round(r3,2)

    def sup_PP(self,):
        s1 : float = (2 * self.PP) - self.High_1
        s2 : float = self.PP - (self.High_1 - self.low_1)
        s3 : float = self.low_1 - 2 * (self.High_1 - self.PP)

        return round(s1, 2), round(s2,2), round(s3,2)

    def result(self,):

        s1 = self.sup_PP()
        r1 = self.resis_PP()
        
        return s1, r1, round(self.PP, 2)

class Patterns:
    # OHLC
    def __init__(self, symbol, timeframe, period)-> None:
        # OHLC
        self.symbol = symbol
        self.timeframe = timeframe
        self.candles_eng =  mt5.copy_rates_from_pos( symbol, time_perid[TIMEFRAME], 1, 3)
        self.candles_doj =  mt5.copy_rates_from_pos( symbol, time_perid[TIMEFRAME], 1, 4)
        self.candles_3 =    mt5.copy_rates_from_pos( symbol, time_perid[TIMEFRAME], 1, 5)
        self.bars =         mt5.copy_rates_from_pos(symbol, time_perid[TIMEFRAME], 1, period)

    def engulfing(self) -> tuple:
        # Candle 1
        self.open_1: float =  self.candles_eng  [1][1]
        self.High_1: float =  self.candles_eng  [1][2]
        self.low_1: float =   self.candles_eng  [1][3]
        self.close_1: float = self.candles_eng  [1][4]
        # Candle 2
        self.open_2: float =  self.candles_eng   [2][1]
        self.High_2: float =  self.candles_eng   [2][2]
        self.low_2: float =   self.candles_eng   [2][3]
        self.close_2: float = self.candles_eng   [2][4]
        # Proces
        self.body_1: float = self.close_1 - self.open_1    # is it + or - ->  close - open
        self.body_2: float = self.close_2 - self.open_2    # is it + or - ->  close - open
        # Check Conditions
        if (self.body_1 < 0) and (self.body_2 > 0) and (abs(self.body_1) * 1.5 < self.body_2):
            return 'buy' , 'Double Ascendig',self.open_2 , self.low_1
        
        elif (self.body_1 > 0) and (self.body_2 < 0) and ( self.body_1 * 1.5 < abs(self.body_2) ):
            return 'sell' , 'Double Descendig', self.High_1 , self.open_2
        
        else:
            return 'No Position','No Pattern' , None , None

    def doji(self) -> tuple:
        # # Candle 1
        self.open_1: float =  self.candles_doj  [1][1]
        self.High_1: float =  self.candles_doj  [1][2]
        self.low_1: float =   self.candles_doj  [1][3]
        self.close_1: float = self.candles_doj  [1][4]
        # # Candle 2
        self.open_2: float = self.candles_doj  [2][1]
        self.High_2: float = self.candles_doj  [2][2]
        self.low_2: float = self.candles_doj   [2][3]
        self.close_2: float = self.candles_doj [2][4]
        # # candle 3
        self.open_3: float = self.candles_doj  [3][1]
        self.High_3: float = self.candles_doj  [3][2]
        self.low_3: float = self.candles_doj   [3][3]
        self.close_3: float = self.candles_doj [3][4]
        # Proces
        self.body_1: float = self.close_1 - self.open_1    # is it + or - ->  close - open
        self.body_2: float = self.close_2 - self.open_2    # is it + or - ->  close - open
        self.body_3: float = self.close_3 - self.open_3    # is it + or - ->  close - open
        self.z = abs(self.High_2 -  self.open_2)
        self.y = abs(self.low_2 - self.close_2)
        self.x = abs(self.close_2 - self.open_2)
        # Conditios
        if (self.body_1 < 0.) and  (self.body_3 > 0.) and (1.001 >= self.x >= 0.001):
            #(self.low_2 / self.open_2 == 1.00) and (0.8 < self.High_2 / self.close_2 > 0.85):
            return 'buy' ,'Acending Doji' , self.low_2 , self.open_2
        
        elif (self.body_1 > 0.)  and (self.body_3 < 0.) and (1.001 >= self.x >= 0.001) :
            # (self.High_2 / self.close_2 == 1.00) and (0.9 < self.low_2 / self.open_2 < 0.99):
            return 'sell', 'Decending Doji', self.open_2 , self.High_2
        
        else:
            return 'No Position','No Pattern' , None , None

    def three_consecutive(self) -> tuple:
        # candle 0
        self.open_0: float =  self.candles_3   [1][1]
        self.close_0: float = self.candles_3   [1][4]
        self.High_0: float =  self.candles_3   [1][2]
        self.low_0: float =   self.candles_3   [1][3]
        # Candle 1
        self.open_1: float =  self.candles_3   [2][1]
        self.close_1: float = self.candles_3   [2][4]
        self.High_1: float =  self.candles_3   [2][2]
        self.low_1: float =   self.candles_3   [2][3]
        # # Candle 2
        self.open_2: float =  self.candles_3   [3][1]
        self.close_2: float = self.candles_3   [3][4]
        self.High_2: float =  self.candles_3   [3][2]
        self.low_2: float =   self.candles_3   [3][3]
        # candle 3
        self.open_3: float =  self.candles_3   [4][1]
        self.close_3: float = self.candles_3   [4][4]
        self.High_3: float =  self.candles_3   [4][2]
        self.low_3: float =   self.candles_3   [4][3]
        # Proces
        self.body_0: float = self.close_0 - self.open_0    # is it + or - ->  close - open
        self.body_1: float = self.close_1 - self.open_1    # is it + or - ->  close - open
        self.body_2: float = self.close_2 - self.open_2    # is it + or - ->  close - open
        self.body_3: float = self.close_3 - self.open_3    # is it + or - ->  close - open
        # Conditions
        if (self.body_0 < 0 ) and (self.body_1 > 0) and (self.body_2 > 0) and (self.body_3 > 0):
            return 'buy', 'Three Soldires' ,self.open_1, self.low_1
        
        elif (self.body_0 > 0 ) and (self.body_1 < 0) and (self.body_2 < 0) and (self.body_3 < 0):
            return 'sell', "three Ravens" , self.open_1, self.High_2
        
        else:
            return 'No Position','No Pattern' , None , None

    def sma(self) -> str:
        temp_list = list()
        for i in self.bars:
            # Converting to tuple and then to array to fix an error.
            temp_list.append(list(i))
            last_close = self.bars[-1][1]  # SMA base on 4: CLOSE , 1: OPEN
            bars = array(temp_list)
            sma = mean(bars[:, 1])  # 4: CLOSE , 1: OPEN Columns

        direction = 'flat'
        if last_close > sma:
            direction = 'buy'
        elif last_close < sma:
            direction = 'sell'
        
        return direction

    def result(self) -> tuple:
#```````````````````````````````````    buy parts   ``````````````````````````````````````
        # if self.sma() == 'buy':
        if self.engulfing()[0]  == 'buy':
            # 2: open , 3: low,
            return 'buy' , self.engulfing()[2], self.engulfing()[3], self.engulfing()[1]

        elif self.doji()[0]  == 'buy':
            # 2: open , 3: low
            return 'buy', self.doji()[2], self.doji()[3] , self.doji()[1]

        elif self.three_consecutive()[0]  == 'buy':
            # 2: open , 3: low
            return 'pass', self.three_consecutive()[2], self.three_consecutive()[3] , self.three_consecutive()[1]
        
#```````````````````````````````````  Sell parts  ````````````````````````````````````````
        elif self.engulfing()[0]  == 'sell':
            # 2: open , 3: low
            return 'sell', self.engulfing()[2], self.engulfing()[3], self.engulfing()[1]

        elif self.doji()[0] ==  'sell':
           # 2: open , 3: High
            return 'sell', self.doji()[2], self.doji()[3], self.doji()[1]
            
        elif self.three_consecutive()[0] ==  'sell':
            # 2: open , 3: High
            return 'pass', self.three_consecutive()[2] , self.three_consecutive()[3], self.three_consecutive()[1]
        else:
            return 'Nothin\'', 0, 0 , 'No Positions'
     
class SUPPORT_RESISTANCE:
    def __init__(self, symbol: str, timeframe, period, n1: int, n2: int, l : int = 60) -> None:
        self.n1 = n1
        self.n2 = n2
        self.l = l
        self.bars = mt5.copy_rates_from_pos(symbol, timeframe, 0, period)
        temp_list = list()
        
        for i in self.bars:
            # Converting to tuple and then to array to fix an error.
            temp_list.append(list(i))
        self.bars = array(temp_list)

    def support(self, l : int) -> int:
        self.l = l
        # n1 n2 before and after candle l
        for i in range(self.l - self.n1 + 1, self.l + 1 ):
            if (self.bars[:, 3][i] > self.bars[:, 3][i-1]): 
                # Compare 2 Lows  to eachother
                return 0
        for i in range(self.l + 1, self.l + self.n2+1):
            # Compare 2 Lows  to eachother:
            if (self.bars[:, 3][i] < self.bars[:, 3][i-1]):
                return 0
        return 1
#OHLC
    def resistance(self, l: int ) -> int: 
        self.l = l
        # n1 n2 before and after candle l
        for i in range(self.l - self.n1 + 1, self.l + 1):
            if (self.bars[:, 2][i] < self.bars[:, 2][i-1]):
                return 0
        for i in range(self.l + 1, self.l + self.n2 + 1):
            if(self.bars[:, 2][i] > self.bars[:, 2][i-1]):
                return 0
        return 1

    def result(self) :
        S: list[float] = []
        R: list[float] = []
        for row in range(3, (len(self.bars) - self.n2 ) ):
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
        tp_rull = price_dict['buy'] + RATIO*POINT + SPREAD

        print('sl rul; ', sl_rull)
        print('price: ', price_dict['buy'])
        print('tp rul: ', tp_rull)
        print('pr - tp ', f"{tp_rull - price_dict['buy'] : .2f}")
        print('sl - tp ', f"{price_dict['buy'] - sl_rull : .2f}")

    elif order_type == 'sell':  # SELL

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
        type_dict = {0: 1, 1: 0}  # 0 represents buy, 1 represents sell - inverting order_type to close the position
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

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    Main    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == '__main__':
    
    if not exists(f'Journal\{str(datetime.now())[:10]}.csv'):

        with open(f'Journal\{str(datetime.now())[:10]}.csv', 'a') as j:
            for k in journal.keys():
                j.writelines(f"{k},")
            j.writelines("\n")      

    ordr_dict = {'buy': 0, 'sell': 1}

    PP = PIVOT(SYMBOL)
    
    sup_PP ,resis_PP, p_point = PP.result()

    
    while True:

        system('cls')

        print("\n")

        print("Symbol: ",SYMBOL, "\n\n")

        # SIGNAL = Signal(SYMBOL, TIMEFRAME, PERIOD)

        # moment = momentom(SYMBOL, TIMEFRAME)

        # results = SIGNAL.result()
        
        # SUPPORTS, RESISTANCES = SUPPORT_RESISTANCE(SYMBOL,TIMEFRAME,110 ,3 ,2).result()
        
        # all_suports = (SUPPORTS + list(sup_PP))

        # all_suports = [float(v) for v in all_suports]
        
        # all_resistance = (RESISTANCES + list(resis_PP))
        
        # all_resistance = [float(i) for i in all_resistance]

        # print('pivot point:',p_point )
        # print('\n')
        # print('PP Resistances: ', resis_PP)
        # print('\n')
        # print('PP Supports: ', sup_PP )
        # print('\n')
        # print('Support: ', SUPPORTS)
        # print('\n')
        # print( 'Resistances: ' ,RESISTANCES)
        # print('\n')
        # print('position: ', results[0] )
        # print('\n')
        # print( 'result: ', results[3])
        # print('\n')
        # print("slope: ", moment)

        # price_now  : float= (list(mt5.copy_rates_from_pos(SYMBOL, time_perid[TIMEFRAME],0,2) )[-1][4]).item()  
        #                     ## get price now and turn it from np float to a native float
        # print('High: ', SIGNAL.__dict__["High_1"])
        # print('Open: ' ,SIGNAL.__dict__['open_1'])
        # print( 'low: ', SIGNAL.__dict__['low_1'])
        # print("price now: ",price_now)  

        # # OHLC.
        
        # if results[0] == 'buy' and (5.00 >= abs(moment) >= 1.10):

        #     for i in all_suports:
                
        #         if ( results[2] >= i >= results[1] )  :
        #             if not mt5.positions_total():
        #                 try:
        #                     # Open Position
        #                     position, req = market_order(SYMBOL, VOLUME, results[0])
        #                     # Regiter Data
        #                     J = updator(str(datetime.now())[:19], results[0], req['price'], req['sl'], req['tp'], position['comment'], results[3])
        #                     print(
        #                         f"{J['comment']}: {results[0]}, by {results[3]}")

        #                     with open(f'Journal\{str(datetime.now())[:10]}.csv', 'a') as j:

        #                         for v in J.values():
        #                             j.writelines(f"{v},")

        #                         j.writelines("\n")

        #                         print('data registed')

        #                     time.sleep(minutes * 2)

        #                 except:
        #                     print(f"{results[0]} poition didn\' open. somthin\' wnt wrong")
        #                     pass
    
        #     # if we have a BUY signal, close all short positions
        #     # for i in range(len(all_suports)):
        #     #     if price_now == all_suports[i]:
        #     #         try:
        #     #             for pos in mt5.positions_get():
        #     #                 # if pos.type == 1:  # pos.type == 1 represent a sell order
        #     #                 close_order(pos.ticket)
        #     #         except:
        #     #             pass
        #     #     else:
        #     #         pass
        
        # # OHLC
        # elif results[0] == "sell" and (5.00 >= abs(moment) >= 1.10) : 
            
        #     for i in all_resistance:

        #         if ( results[2] <= i <= results[1] ) :
        #             if not mt5.positions_total():
        #                 try:
        #                     # Open Position
        #                     position, req = market_order(SYMBOL, VOLUME, results[0])
        #                     # Regiter Data
        #                     J = updator(str(datetime.now())[:19], results[0], req['price'], req['sl'], req['tp'], position['comment'], results[3])
        #                     print(
        #                         f"{J['comment']}: {results[0]}, by {results[3]}")

        #                     with open(f'Journal\{str(datetime.now())[:10]}.csv', 'a') as j:
        #                         for v in J.values():
        #                             j.writelines(f"{v},")
        #                         j.writelines("\n")

        #                         print('data registed')

        #                     time.sleep(2 * minutes)

        #                 except:
        #                     print(f"{results[0]} poition didn\' open. somthin\' wnt wrong")
        #                     pass
                        
        #     # for i in range(len(all_resistance)) :
        #     #     if price_now == all_suports[i] :
        #     #             try:
        #     #                 for pos in mt5.positions_get():
        #     #                 # if pos.type == 1:  # pos.type == 1 represent a sell order
        #     #                     close_order(pos.ticket)   
        #     #             except:
        #     #                 pass
                        
        #     #             else:
        #     #                 pass

        time.sleep(1)

        system('cls')

        # break
        
######### END ############

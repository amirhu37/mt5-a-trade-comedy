__name__ = "mtkit"
from typing import List, Literal
import MetaTrader5 as mt5
import numpy as np
import pandas as pd

ordr_dict = {'buy': 0,
             'sell': 1}
time_perid = {
    "1m":          mt5.TIMEFRAME_M1,
    '3m':          mt5.TIMEFRAME_M3,
    '5m':          mt5.TIMEFRAME_M5,
    '15m':         mt5.TIMEFRAME_M15,
    '30m':          mt5.TIMEFRAME_M30,
    '16385m':      mt5.TIMEFRAME_H1,
    '1h':          mt5.TIMEFRAME_H1,
    '4h':         mt5.TIMEFRAME_H4,
    '1d':        mt5.TIMEFRAME_D1,
    '1w':       mt5.TIMEFRAME_W1, }



class NotRunError(Exception):
    def __init__(self) -> None:
        super().__init__()
        pass
    


run = mt5.initialize()
print( "mt5 Conncection Status: ",  mt5.last_error()[1])
if not mt5.initialize() == True :
    raise NotRunError (f"MetaTRader can not Run,\ncheck installation or network Connection or {mt5.last_error()[1]}")










def Symbol_data(sym: str, time_frame: Literal["1m",'3m','5m','15m','30m','16385m','1h','4h','1d','1w'], 
                bar_range: int, method: Literal['all','time', 'open', 'high', 'low', 'close','volume'], Open_candle: bool = False) -> pd.DataFrame | pd.Timestamp | pd.Series:
    """
    excract SYMBOL datas such as:
    ----
    method: 't' or 'time' , 'o' or'open' , 'h' or 'high', 'l' or 'low', 'c' or 'close', 'v' or 'volume'

    open_candle : means should include current open candle or not
    """
    o = 0 if Open_candle == True else 1
    method = method.lower().strip()
    # sym = sym.upper()
    if not time_frame in time_perid.keys() :
        raise KeyError (f"Insert Wrong Time Frame, the correct are {time_perid.keys()}")
    bars = mt5.copy_rates_from_pos(sym, time_perid[time_frame], o, bar_range)
    # print(bars)
    # assert bars , "NO Symbol with This name detectected"
    # if not bars : raise KeyError ("NO Symbol with This name detectected")

    OHLC = {'t': 0, 'o': 1, 'h': 2, 'l': 3, 'c': 4, 'v': 5, 'all' : None,
            'time': 0, 'open': 1, 'high': 2, 'low': 3, 'close': 4, 'volume': 5}
    
    if not method in OHLC.keys(): raise KeyError (f'Wrong OHLC key. Corrorects Are {OHLC.keys()}')

    temp_list = np.array([i for i in bars])

    if method == 'all':
        data = pd.DataFrame([list(temp_list[i]) for i in range(len(temp_list))],
                            columns= ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', '0', '1'])
        # print(data)
        data['Date'] = pd.to_datetime(data['Date'], unit = 's')
        data.drop(['0', '1'], axis = 1, inplace=True)
        return data
  
    elif method in OHLC:
        data = np.array([temp_list[i][OHLC[method]] for i in range(len(temp_list))])
        if OHLC in ['t', 'time']:
            return pd.Series(pd.to_datetime(data, unit = 's'))
        else :
            return pd.Series(data)
        
    else:
        raise KeyError ('unvalid keyword')



def Symbol_info(s: str):
    """
    SYMBOLs can Have difrent pips and spreads.
    maybe I can help you with that.\nBTW rr is Risk/Reward Ratio
    -
    return: spread, point
    """
    DATA: dict = mt5.symbol_info(s)._asdict()
    if not DATA : raise KeyError ("NO Symbol with This name detectected")
    spread = DATA['spread'] 
    pip = DATA['point']


    return spread * pip , pip


# ATR
def _Average_True_Range(symbol: str, time_frame: str, period: int, Open_candle: bool = False) -> float:
    high = Symbol_data(symbol, time_frame, period, "h", Open_candle)
    low = Symbol_data(symbol, time_frame, period, "l", Open_candle)
    close = Symbol_data(symbol, time_frame, period, "c", Open_candle)

    true_range = []
    for i in range(1, len(high) ):
        high_low = high[i] - low[i]
        high_close = abs(high[i] - close[i-1])
        low_close = abs(low[i] - close[i-1])
        true_range.append(max(high_low, high_close, low_close))

    # Calculate AVERAGE
    atr = np.mean(true_range[-period:])
    return round(atr, 2)



def _amount_sl(sym,  type, percent_equity = 1):
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

        sl = Buyyers - amount  

    elif type_ == 'sell':
        sl = Sellers + amount 
    else:
        raise KeyError('KeyWord UnRecugnized !')
    # print(Buyyers, sl, amount)
    return sl

def _amount_tp(sym, type, percent_equity = 1):
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

def _pip_sl(sym,  type, pip_grade = 1):
    # sym = sym.upper()
    type_ = type.lower()
    Buyyers : float = mt5.symbol_info_tick(sym)._asdict()['bid'] # Buyyers
    Sellers : float = mt5.symbol_info_tick(sym)._asdict()['ask'] # Sellers 
    spread, pip = Symbol_info(sym)
    spread *= pip
    # print(amount)
    if type_ == 'buy':
        sl = Buyyers -  (pip * pip_grade)

    elif type_ == 'sell':
        sl = Sellers + (pip * pip_grade)
    else:
        raise KeyError('KeyWord UnRecugnized !')
    # print(Buyyers, sl, amount)
    return sl

def _pip_tp(sym, type, pip_grade = 1):
    # sym = sym.upper()
    type = type.lower()
    Buyyers : float = mt5.symbol_info_tick(sym)._asdict()['bid'] # Buyyers
    Sellers : float = mt5.symbol_info_tick(sym)._asdict()['ask'] # Sellers
    # if not Buyyers : raise KeyError ("NO Symbol with This name detectected")
    # if not Sellers : raise KeyError ("NO Symbol with This name detectected")
    spread, pip = Symbol_info(sym)


    spread *= pip
    if type == 'buy':
        tp = Sellers + (pip * pip_grade)
    elif type == 'sell':
        tp = Buyyers -  (pip * pip_grade)
    else:
        raise KeyError('KeyWord UnRecugnized !')
    
    return tp

# print(pip_sl('XAUUSD', 'buy'))

def _spread_sl(sym,  type, percent = 1):
    sym = sym.upper()
    type_ = type.lower()
    Buyyers : float = mt5.symbol_info_tick(sym)._asdict()['bid'] # Buyyers
    Sellers : float = mt5.symbol_info_tick(sym)._asdict()['ask'] # Sellers

    spread, pip = Symbol_info(sym)

    if type_ == 'buy':

        sl = Buyyers -    (spread *  percent)

    elif type_ == 'sell':
        sl = Sellers +   (spread *  percent)
    else:
        raise KeyError('KeyWord UnRecugnized !')
    # print(Buyyers, sl, amount)
    return sl

def _spread_tp(sym, type, percent = 1):

    sym = sym.upper()
    type = type.lower()
    Buyyers : float = mt5.symbol_info_tick(sym)._asdict()['bid'] # Buyyers
    Sellers : float = mt5.symbol_info_tick(sym)._asdict()['ask'] # Sellers

    spread, pip = Symbol_info(sym)

    # print(Buyyers + (spread * percent) )
    # spread *= pip
    if type == 'buy':
        tp = Sellers + (spread * percent) 

    elif type == 'sell':
        tp = Buyyers - (spread * percent) 
    else:
        raise KeyError('KeyWord UnRecugnized !')
    
    return tp


def _candle_sl(sym,  type, time_frame, candle_count = 1):

    sym = sym.upper()
    type_ = type.lower()

    highs = Symbol_data(sym, time_frame , candle_count, 'h')
    lows = Symbol_data(sym, time_frame , candle_count, 'l')


    spread, pip = Symbol_info(sym)
    spread *= pip
    # print(amount)
    if type_ == 'buy':
        sl = np.mean(lows) 
    elif type_ == 'sell':
        sl = np.mean(highs) 
    else:
        raise KeyError('Sl KeyWord UnRecugnized !')

    return sl


def _candle_tp(sym, type, time_frame, candle_count = 1):

    sym = sym.upper()
    type = type.lower()
    sym = sym.upper()
    type_ = type.lower()

    highs = Symbol_data(sym, time_frame , candle_count, 'h')
    lows = Symbol_data(sym, time_frame , candle_count, 'l')

    spread, pip = Symbol_info(sym)


    spread *= pip
    if type == 'buy':
        tp = np.mean(highs) 

    elif type == 'sell':
        tp = np.mean(lows)  
    else:
        raise KeyError('KeyWord UnRecugnized !')
    
    return tp


def market_order(*, symbol: str, time_frame: Literal['1m', '3m', '5m', '15m', '30m', '1h', '4h', '1d', '1w'],
                 volume: float, order_type: Literal['buy', 'sell'], deviation: int = 1, 
                 sl:  Literal['step', 'str', 'amount', "candle", 'spread', 'pip'] | float | None = "step" ,
                 tp:  Literal['step', 'str', 'amount', "candle", 'spread', 'pip'] | float | None = "step" ,
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
    order_type = order_type.lower().strip()
    if not order_type in ['buy', 'sell']:
        raise KeyError ("Order Type is Not support")
    
    tick = mt5.symbol_info_tick(symbol)

    SPREAD, POINT = Symbol_info(symbol)

    order_dict = {'buy': 0, 'sell': 1}
    price_dict = {'buy': tick.ask, 'sell': tick.bid}
    SPREAD *= POINT
    POINT *= 100

    buy_sl = {
        None : 0.0,
        sl : sl,
        "amount": _amount_sl(symbol, order_type, sl_rate),
        "step"   : price_dict[order_type]  - (POINT* sl_rate)  - SPREAD,
        "atr"   : price_dict[order_type] - _Average_True_Range(symbol , time_frame, sl_rate),
        "candle" : _candle_sl(symbol, order_type, time_frame, sl_rate),
        'spread' : _spread_sl(symbol, order_type, sl_rate),
        'pip'   : _pip_sl(symbol, order_type, sl_rate)
    }
    buy_tp = {
        None : 0.0,
        tp : tp,
        "amount":  _amount_tp(symbol, order_type, tp_rate),
        "step": price_dict[order_type]  + (POINT* tp_rate)  + SPREAD,
        "atr" :  price_dict[order_type] + _Average_True_Range(symbol , time_frame, tp_rate),
        "candle" : _candle_tp(symbol, order_type, time_frame, tp_rate),
        'spread' : _spread_tp(symbol, order_type, sl_rate),
        'pip'   : _pip_tp(symbol, order_type, sl_rate)


    }
    
    sell_sl = {
        None : 0.0,
        sl : sl,
        "amount": _amount_sl(symbol, order_type, sl_rate),
        "step": price_dict[order_type] + SPREAD + (POINT * sl_rate),
        "atr" : price_dict[order_type] + _Average_True_Range(symbol , time_frame, sl_rate),
        "candle" : _candle_sl(symbol, order_type, time_frame, sl_rate),
        'spread' : _spread_sl(symbol, order_type, sl_rate),
        'pip'    : _pip_sl(symbol, order_type, tp_rate)


    }
    sell_tp = {
        None : 0.0,
        tp : tp,
        "amount": _amount_tp(symbol,  order_type, tp_rate),
        "step":price_dict[order_type] - (tp_rate * POINT) + SPREAD,
        "atr"  : price_dict[order_type] - _Average_True_Range(symbol , time_frame, tp_rate),
        "candle" : _candle_tp(symbol, order_type, time_frame, tp_rate),
        'spread' : _spread_tp(symbol, order_type, sl_rate),
        'pip'    : _pip_tp(symbol, order_type, tp_rate)

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



def kill_mt5():
    mt5.shut_down()
######### END ############



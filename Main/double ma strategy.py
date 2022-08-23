import time
import TradeToolKit as KIT
import MetaTrader5 as mt5

SYMBOL = "BTCUSD"
VOLUME =  0.03
TIMEFRAME =  '1m'
PERIOD = 100   

m_avg_1 = moving_average_1 = 24
m_avg_2 = moving_average_2 = 99

ma_1 = 10 * m_avg_1
ma_2 = 10 * m_avg_2   
ma_vol = 200 



RR_RATIO = 2 
DEVIATION = 20

order = KIT.market_order


# 'buy': 0, 'sell': 1
# print(vol)
# print(vol_ma)


print(mt5.symbols_total())

if __name__ == '__main__':

    while True:

        ma_short_term = KIT.moving_average(SYMBOL, TIMEFRAME, moving_average_1 +1 , moving_average_1 ,'open' )
        # print(ma_short_term)
        _1_0 =  KIT.moving_average(SYMBOL, TIMEFRAME, moving_average_1 , moving_average_1 ,'open' )[0]       #ma_short_term[-1]
        _1_1 =  KIT.moving_average(SYMBOL, TIMEFRAME, moving_average_1 +1 , moving_average_1 ,'open' )[0]   # ma_short_term[-2]

        
        vol = KIT.Symbol_data(SYMBOL, TIMEFRAME, 1 , 'v')[-1]
        print(SYMBOL, TIMEFRAME)
        print("volume: ",vol)

        # ma_long_term = KIT.moving_average(SYMBOL, TIMEFRAME,  moving_average_2 + 1 , moving_average_2 ,'open' )
        # print(ma_long_term)

        vol_ma = KIT.moving_average(SYMBOL, TIMEFRAME, 200, 100 ,'volume' )[-1]

        print("vol ma: ",vol_ma)
        
        _2_0 = KIT.moving_average(SYMBOL, TIMEFRAME, moving_average_2 , moving_average_2 ,'open' )[0]    #ma_long_term[-1]
        _2_1 = KIT.moving_average(SYMBOL, TIMEFRAME, moving_average_2 +1 , moving_average_2 ,'open' )[0] #ma_long_term[-2]

        print(f"MA {moving_average_1}: ",  _1_1, _1_0)
        print(f"MA {moving_average_2}: " , _2_1, _2_0)

        if _1_1 <= _2_1 and _1_0 > _2_0 and vol >= vol_ma :
            print('CROSS 1')
            # if we have a BUY signal, close all short positions
            try:
                for pos in mt5.positions_get():
                    if pos.type == 1:
                        # pos.type == 1 represent a sell order
                        KIT.close_order(pos.ticket)
            except:
                pass

            if not mt5.positions_total():
                order(symbol= SYMBOL, volume= VOLUME, order_type= 'buy', deviation= DEVIATION , RATIO = RR_RATIO,
                sl= -1.0, tp = -1.0)

            # _1_1 > _2_1 and _1_0 < _2_0 and vol >= vol_ma

            # _1_1 < _2_1 and _1_0 > _2_0 and vol >= vol_ma
            
        if _1_1 >= _2_1 and _1_0 < _2_0 and vol >= vol_ma :
            print('CROSS 2')
            # if we have a BUY signal, close all short positions
            try:
                for pos in mt5.positions_get():
                    if pos.type == 0:
                        # pos.type == 1 represent a sell order
                        KIT.close_order(pos.ticket)
            except:
                pass

            if not mt5.positions_total():
                order(symbol= SYMBOL, volume= VOLUME, order_type= 'sell', deviation= DEVIATION, RATIO = RR_RATIO,
                sl = -1.0 , tp = -1.0 )

        # print('ok')
        time.sleep(2)


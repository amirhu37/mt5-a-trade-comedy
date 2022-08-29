import time
import TradeToolKit as KIT
import MetaTrader5 as mt5

SYMBOL = "BTCUSD"
VOLUME =  0.03
TIMEFRAME =  '1m'
PERIOD = 100   

rsi_1 = 25
rsi_2 = 100

RR_RATIO = 2 
DEVIATION = 20

order = KIT.market_order


if __name__ == '__main__':

    while True:
        print(SYMBOL, TIMEFRAME)
        ce = KIT.chandelier_exit(SYMBOL, TIMEFRAME, 1, 1.85, 'high')

        rsi1 = KIT.relative_strength_index(SYMBOL, TIMEFRAME, rsi_1 ,'open')

        rsi2 = KIT.relative_strength_index(SYMBOL, TIMEFRAME, rsi_2 ,'open')

        if rsi1[0] <= rsi2[0] and rsi1[1] >  rsi2[1] and ce== 1:

            print('CROSS 1')
            # if we have a BUY signal, close all short positions
            try:
                for pos in mt5.positions_get():
                    KIT.close_order(pos.ticket)
            except:
                pass

            if not mt5.positions_total():
                order(symbol= SYMBOL, volume= VOLUME, order_type= 'buy', deviation= DEVIATION , RATIO = RR_RATIO,
                sl= -1.0, tp = -1.0)

        if rsi1[0] >= rsi2[0] and rsi1[1] <  rsi2[1] and ce== 0:
            # if we have a BUY signal, close all short positions
            try:
                for pos in mt5.positions_get():
                    KIT.close_order(pos.ticket)
            except:
                pass

            if not mt5.positions_total():
                order(symbol= SYMBOL, volume= VOLUME, order_type= 'sell', deviation= DEVIATION, RATIO = RR_RATIO,
                sl = -1.0 , tp = -1.0 )

        # print('ok')
        time.sleep(2)
        # break

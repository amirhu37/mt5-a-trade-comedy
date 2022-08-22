import os
import re
import time
import TradeToolKit as KIT
from MetaTrader5 import initialize , positions_total
from multiprocessing import Process

initialize()

SYMBOL = "BTCUSD"
VOLUME =  0.01
TIMEFRAME =  5
PERIOD = 100   
ma = 30    
RR_RATIO = 2 
DEVIATION = 20

order = KIT.market_order

s = KIT.moving_average(SYMBOL, TIMEFRAME, 100,10 ,'open', )

print(len(s))

def KIT():
    print('KIT') 
    spread, point, _ = KIT.Symbol_info(SYMBOL)
    Tr = KIT.Trend(SYMBOL, TIMEFRAME, ma)
    
    slope ,_ = Tr.trend_fit()
    print(slope)
    line_1, line_2 = Tr.trend_line(ma)

    c2 = KIT.Symbol_data(SYMBOL, TIMEFRAME, 1, 'c')[0]
    print(slope,'\n' ,c2 )
     
    for i,j in list(zip(line_1, line_2)) :
        # Line 1 => KIT  line
        # Line 2 => Proxy line
        if slope > 0: # Positive Slope
            if(c2 > j ) : # bUY
                if not positions_total():
                    order(symbol= SYMBOL, volume= VOLUME, order_type= 'buy', deviation= DEVIATION, SPREAD= spread, POINT= point, RATIO=RR_RATIO )
            
            elif (c2 < i): # SELL
                if not positions_total():
                    order(symbol= SYMBOL, volume= VOLUME, order_type= 'sell', deviation= DEVIATION, SPREAD= spread, POINT= point, RATIO=RR_RATIO )

        elif slope < 0: # Negetive Slope
            if ( c2 > i): # BUY
                if not positions_total():
                    order(symbol= SYMBOL, volume= VOLUME, order_type= 'buy', deviation= DEVIATION, SPREAD= spread, POINT= point, RATIO=RR_RATIO )

            elif (c2 < j ) : #SELL
                if not positions_total():
                    order(symbol= SYMBOL, volume= VOLUME, order_type= 'sell', deviation= DEVIATION, SPREAD= spread, POINT= point, RATIO=RR_RATIO )
            
        
        else:
            pass
        




if __name__ == '__KIT__':

    C = 0
 
    while True:
        break
        print(C)
        KIT()
        # pr1 = Process(target= KIT)
        # pr1.start()

        # pr2 = Process(target= make)
        # pr2.start()

        # pr1.join()
        # pr2.join()

        time.sleep(1)
        os.system('cls')
        C += 1

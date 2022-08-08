import os
import re
import time
import main
from MetaTrader5 import initialize , positions_total
from multiprocessing import Process

initialize()

SYMBOL = "BTCUSD"
VOLUME =  0.01
TIMEFRAME =  1
PERIOD = 100   
ma = 30    
RR_RATIO = 2 
DEVIATION = 20

order = main.market_order

spread, point, _ = main.Symbol_info(SYMBOL)

# def flag():
#     print('flg')
#     new = 0
#     time.sleep(60*TIMEFRAME*(ma//2))
#     new = 1
#     return new



def make():
    print('T line')
    Tr = main.Trend_reg(SYMBOL, TIMEFRAME, ma)
    slopee ,_ = Tr.trend_reg()
    line_1e, line_2e = Tr.trend_line(ma)

    _, c2e= main.Symbol_data(SYMBOL, TIMEFRAME, 2, 'c')
    # time.sleep(10)
    return slopee, c2e, line_1e, line_2e




def Main():
    print('main') 
    slope, c2, line_1, line_2 = make()   

    print(slope,'\n' ,c2 )
     
    for i,j in list(zip(line_1, line_2)) :
        # Line 1 => Main  line
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
        

if __name__ == '__main__':

    C = 0
 
    while True:
        print(C)
        Main()
        # pr1 = Process(target= Main)
        # pr1.start()

        # pr2 = Process(target= make)
        # pr2.start()

        # pr1.join()
        # pr2.join()

        time.sleep(1)
        os.system('cls')
        C += 1

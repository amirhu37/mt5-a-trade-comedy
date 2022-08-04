import os
import time
import main
from MetaTrader5 import initialize , positions_total

initialize()

SYMBOL = "ETHUSD"
VOLUME =  0.01
TIMEFRAME =  1
PERIOD = 150      
RR_RATIO = 2 
DEVIATION = 20
MINUTE = 60
minutes = TIMEFRAME * MINUTE



spread, point, _ = main.Symbol_info(SYMBOL)
order = main.market_order


Tr = main.Trend_np(SYMBOL, TIMEFRAME, 10)

slope, intr = Tr.trend_np()

sup, res = main.SUPPORT_RESISTANCE(SYMBOL, TIMEFRAME, PERIOD, 3, 2).result()

line = Tr.trend_line()

# data = main.Symbol_data('BTCUSD',TIMEFRAME,PERIOD, 'o')

o1 , o2 = main.Symbol_data(SYMBOL, TIMEFRAME, 2, 'o', True)
l1 , l2 = main.Symbol_data(SYMBOL, TIMEFRAME, 2, 'l', True)

h1 , h2 = main.Symbol_data(SYMBOL, TIMEFRAME, 2, 'h', True)
c1 , c2 = main.Symbol_data(SYMBOL, TIMEFRAME, 2, 'c', True)

if __name__ == '__main__':
    counter = 0
    while True:
        print(f"counter : {counter}")
        print(f"spread : {spread} ")
        print(f" slope : {slope}")
        # print( c1, c2 )
        print(f"t line : {line}")
        print(f"sup: {sup} \nres: {res}")

        for i, j in list(zip(line, sup)) :
            # print(f"{i}: {sup}, {j}: {line}")
            if o1 > i and slope < 0 and l1 <= j <= c1 :

                if not positions_total():
                
                    order(symbol= SYMBOL, volume= VOLUME, order_type= 'buy', deviation= DEVIATION, SPREAD= spread, POINT= point, RATIO=RR_RATIO )
            else:
                pass
            
        for i, j in list(zip(line, res)):
            # print(f'{i}: {res}, {j}: {line}')
            if c2 < i and slope > 0 and h1 >= j >= o1:

                if not positions_total():

                    order(symbol= SYMBOL, volume= VOLUME, order_type= 'sell', deviation= DEVIATION, SPREAD= spread, POINT= point, RATIO=RR_RATIO )
            else:
                pass
            


        
        time.sleep(10)
        os.system('cls')
        counter += 1
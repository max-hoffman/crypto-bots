import websocket
from poloniex import Poloniex

from multiprocessing.dummy import Process as Thread
import json

import logging
logger = logging.getLogger(__name__)

class PoloBot():

    def __init__(self, api=None):
        self.api = api
        if not self.api:
            self.api = Poloniex()
            self.tick_count = 0
            print(self.api.returnOrderBook(currencyPair="BTC_ETH", depth="10"))
            self._ws = websocket.WebSocketApp('wss://api2.poloniex.com/',
                                              on_open=self.on_open,
                                              on_message=self.on_message,
                                              on_error=self.on_error,
                                              on_close=self.on_close)

    def on_message(self, ws, message):
        message = json.loads(message)
        if 'error' in message:
            return logger.error(message['error'])
            
        # if message[0] == 1002:
        #     return logger.info('Subscribed to ticker')

        if message[1] == 0:
            return logger.info('Unsubscribed to ticker')

        data = message[2]
        data = [float(dat) for dat in data]
        tick = {'id': data[0],
                        'last': data[1],
                        'lowestAsk': data[2],
                        'highestBid': data[3],
                        'percentChange': data[4],
                        'baseVolume': data[5],
                        'quoteVolume': data[6],
                        'isFrozen': data[7],
                        'high24hr': data[8],
                        'low24hr': data[9]
                        }
        print(tick)
        return

    def on_error(self, ws, error):
        logger.error(error)
        return

    def on_close(self, ws):
        if self._t._running:
            try:
                self.stop()
            except Exception as e:
                logger.exception(e)
            try:
                self.start()
            except Exception as e:
                logger.exception(e)
                self.stop()
        else:
            logger.info("Websocket closed!")

    def on_open(self, ws):
        self._ws.send(json.dumps({'command': 'subscribe', 'channel': 1002}))

    @property
    def status(self):
        """
        Returns True if the websocket is running, False if not
        """
        try:
            return self._t._running
        except:
            return False

    def start(self):
        """ Run the websocket in a thread """
        self._t = Thread(target=self._ws.run_forever)
        self._t.daemon = True
        self._t._running = True
        self._t.start()
        logger.info('Websocket thread started')

    def stop(self):
        """ Stop/join the websocket thread """
        self._t._running = False
        self._ws.close()
        self._t.join()
        logger.info('Websocket thread stopped/joined')

 
if __name__ == "__main__":
    import pprint
    from time import sleep
    logging.basicConfig(level=logging.DEBUG)
    # websocket.enableTrace(True)
    ticker = PoloBot()
    try:
        ticker.start()
        for i in range(3):
            sleep(5)
            pprint.pprint(ticker('USDT_BTC'))
    except Exception as e:
        logger.exception(e)
    ticker.stop()
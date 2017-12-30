import gdax, time
import tensorflow as tf
from model import RLAgent

with tf.Session() as session:

    class myWebsocketClient(gdax.WebsocketClient):
        def on_open(self):
            self.url = "wss://ws-feed.gdax.com/"
            self.products = ["ETH-USD"]
            self.iteration = 0
            self.usd = 10
            self.eth = 0
            self.last_eth = 0
            self.last_state = [self.usd, self.eth, 700, 0]
            print("Lets count the messages!")

        def on_close(self):
            global agent
            print("-- Goodbye! --")
            agent.kill()

        def on_message(self, msg):
            global agent
            self.iteration += 1
            # if 'price' in msg and 'type' in msg:
            #     print ("Message type:", msg["type"],
            #            "\t@ {:.3f}".format(float(msg["price"])))

            # get new trade
            eth_price = float(msg["price"])
            current_net = self.eth * eth_price + self.usd

            # back-propogate with last net
            agent.train(state=self.last_state,
                        target=current_net,
                        iteration=self.iteration)

            # forward pass to predict next action and state
            last_action = agent.action([eth_price, self.eth, self.usd])
            print("last action", last_action)
            self.usd, self.eth = agent.step([eth_price, self.eth, self.usd], last_action)
            self.last_state = [self.usd, self.eth, eth_price, last_action]

    state_size = 4
    output_size = 1
    logs_path = 'logs'
    epochReward = 0
    iteration = 0

    agent = RLAgent(state_size, output_size, session, logs_path)
    session.run(tf.global_variables_initializer())

    wsClient = myWebsocketClient()
    wsClient.start()
    print(wsClient.url, wsClient.products)
    while (wsClient.iteration < 500):
        print ("\niteration =", "{} \n".format(wsClient.iteration))
        time.sleep(1)
    wsClient.close()
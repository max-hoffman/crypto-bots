import gdax, time
from model import RLAgent

with tf.Session() as session:

    class myWebsocketClient(gdax.WebsocketClient, agent):
        def on_open(self):
            self.url = "wss://ws-feed.gdax.com/"
            self.products = ["ETH-USD"]
            self.iteration = 0
            self.agent = agent
            self.usd = 10
            self.eth = 0
            self.last_eth = 0
            self.last_state = [self.usd, self.eth, ]
            print("Lets count the messages!")

            self.last_state = 1

        def on_close(self):
            print("-- Goodbye! --")
            self.agent.kill()

        def on_message(self, msg):
            self.iteration += 1
            # if 'price' in msg and 'type' in msg:
            #     print ("Message type:", msg["type"],
            #            "\t@ {:.3f}".format(float(msg["price"])))

            # get new trade
            eth_price = float(msg["price"])

            # predict current net
            current_net = self.eth * eth_price + self.usd

            # back-propogate with last net
            self.agent.train(self.last_state, current_net, self.iteration)

            # forward pass to predict next action and state
            last_action = self.agent.action([eth_price, self.eth, self.usd])
            self.usd, self.eth = self.agent.step([eth_price, self.eth, self.usd], action)
            self.last_state = [self.usd, self.eth, eth_price, last_action]
    state_size = 20
    action_size = 41
    training_epochs = 5
    terminal = False
    logs_path = 'logs'
    epochReward = 0
    iteration = 0

    agent = RLAgent(state_size, action_size, session, logs_path)
    session.run(tf.initialize_all_variables())

    wsClient = myWebsocketClient()
    wsClient.start()
    print(wsClient.url, wsClient.products)
    while (wsClient.message_count < 500):
        print ("\nmessage_count =", "{} \n".format(wsClient.message_count))
        time.sleep(1)
    wsClient.close()
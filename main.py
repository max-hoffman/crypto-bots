import gdax, time
import tensorflow as tf
import numpy as np
from model import RLAgent
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.rc('axes',edgecolor='grey')

def print_to_console(data, iteration):
    plt.close("all")
    plt.figure(iteration / 200)
    fig, ax = plt.subplots()
    ax.plot(data[:,0], data[:,1])
    ax.set_xlabel("step")
    ax.set_ylabel("value ($)")
    # ax.spines["left"].set_color("white")
    # ax.spines["bottom"].set_color("white")
    # ax.xaxis.label.set_color("white")
    # ax.tick_params(axis="x", colors="white")
    plt.show()
    return 

with tf.Session() as session:

    class myWebsocketClient(gdax.WebsocketClient):
        def on_open(self):
            self.url = "wss://ws-feed.gdax.com/"
            self.products = ["ETH-USD"]
            self.iteration = 0
            self.usd = 10
            self.eth = .05
            self.last_net = self.usd + self.eth * 700
            self.last_state = [self.usd, self.eth, 700, 0]
            self.nets = []
            print("Lets count the messages!")

        def on_close(self):
            global agent
            print("-- Goodbye! --")
            agent.kill()

        def on_message(self, msg):
            global agent
            self.iteration += 1
            if 'price' in msg and 'type' in msg:

                # get new trade
                eth_price = float(msg["price"])
                current_net = self.eth * eth_price + self.usd

                # back-propogate with last net
                agent.train(state=self.last_state,
                            target=current_net,
                            reward=current_net - self.last_net,
                            iteration=self.iteration)

                # forward pass to predict next action and state
                last_action = agent.action([eth_price, self.eth, self.usd])
                self.usd, self.eth = agent.step([self.usd, self.eth, eth_price], last_action)
                self.last_state = [self.usd, self.eth, eth_price, last_action]
                self.last_net = current_net

                self.nets.append([self.iteration, current_net])
                if self.iteration % 200 == 0:
                    if self.iteration % 800 == 0:
                        print_to_console(np.array(self.nets), self.iteration)
                    print("iteration", self.iteration)
                    print("state", self.last_state)
                    print("current net", current_net)
                    print("-----------")
                
    input_size = 4
    output_size = 1
    hidden_size = 32
    cell_layers = 8

    logs_path = "logs"
    epochReward = 0
    iteration = 0

    agent = RLAgent(state_size=input_size,
                    action_size=output_size,
                    hidden_size=hidden_size,
                    cell_layers=cell_layers,
                    session=session,
                    logs_path=logs_path)
    session.run(tf.global_variables_initializer())

    wsClient = myWebsocketClient()
    wsClient.start()
    print(wsClient.url, wsClient.products)
    while (wsClient.iteration < 30000):
        continue
        # print ("\niteration =", "{} \n".format(wsClient.iteration))
        # time.sleep(1)
    wsClient.close()

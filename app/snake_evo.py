import numpy as np
# import whatever makes the snakes go
import copy
import sys
import json


TYPE = {"blank": 0, "food": 1, "you": 2, "obstacle": 3}
WIDTH = 11
HEIGHT = 11


class NN():
    def __init__(self, width=WIDTH, height=HEIGHT):
        # Number of nodes? 
        # We want up-down-left-right, not real veolocity values
        self.FC1 = np.random.normal(0,np.sqrt(2/(WIDTH*HEIGHT+64)),(WIDTH*HEIGHT,64))
        self.bias1 = np.random.normal(0,np.sqrt(2/(WIDTH*HEIGHT+64)),(1,64))
        self.FC2 = np.random.normal(0,np.sqrt(2/(64+64)),(64,64))
        self.bias2 = np.random.normal(0,np.sqrt(2/(64+64)),(1,64))
        self.FC3 = np.random.normal(0,np.sqrt(2/(64+4)),(64,4))
        self.bias3 = np.random.normal(0,np.sqrt(2/(64+4)),(1,4))


    def relu(self, X):
        return X * (X>=0)

    def sigmoid(self, X):
        return 1 / (1 + np.exp(-X)) 

    def softmax(self, X):
        return np.exp(X) / np.sum(np.exp(X))

    def predict_proba(self, X):
        # If you changed the structure, change the prediction
        # We want up-down-left-right, not real veolocity values
        X = np.array(X).reshape((-1,WIDTH*HEIGHT))
        X = X @ self.FC1 + self.bias1
        X = self.relu(X)
        X = X @ self.FC2 + self.bias2
        X = self.relu(X)
        X = X @ self.FC3 + self.bias3
        X = self.softmax(X)
        return X

    def predict(self, X):
        # Our prediction must be different since our action must be different
        # We want up-down-left-right, not real veolocity values
        X = self.predict_proba(X)
        X = np.argmax(X, axis=1).reshape((-1, 1))
        #print(X)
        #sys.stdout.flush()
        return X[0][0]


    def mutate(self, stdev=0.03):
        # If you changed the structure, change the mutation
        self.FC1 += np.random.normal(0, stdev, self.FC1.shape)
        self.FC2 += np.random.normal(0, stdev, self.FC2.shape)
        self.FC3 += np.random.normal(0, stdev, self.FC3.shape)
        self.bias1 += np.random.normal(0, stdev, self.bias1.shape)
        self.bias2 += np.random.normal(0, stdev, self.bias2.shape)
        self.bias3 += np.random.normal(0, stdev, self.bias3.shape)

    def save_weights(self, filename):
        # If you changed the structure, change the saving process
        np.savez(filename, FC1=self.FC1, 
                            FC2=self.FC2, 
                            FC3=self.FC3, 
                            bias1=self.bias1, 
                            bias2=self.bias2, 
                            bias3=self.bias3)

    def load_weights(self, filename):
        # If you changed the structure, change the saving process
        npzfile = np.load(filename)
        self.FC1 = npzfile['FC1']
        self.FC2 = npzfile['FC2']
        self.FC3 = npzfile['FC3']
        self.bias1 = npzfile['bias1']
        self.bias2 = npzfile['bias2']
        self.bias3 = npzfile['bias3']

    def load_brain(self, jsonbrain):
        D = json.loads(jsonbrain)
        self.FC1 = D['FC1']
        self.FC2 = D['FC2']
        self.FC3 = D['FC3']
        self.bias1 = D['bias1']
        self.bias2 = D['bias2']
        self.bias3 = D['bias3']

    def brain_to_json(self):
        return json.dumps({"FC1" : self.FC1.tolist(),
                            "FC2" : self.FC2.tolist(),
                            "FC3" : self.FC3.tolist(),
                            "bias1" : self.bias1.tolist(),
                            "bias2" : self.bias2.tolist(),
                            "bias3" : self.bias3.tolist()})


# Snake representation
def parse_json(json_string, width=WIDTH, height=HEIGHT):
    state = json.loads(json_string)
    #print(state.keys())

    board = np.zeros((width, height))

    food_x = []
    food_y = []

    for f in state["board"]["food"]:
        food_x.append(int(f["x"]))
        food_y.append(int(f["y"]))


    obstacle_x = []
    obstacle_y = []

    for i in range(1, len(state["you"]["body"])):
        obstacle_x.append(state["you"]["body"][i]["x"])
        obstacle_y.append(state["you"]["body"][i]["y"])

    for snek in state["board"]["snakes"]:
        is_head = True
        for seg in snek["body"]:
            if is_head:
                if len(snek["body"]) < len(state["you"]["body"]):
                    food_x.append(seg["x"])
                    food_y.append(seg["y"])
                else:
                    obstacle_x.append(seg["x"])
                    obstacle_y.append(seg["y"])
                is_head = False
            else:
                obstacle_x.append(seg["x"])
                obstacle_y.append(seg["y"])

    board[state["you"]["body"][0]["y"], state["you"]["body"][0]["x"]] = TYPE["you"]

    board[food_y, food_x] = TYPE["food"]

    board[obstacle_y, obstacle_x] = TYPE["obstacle"]

    #print(board)

    return board.flatten()



def run_simulation(network, env, count=20, penalize_angle=False, penalize_displacement=False):
    # network is the particular NN being checked
    # env is a gym construct, replace with the snake environment
    # We want different observations for penalizing

    scores = []
    # Do we want a different default count for training?
    for _ in range(count):
        # Replace with snake-specific reset protocol
        #observation = env.reset()
        score = 0
        while True:
            # New score calculation:
            # score += snake_length
            score += 1
            action = network.predict(observation)[0,0]
            # Rather than get reward, done etc. from the environment, this will be snake-engine specific
            #observation, reward, done, info = env.step(action)
            #if some_kind_of_penalty:
                # score -= value_of_penalty
            # Will need to determine done-ness from snake environment
            if done:
                break
        scores.append(score)
    return np.mean(scores)


def evolve(filename='networks/best_network.npz', penalize_angle=False, penalize_displacement=False):

    # We need to set up the snake environment some other way
    #env = gym.make('CartPole-v1')
    networks = []

    for i in range(100):
        networks.append(NN())

    # May want different number of generations?
    for generation in range(20):
        scores = []
        for network in networks:
            # Same function but with different parameters and environment
            #score = run_simulation(network, env, penalize_angle=penalize_angle, penalize_displacement=penalize_displacement)
            scores.append(score)
        networks = [networks[x] for x in np.argsort(scores)[::-1]]
        scores = np.sort(scores)[::-1]
        new_networks = []
        for i in range(10):
            for j in range(10):
                new_network = copy.deepcopy(networks[i])
                # Why aren't we mutating the first in each row?
                if j > 0:
                    new_network.mutate()
                new_networks.append(new_network)
        networks = new_networks
        print("Generation {}: Best Score={}".format(generation, scores[0]))
        # There is no best possible score, remove this?
        #if scores[0] == 500: #best possible score without penalties
        #    break


    best_network = networks[0]
    if filename:
        best_network.save_weights(filename)
    return best_network


def demonstrate(network):
    # Making the environment and getting observations is different for snakes
    #env = gym.make('CartPole-v1')
    #observation = env.reset()
    score = 0
    while True:
        # Will likely not render as in AI Gym, this will be snake-specific
        #env.render()
        # Replace with our particular score function
        #score += 1
        # Action will be up-down-left-right, not a continuous velocity
        # action = network.predict(observation)[0,0]
        # Stepping through the simulation is different with snakes
        #observation, reward, done, info = env.step(action)
        if done:
            # Close will be snake-specific
            # env.close()
            return score

# if __name__ == '__main__':
#     if len(sys.argv[1:]) > 0:
#         network = NN()
#         network.load_weights(sys.argv[1])
#     else:
#         network = evolve()
#     while True:
#         score = demonstrate(network)
# print("Survived {} steps".format(score))
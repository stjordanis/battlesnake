import numpy as np
# import whatever makes the snakes go
import copy
import sys

class NN():
    def __init__(self):
        # Number of nodes? 
        # We want up-down-left-right, not real veolocity values
        self.FC1 = np.random.normal(0,np.sqrt(2/(4+32)),(4,32))
        self.bias1 = np.random.normal(0,np.sqrt(2/(4+32)),(1,32))
        self.FC2 = np.random.normal(0,np.sqrt(2/(32+16)),(32,16))
        self.bias2 = np.random.normal(0,np.sqrt(2/(32+16)),(1,16))
        self.FC3 = np.random.normal(0,np.sqrt(2/(16+2)),(16,2))
        self.bias3 = np.random.normal(0,np.sqrt(2/(16+2)),(1,2))


    def relu(self, X):
        return X * (X>=0)

    def sigmoid(self, X):
        return 1 / (1 + np.exp(-X)) 

    def softmax(self, X):
        return np.exp(X) / np.sum(np.exp(X))

    def predict_proba(self, X):
        # If you changed the structure, change the prediction
        # We want up-down-left-right, not real veolocity values
        X = np.array(X).reshape((-1,4))
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
        #X = self.predict_proba(X)
        #return np.argmax(X, axis=1).reshape((-1, 1))


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

if __name__ == '__main__':
    if len(sys.argv[1:]) > 0:
        network = NN()
        network.load_weights(sys.argv[1])
    else:
        network = evolve()
    while True:
        score = demonstrate(network)
print("Survived {} steps".format(score))
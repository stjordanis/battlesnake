import numpy as np
import sys
import subprocess
import os
import time
import snake_evo
import copy

hosts = ["8080", "8081", "8082"]
N = 3



def evolve(filename='networks/best_network.npz'):

    networks = []

    for i in range(25):
        networks.append(snake_evo.NN())

    # May want different number of generations?
    for generation in range(50):
        scores = np.zeros(25)
        matches = [x for x in range(25)] + [x for x in range(25)] + [x for x in range(25)]
        matches = np.array(matches)
        np.random.shuffle(matches)
        matches = matches.reshape((25, 3))
        c = 0
        for snake1_id, snake2_id, snake3_id in matches:
            c += 1
            snake1, snake2, snake3 = networks[snake1_id], networks[snake2_id], networks[snake3_id]
            snake1.save_weights('snake1_brain')
            snake2.save_weights('snake2_brain')
            snake3.save_weights('snake3_brain')
            score1, score2, score3 = run_game(snake1, snake2, snake3)
            scores[snake1_id] += score1/3
            scores[snake2_id] += score2/3
            scores[snake3_id] += score3/3
            print(f'{c}/25, g{generation}')
            print(f'{snake1_id}: {score1}')
            print(f'{snake2_id}: {score2}')
            print(f'{snake3_id}: {score3}')
        networks = [networks[x] for x in np.argsort(scores)[::-1]]
        scores = np.sort(scores)[::-1]
        print(scores)
        best_network = networks[0]
        if filename:
            best_network.save_weights(filename)
        new_networks = []
        for i in range(5):
            for j in range(5):
                new_network = copy.deepcopy(networks[i])
                if j > 0:
                    new_network.mutate()
                new_networks.append(new_network)
        networks = new_networks
        print("Generation {}: Best Score={}".format(generation, scores[0]))



    best_network = networks[0]
    if filename:
        best_network.save_weights(filename)
    return best_network


def run_game(snake1, snake2, snake3):
    try:
        s1 = subprocess.Popen(["python", "/Users/conor/Desktop/battlesnake2k19/battlesnake/app/main.py", hosts[0], 'snake1_brain.npz'])
        s2 = subprocess.Popen(["python", "/Users/conor/Desktop/battlesnake2k19/battlesnake/app/main.py", hosts[1], 'snake2_brain.npz'])
        s3 = subprocess.Popen(["python", "/Users/conor/Desktop/battlesnake2k19/battlesnake/app/main.py", hosts[2], 'snake3_brain.npz'])

        game = subprocess.Popen(["make", "-f", "/Users/conor/go/src/github.com/battlesnakeio/engine/Makefile", "run-game"])
        time.sleep(N)
        with open(f'snake_{hosts[0]}.txt', 'r') as f:
            score1 = float(f.readline())
        with open(f'snake_{hosts[1]}.txt', 'r') as f:
            score2 = float(f.readline())
        with open(f'snake_{hosts[2]}.txt', 'r') as f:
            score3 = float(f.readline())

    finally:
        s1.kill()
        s2.kill()
        s3.kill()
        game.kill()

    return score1, score2, score3

if __name__ == '__main__':
    evolve()
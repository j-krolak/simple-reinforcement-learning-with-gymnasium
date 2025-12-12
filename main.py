import gymnasium as gym
import numpy as np
import  matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

MODEL_FILE = "frozen_lake8x8.pkl"

def rl_learn(is_training=True, render=False):
    env = gym.make("FrozenLake-v1",desc=None, map_name="8x8", is_slippery=True, render_mode="human" if render else None)

    if(is_training):
        q = np.zeros((env.observation_space.n, env.action_space.n))
    else:
        f = open(MODEL_FILE, "rb")
        q = pickle.load(f)
        f.close()

    epsilon = 1
    learning_rate_a = 0.9
    discount_factor_g = 0.9
    episods = 15000

    if is_training == False:
        episods = 10

    epsilon_decay_rate = 0.0001
    rng = np.random.default_rng()

    rewards_per_episod = np.zeros(episods)
    for i in tqdm(range(episods)):
        state = env.reset()[0]
        terminated = False
        truncated = False

        while(not terminated and not truncated):
            if rng.random() < epsilon and is_training :
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state, :])
        
            new_state, reward, terminated, truncated, _ = env.step(action)

            if is_training:
                q[state, action] = q[state, action] + learning_rate_a*(reward + discount_factor_g * np.max(q[new_state, :]) - q[state, action])
            
            if reward:
                rewards_per_episod[i] += 1

            state = new_state

        if is_training:
            epsilon = max(epsilon - epsilon_decay_rate, 0)
        if epsilon == 0: 
            learning_rate_a = 0.0001
    
    env.close()
    if not is_training:
        return

    sum_rewards  = np.zeros(episods)
    for t in range(episods):
        sum_rewards[t] = np.sum(rewards_per_episod[max(0, t-100):(t+1)])
    plt.plot(sum_rewards)
    plt.savefig("frozen_lake8x8.png")


    f = open(MODEL_FILE, "wb")
    pickle.dump(q, f)
    f.close()

if __name__ == "__main__":
    is_training = True if input("Is training? [Y/n]: ").strip() == "Y" else False
    render = True if input("Render? [Y/n]: ").strip() == "Y" else False
    rl_learn(is_training, render)











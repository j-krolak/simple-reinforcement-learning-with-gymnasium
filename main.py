import gymnasium as gym
from gymnasium.utils.play import play
import numpy as np
import matplotlib
matplotlib.use('Agg')
import  matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

MODEL_FILE = "frozen_lake8x8.pkl"


def convert_state(state):
    return state[0]  + state[1] * 32 + state[2]*32*11

def rl_learn(is_training=True, render=False, human=False):
    render_mode = None
    if human:
        render_mode = "rgb_array"
    elif render:
        render_mode="human"
    env = gym.make("Blackjack-v1",  render_mode=render_mode)
    
    mapping = { 
        (ord('s'), ): 0,
        (ord('h'), ): 1,
        
    }
    if human == True:
        play(env, keys_to_action=mapping, wait_on_player=True, fps=5)
        return     


    observation_space_n = 32 * 11 * 3
    if(is_training):
        q = np.zeros((observation_space_n , env.action_space.n))
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
        state = convert_state(env.reset()[0])
        terminated = False
        truncated = False

        while(not terminated and not truncated):
            if rng.random() < epsilon and is_training :
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state, :])
        
            new_state, reward, terminated, truncated, _ = env.step(action)

            new_state = convert_state(new_state)
            if is_training:
                q[state, action] = q[state, action] + learning_rate_a*(reward + discount_factor_g * np.max(q[new_state, :]) - q[state, action])
            
            rewards_per_episod[i] += reward

            state = new_state

        if is_training:
            epsilon = max(epsilon - epsilon_decay_rate, 0)
        if epsilon == 0: 
            learning_rate_a = 0.0001
    
    env.close()
    if not is_training:
        return

    window_size = 100
    sum_rewards = np.zeros(episods - window_size + 1)
    
    # --- RYSOWANIE WYKRESU ---
    window_size = 1000 # Większe okno dla czytelności przy 100k epizodów
    sum_rewards = np.zeros(episods - window_size + 1)
    
    current_sum = np.sum(rewards_per_episod[:window_size])
    
    # POPRAWKA: Dzielimy pierwszy element przez window_size
    sum_rewards[0] = current_sum / window_size 
    
    for t in range(1, len(sum_rewards)):
        current_sum = current_sum - rewards_per_episod[t-1] + rewards_per_episod[t+window_size-1]
        sum_rewards[t] = current_sum / window_size

    plt.plot(sum_rewards)
    plt.title(f"Średnia nagroda (okno {window_size})")
    plt.xlabel("Epizody")
    plt.ylabel("Średnia nagroda (-1 przegrana, 0 remis, 1 wygrana)")
    plt.grid(True)
    plt.savefig("blackjack.png")
    print("Wykres zapisany jako blackjack.png")
   
    f = open(MODEL_FILE, "wb")
    pickle.dump(q, f)
    f.close()

if __name__ == "__main__":
    is_training = True if input("Is training? [Y/n]: ").strip() == "Y" else False
    render = True if input("Render? [Y/n]: ").strip() == "Y" else False
    human = True if input("Do you want to control? [Y/n]: ").strip() == "Y" else False
    rl_learn(is_training, render, human)











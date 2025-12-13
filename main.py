import gymnasium as gym
from gymnasium.utils.play import play
import numpy as np
import matplotlib
matplotlib.use('Agg')
import  matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

MODEL_FILE = "blackjack.pkl"
INTERACTIVE = False


def convert_state(state):
    return state[0]  + state[1] * 32 + state[2]*32*11

def get_next_action(env, q, rng, state, epsilon):
    if rng.random() < epsilon:
        return env.action_space.sample()
    else:
         return np.argmax(q[state, :])
        
MAPPING = { 
        (ord('s'), ): 0,
        (ord('h'), ): 1,
}

def rl_learn(mode="Sarsa", chart_filename="blackjack.png") :
    '''
    mode - "Sarsa", "Q-learning", "Human", "Random"
    '''
    match mode:
        case "Human":
            render_mode = "rgb_array" 
        case "Random": 
            render_mode = "human"
        case _:
            render_mode = None


    env = gym.make("Blackjack-v1",  render_mode=render_mode)
    
    if mode == "Human":
        play(env, keys_to_action=MAPPING, wait_on_player=True, fps=5)
        return     


    observation_space_n = 32 * 11 * 3 
    if(mode in ["Sarsa", "Q-learning"]):
        q = np.zeros((observation_space_n , env.action_space.n))
    else:
        f = open(MODEL_FILE, "rb")
        q = pickle.load(f)
        f.close()

    epsilon = 1
    learning_rate_a = 0.001
    discount_factor_g = 0.9
    episodes = 200000

    if mode == "Random":
        episodes = 10

    epsilon_decay_rate = epsilon / (episodes / 2)
    rng = np.random.default_rng()

    rewards_per_episod = np.zeros(episodes)

    for i in tqdm(range(episodes)):
        state = convert_state(env.reset()[0])
        terminated = False
        truncated = False

        action = get_next_action(env,q, rng, state, epsilon) if mode in ["Q-learning", "Sarsa"] else env.action_space.sample()
        while(not terminated and not truncated):
        
            new_state, reward, terminated, truncated, _ = env.step(action)

            new_state = convert_state(new_state)

            next_action = get_next_action(env,q, rng, new_state, epsilon)

            if mode == "Q-learning":
                q[state, action] = q[state, action] + learning_rate_a*(reward + discount_factor_g * np.max(q[new_state, :]) - q[state, action])

            if mode == "Sarsa":
                q[state, action] = q[state, action] + learning_rate_a*(reward + discount_factor_g * q[new_state, next_action] - q[state, action])

            rewards_per_episod[i] += reward

            state = new_state
            action = next_action

        if mode in ["Sarsa", "Q-learning"]:
            epsilon = max(epsilon - epsilon_decay_rate, 0)
        if epsilon == 0: 
            learning_rate_a = 0.0001
    
    env.close()

    if mode in ["Human", "Random"] : 
        return
    
    window_size = 1000 # Większe okno dla czytelności przy 100k epizodów
    sum_rewards = np.zeros(episodes - window_size + 1)
    
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
    plt.savefig(chart_filename)
    print(f"Wykres zapisany jako {chart_filename}")
   
    f = open(MODEL_FILE, "wb")
    pickle.dump(q, f)
    f.close()

if __name__ == "__main__":

    if INTERACTIVE:
        '''
        is_training = True if input("Is training? [Y/n]: ").strip() == "Y" else False
        render = True if input("Render? [Y/n]: ").strip() == "Y" else False
        human = True if input("Do you want to control? [Y/n]: ").strip() == "Y" else False
        sarsa = True if input("Do you want to use SARSA? [Y/n]: ").strip() == "Y" else False
        rl_learn(is_training, render, human, sarsa)
        '''
    else:
        rl_learn( "Q-learning", "q-learning.png")
        rl_learn("Sarsa", "sarsa.png")










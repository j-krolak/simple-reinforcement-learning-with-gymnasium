import gymnasium as gym
from gymnasium.utils.play import play
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle


MODEL_FILE = "blackjack.pkl"
INTERACTIVE = False


def convert_state(state):
    return state[0] + state[1] * 32 + state[2]*32*11


def get_next_action(env, q, rng, state, epsilon):
    if rng.random() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(q[state, :])
        
MAPPING = { 
    (ord('s'), ): 0,
    (ord('h'), ): 1,
}


def rl_learn(mode="Sarsa", chart_filename="blackjack.png"):
    '''
    mode - "Sarsa", "Q-learning", "Human", "Random"
    '''
    match mode:
        case "Human":
            render_mode = "rgb_array" 
        case "Random": 
            render_mode = "rgb_array"
        case _:
            render_mode = None

    env = gym.make("Blackjack-v1", render_mode=render_mode)
    
    if mode == "Human":
        play(env, keys_to_action=MAPPING, wait_on_player=True, fps=5)
        return     

    observation_space_n = 32 * 11 * 3 
    if(mode in ["Sarsa", "Q-learning"]):
        q = np.zeros((observation_space_n, env.action_space.n))
    else:
        f = open(MODEL_FILE, "rb")
        q = pickle.load(f)
        f.close()

    epsilon = 1
    learning_rate_a = 0.001
    discount_factor_g = 0.9
    episodes = 200000

    if mode == "Random":
        episodes = 10000

    epsilon_decay_rate = epsilon / (episodes / 2)
    rng = np.random.default_rng()

    rewards_per_episod = np.zeros(episodes)
    # Nowe tablice dla win rate
    wins_per_episode = np.zeros(episodes)
    draws_per_episode = np.zeros(episodes)
    losses_per_episode = np.zeros(episodes)

    for i in tqdm(range(episodes)):
        state = convert_state(env.reset()[0])
        terminated = False
        truncated = False

        action = get_next_action(env, q, rng, state, epsilon) if mode in ["Q-learning", "Sarsa"] else env.action_space.sample()
        while(not terminated and not truncated):
            new_state, reward, terminated, truncated, _ = env.step(action)

            new_state = convert_state(new_state)
            
            if mode == 'Random':
                next_action = env.action_space.sample()
            else:
                next_action = get_next_action(env, q, rng, new_state, epsilon)

            if mode == "Q-learning":
                q[state, action] = q[state, action] + learning_rate_a*(reward + discount_factor_g * np.max(q[new_state, :]) - q[state, action])

            if mode == "Sarsa":
                q[state, action] = q[state, action] + learning_rate_a*(reward + discount_factor_g * q[new_state, next_action] - q[state, action])

            rewards_per_episod[i] += reward

            state = new_state
            action = next_action

        # Zliczanie wyników
        if rewards_per_episod[i] > 0:
            wins_per_episode[i] = 1
        elif rewards_per_episod[i] == 0:
            draws_per_episode[i] = 1
        else:
            losses_per_episode[i] = 1

        if mode in ["Sarsa", "Q-learning"]:
            epsilon = max(epsilon - epsilon_decay_rate, 0)
        if epsilon == 0: 
            learning_rate_a = 0.0001
    
    env.close()

    if mode == "Human":
        return
    
    if mode == "Random": 
        total_wins = np.sum(wins_per_episode)
        total_draws = np.sum(draws_per_episode)
        total_losses = np.sum(losses_per_episode)
    
        print(f"\n=== Statystyki dla {mode} ({episodes} epizodów) ===")
        print(f"Win Rate: {(total_wins / episodes) * 100:.2f}%")
        print(f"Draw Rate: {(total_draws / episodes) * 100:.2f}%")
        print(f"Loss Rate: {(total_losses / episodes) * 100:.2f}%")
        print(f"Średnia nagroda: {np.mean(rewards_per_episod):.4f}")
        return
    
    window_size = 1000
    sum_rewards = np.zeros(episodes - window_size + 1)
    win_rate = np.zeros(episodes - window_size + 1)
    
    current_sum = np.sum(rewards_per_episod[:window_size])
    sum_rewards[0] = current_sum / window_size
    
    # Obliczanie win rate dla pierwszego okna
    current_wins = np.sum(wins_per_episode[:window_size])
    win_rate[0] = (current_wins / window_size) * 100
    
    for t in range(1, len(sum_rewards)):
        current_sum = current_sum - rewards_per_episod[t-1] + rewards_per_episod[t+window_size-1]
        sum_rewards[t] = current_sum / window_size
        
        # Aktualizacja win rate
        current_wins = current_wins - wins_per_episode[t-1] + wins_per_episode[t+window_size-1]
        win_rate[t] = (current_wins / window_size) * 100

    # Tworzenie wykresu z dwoma subplot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Wykres średniej nagrody
    ax1.plot(sum_rewards)
    ax1.set_title(f"Średnia nagroda (okno {window_size})")
    ax1.set_xlabel("Epizody")
    ax1.set_ylabel("Średnia nagroda")
    ax1.grid(True)
    
    # Wykres win rate
    ax2.plot(win_rate, color='green')
    ax2.set_title(f"Win Rate (okno {window_size})")
    ax2.set_xlabel("Epizody")
    ax2.set_ylabel("Win Rate (%)")
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(chart_filename)
    print(f"Wykres zapisany jako {chart_filename}")
    
    # Statystyki końcowe
    final_win_rate = (np.sum(wins_per_episode[-10000:]) / 10000) * 100
    final_draw_rate = (np.sum(draws_per_episode[-10000:]) / 10000) * 100
    final_loss_rate = (np.sum(losses_per_episode[-10000:]) / 10000) * 100
    
    print(f"\n=== Statystyki dla {mode} (ostatnie 10000 epizodów) ===")
    print(f"Win Rate: {final_win_rate:.2f}%")
    print(f"Draw Rate: {final_draw_rate:.2f}%")
    print(f"Loss Rate: {final_loss_rate:.2f}%")
    print(f"Średnia nagroda: {np.mean(rewards_per_episod[-10000:]):.4f}")
   
    f = open(MODEL_FILE, "wb")
    pickle.dump(q, f)
    f.close()


if __name__ == "__main__":
    if INTERACTIVE:
        pass
    else:
        rl_learn("Q-learning", "q-learning.png")
        rl_learn("Sarsa", "sarsa.png")
        rl_learn("Random")

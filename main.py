import gymnasium as gym
from gymnasium.utils.play import play
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import os.path



def convert_state_to_number(state):
    return state[0] + state[1] * 32 + state[2]*32*11

def get_next_action(env, q, rng, state, epsilon):
    if rng.random() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(q[state, :])
        
def save_model(q, filename):
    f = open(os.path.join("models", filename), "wb")
    pickle.dump(q, f)
    f.close()
    
def load_model(filename):
    f = open(filename, "rb")
    q = pickle.load(f)
    f.close()
    return q

MAPPING = { 
    (ord('s'), ): 0,
    (ord('h'), ): 1,
}

class RL_Model:
    def __init__(self, model_type="Sarsa"):
        self.model_type = model_type
        self.rewards_per_episod = np.zeros(0)
        
    def stats(self, chart_filename):
        window_size = 1000
        wins_per_episode = np.zeros(self.episodes)
        draws_per_episode = np.zeros(self.episodes)
        losses_per_episode = np.zeros(self.episodes)

        for i in range(len(self.rewards_per_episod)):
            if self.rewards_per_episod[i] > 0:
                wins_per_episode[i] = 1
            elif self.rewards_per_episod[i] == 0:
                draws_per_episode[i] = 1
            else:
                losses_per_episode[i] = 1


        sum_rewards = np.zeros(self.episodes - window_size + 1)
        win_rate = np.zeros(self.episodes - window_size + 1)
        
        current_sum = np.sum(self.rewards_per_episod[:window_size])
        sum_rewards[0] = current_sum / window_size


        # Calculate a win rate for the first window
        current_wins = np.sum(wins_per_episode[:window_size])
        win_rate[0] = (current_wins / window_size) * 100
        
        for t in range(1, len(sum_rewards)):
            current_sum = current_sum - self.rewards_per_episod[t-1] + self.rewards_per_episod[t+window_size-1]
            sum_rewards[t] = current_sum / window_size
            
            # Update win rate
            current_wins = current_wins - wins_per_episode[t-1] + wins_per_episode[t+window_size-1]
            win_rate[t] = (current_wins / window_size) * 100

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        # Mean reward chart
        ax1.plot(sum_rewards)
        ax1.set_title(f"Średnia nagroda (okno {window_size})")
        ax1.set_xlabel("Epizody")
        ax1.set_ylabel("Średnia nagroda")
        ax1.grid(True)
        
        # Win rate chart
        ax2.plot(win_rate, color='green')
        ax2.set_title(f"Win Rate (okno {window_size})")
        ax2.set_xlabel("Epizody")
        ax2.set_ylabel("Win Rate (%)")
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(chart_filename)
        print(f"Wykres zapisany jako {chart_filename}")
        
        # Final stats
        final_win_rate = (np.sum(wins_per_episode[-10000:]) / 10000) * 100
        final_draw_rate = (np.sum(draws_per_episode[-10000:]) / 10000) * 100
        final_loss_rate = (np.sum(losses_per_episode[-10000:]) / 10000) * 100
        
        print(f"\n=== Statystyki dla {self.model_type} (ostatnie 10000 epizodów) ===")
        print(f"Win Rate: {final_win_rate:.2f}%")
        print(f"Draw Rate: {final_draw_rate:.2f}%")
        print(f"Loss Rate: {final_loss_rate:.2f}%")
        print(f"Średnia nagroda: {np.mean(self.rewards_per_episod[-10000:]):.4f}")
       
    def train(self, model_path = None):
        '''
        model - "Sarsa", "Q-learning", "Human", "Random"
        '''
        rng = np.random.default_rng()

        render_mode = "rgb_array" if self.model_type in ["Human", "Random"] else None
        env = gym.make("Blackjack-v1", render_mode=render_mode)
        
        if self.model_type == "Human":
            play(env, keys_to_action=MAPPING, wait_on_player=True, fps=5)
            return     

        # Hyperparameters
        epsilon = 1
        learning_rate_a = 0.001
        discount_factor_g = 0.9
        episodes = 200000
        epsilon_decay_rate = epsilon / (episodes / 2)
        self.episodes = episodes

        observation_space_n = 32 * 11 * 3 
        if model_path == None:
            q = np.zeros((observation_space_n, env.action_space.n))
        else:
            q = load_model(model_path)

        rewards_per_episod = np.zeros(episodes)

        for i in tqdm(range(episodes)):
            state = convert_state_to_number(env.reset()[0])
            terminated = False
            truncated = False

            action = get_next_action(env, q, rng, state, epsilon) if self.model_type in ["Q-learning", "Sarsa"] else env.action_space.sample()
            while(not terminated and not truncated):
                new_state, reward, terminated, truncated, _ = env.step(action)

                new_state = convert_state_to_number(new_state)
                
                if self.model_type == 'Random':
                    next_action = env.action_space.sample()
                else:
                    next_action = get_next_action(env, q, rng, new_state, epsilon)

                if self.model_type == "Q-learning":
                    q[state, action] = q[state, action] + learning_rate_a*(reward + discount_factor_g * np.max(q[new_state, :]) - q[state, action])

                if self.model_type == "Sarsa":
                    q[state, action] = q[state, action] + learning_rate_a*(reward + discount_factor_g * q[new_state, next_action] - q[state, action])

                rewards_per_episod[i] += reward

                state = new_state
                action = next_action

            epsilon = max(epsilon - epsilon_decay_rate, 0)

            if epsilon == 0: 
                learning_rate_a = 0.0001

            if i % (episodes//3) == 0 and i != 0:
                j = i // (episodes//3)
                save_model(q, f"model-{self.model_type}-level-{j}.pkl")
        
        env.close()

        self.rewards_per_episod = rewards_per_episod
        save_model(q, f"model-{self.model_type}-level-3.pkl")


if __name__ == "__main__":
    sarsa = RL_Model("Sarsa")
    q_learning = RL_Model("Q-learning")
    random_agent = RL_Model("Random")

    sarsa.train()
    sarsa.stats("sarsa.png")

    q_learning.train()
    sarsa.stats("q-learning.png")

    random_agent.train()
    random_agent.stats("random_agent.png")

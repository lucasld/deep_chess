from collections import deque
import numpy as np
from random import shuffle

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.models import Model

#from pettingzoo.classic import tictactoe_v3
from pettingzoo.classic.chess import chess


class Environment:
    def __init__(self):
        # TODO: add seed!!
        self.env = chess.env()
        self.pit_number = 30
        # list that accumulates all actions taken to create on trajectory
        self.action_acc = []
        self.reset_env()
    

    def reset_env(self, seed=np.random.randint(0, 1e9)):
        self.env.reset(seed=seed)
        self.agent_iterator = iter(self.env.agent_iter())
        self.agent = next(self.agent_iterator)
        self.observation, self.reward, self.termination,\
        self.truncation, self.info = self.env.last()
        self.action_acc = []
    
    
    def execute_step(self, action):
        self.env.step(action)
        self.agent = next(self.agent_iterator)
        self.observation, self.reward, self.termination,\
        self.truncation, self.info = self.env.last()
        # add action to action accumulator
        self.action_acc.append(action)

    
    def is_terminal(self):
        """Returns true if game is either truncated or terminated.
        
        :return: if game is terminated true, else false
        :rtype: boolean
        """
        return self.truncation or self.termination


    def pit(self, agent1, agent2):
        """Pits to agents against each other self.pit_number times.
        
        :param agent1: agent 1, this is get_action_probs() function of
            a MCTS-oject
        :type agent1: get_action_probs() function of MCTS-object
        :param agent2: agent 2, this is get_action_probs() function of
            a MCTS-oject
        :type agent2: get_action_probs() function of MCTS-object"""
        agents = (agent1, agent2)
        agent1_wins, agent2_wins, draws = 0, 0, 0
        for pit_i in range(self.pit_number):
            # for each pit flip which agent starts first
            # play a game between the two agents
            self.reset_env()
            player = pit_i%2
            while self.truncation and self.termination:
                pi = agents[player].get_action_probs()
                mask = self.observation['action mask']
                masked_pi = pi
                masked_pi[mask==0] = 0
                # take action with highest action value
                action = np.argmax(masked_pi)
                agents[player].execute_step(action)
                player = int(not player)
                #player = next(self.agent_iterator) TODO
            # add win to resictive win-counter (or draw-counter)
            if self.reward == 0:
                draws += 1
            elif player == 1:
                agent1_wins += 1
            elif player == 0:
                agent2_wins += 1
        return agent1_wins, agent2_wins, draws


    def get_state_string(self):
        """Creates string representation of current board state."""
        return str(self.observation['observation'])


    def create_copy(self):
        """Function that creates a copy of the environment itself."""
        env_copy = Environment()
        for action in self.action_acc:
            env_copy.execute_step(action)
        return env_copy



class MCTS:
    def __init__(self, env, network, num_traverses=2):
        self.env = env
        self.network = network
        self.num_traverses = num_traverses

        self.c=np.sqrt(2)  # ucb factor
        
        self.Qsa = {}  # stores the Q value for edge (s, a)
        self.Nsa = {}  # number of visits of edge (s, a)
        self.Ns = {}  # number of visists state s was visited
        self.Ps = {}  # stores policy for state s
    

    def get_action_probs(self):  #TODO: this function should not receive input as it should use self.env
        # current state string representation
        state_string = self.env.get_state_string()
        # traverse tree a number of times starting from self.env's current
        # state and by this, create tree
        for _ in range(self.num_traverses):
            print("traverse:", _)
            env_temp = self.env.create_copy()
            self.search(env_temp)
        # TODO: add temperature
        # calculate move probabilities
        sa_counts = []
        for action in range(len(self.env.observation["action_mask"])):
            if (state_string, action) in self.Nsa.keys():
                sa_counts.append(self.Nsa[(state_string, action)])
            else:
                sa_counts.append(0)
        total_visits = sum(sa_counts)
        move_probs = [x/total_visits for x in sa_counts]
        return move_probs


    def search(self, env):
        s = env.get_state_string()
        # check if game is in terminal state
        if env.is_terminal():
            return -env.reward
        # check if state is a leaf node
        if s not in self.Ps:
            self.Ps[s], v = self.network(env.observation["observation"].reshape(1, 8, 8, 111))  # TODO: check if calling network like this works
            self.Ps[s] = self.Ps[s][0]
            v = v[0]
            # masking invalid moves
            legal_moves = env.observation["action_mask"]
            self.Ps[s] = self.Ps[s] * legal_moves
            # normalizing values
            sum_p = sum(self.Ps[s])
            if sum_p > 0:  #TODO: check else for this if...
                self.Ps[s] /= sum_p
            self.Ns[s] = 0
            return -v
        # pick the action with the highest upper confidence bound
        ucb_best = -np.inf
        a_best = -1
        for a, action_prob in enumerate(self.Ps[s]):
            if env.observation['action_mask'][a] > 0:
                # calc ucb  TODO: find out what formula to use here
                if (s, a) in self.Qsa:
                    ucb = self.Qsa[(s, a)] + self.c * action_prob * np.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
                else:
                    ucb = self.c * action_prob * np.sqrt(self.Ns[s] + 1e-8)
                if ucb > ucb_best:
                    ucb_best = ucb
                    a_best = a
        a = a_best
        # take action with best ucb
        env.execute_step(a)
        # retrieve value
        v = self.search(env)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1
        
        self.Ns[s] += 1
        return -v
    

    def reset(self):
        """Resets the tree which means that it clears all dictonaries"""
        self.Qsa = {}  
        self.Nsa = {}
        self.Ns = {}
        self.Ps = {}



def create_value_policy_network():
    input_layer = tf.keras.Input(shape=(8, 8, 111))  # TODO: 111 instead of 20?

    # Convolutional layers
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)

    # Flatten the output of the convolutional layers
    x = Flatten()(x)

    # Fully connected layers
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)

    # Output layers for policy and value estimates
    policy_output = Dense(4672, activation='softmax', name='policy_output')(x)
    value_output = Dense(1, activation='tanh', name='value_output')(x)

    model = Model(inputs=input_layer, outputs=[policy_output, value_output])
    return model


class AlphaZero:
    def __init__(self, env, nnet):
        self.env = env
        self.nnet = nnet
        # copy of nnet, used to compete against new network
        self.pnet = tf.keras.models.clone_model(nnet)
        # path where weights are saved
        self.weight_path = './checkpoints/my_checkpoint'
        self.mcts = MCTS(self.env, self.nnet)
        # number of self-play steps
        self.num_self_play = 100
        # max amount of train examples that are saved
        self.max_self_play_examples = 100
        # example history
        self.max_examples = 100
        self.example_hist = []
        # threshold prob at which nnet will be chosen as new network
        self.win_prob_threshold = 0.55
    
    def learn(self, num_iter=1000):
        # repeat self-play-train self.num_iter times
        for i in range(num_iter):
            print("i:", i)
            sp_examples = deque([], maxlen=self.max_self_play_examples)
            # self-play self.num_self_play times
            for spi in range(self.num_self_play):
                print(spi)
                # reset the monte carlo tree
                self.mcts.reset()
                new_train_examples = self.execute_episode()
                sp_examples += new_train_examples
            # add self play examples to example hist
            self.example_hist.append(sp_examples)
            # remove oldest example if there are to many examples
            if len(self.example_hist) > self.max_examples:
                self.example_hist.pop(0)
            train_examples = []
            # shuffle examples
            for e in self.example_hist:
                train_examples.extend(e)
            shuffle(train_examples)

            # save nnet weights
            self.nnet.save_weights(self.weight_path)
            # load weights into pnet
            self.pnet.load_weights(self.weight_path)
            
            # train neural network
            self.nnet.train(train_examples)

            # create mcts-object for both neural networks
            nmcts = MCTS(self.env, self.nnet)
            pmcts = MCTS(self.env, self.pnet)
            
            # pit new policies against each other
            n_wins, p_wins, draws = self.env.pit(
                nmcts.get_action_probs,
                pmcts.get_action_probs
            )
            if n_wins / (n_wins + p_wins) >= self.win_prob_threshold:
                self.nnet.save_weights(self.weight_path)
            else:
                self.nnet.load_weights(self.weight_path)

    
    def execute_episode(self):
        experience_replay = []
        self.env.reset_env()
        # repeat until game ended
        while not self.env.is_terminal():
            pi = self.mcts.get_action_probs()
            experience_replay.append((self.env.observation['observation'], pi, None))


            # choose action based on the policy - only legal actions
            action = np.random.choice(len(pi), p=pi)
            self.env.execute_step(action)

        for player_i, (observation, pi, _) in enumerate(experience_replay):
            # flip player reward when looking at opponent
            reward = self.env.reward
            r = reward if player_i%2 else -reward
            experience_replay[player_i] = (observation, pi, r)
        return experience_replay



env = Environment()
nnet = create_value_policy_network()
learner = AlphaZero(env, nnet)
learner.learn()
from collections import deque
from random import shuffle


class Environment:
    def __init__(self):
        pass

    def pit(self, agent1, agent2):
        pass


class MCTS:
    def __init__(self, env, network):
        pass

    def get_action_probs(self):
        pass


class NeuralNetwork:
    pass


class AlphaZero:
    def __init__(self, env, nnet):
        self.env = env
        self.nnet = nnet
        # copy of nnet, used to compete against new network
        self.pnet = tf.keras.models.clone_model(nnet)
        # path where weights are saved
        self.weight_path = './checkpoints/my_checkpoint'
        self.mcts = MCTS(self.env, self.nnet)
        # number of iterations
        self.num_iter = 1000
        # number of self-play steps
        self.num_self_play = 100
        # max amount of train examples that are saved
        self.max_self_play_examples = 100
        # example history
        self.max_examples = 100
        self.example_hist = []
        # threshold prob at which nnet will be chosen as new network
        self.win_prob_threshold = 0.55
    
    def learn(self):
        # repeat self-play-train self.num_iter times
        for i in range(self.num_iter):
            sp_examples = deque([], maxlen=self.max_self_play_examples)
            # self-play self.num_self_play times
            for spi in range(self.num_self_play):
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
                lambda x: nmcts.get_action_probs(x),
                lambda x: pmcts.get_action_probs(x)
            )
            if n_wins / (n_wins + p_wins) >= self.win_prob_threshold:
                self.nnet.save_weights(self.weight_path)
            else:
                self.nnet.load_weights(self.weight_path)

    
    def execute_episode(self):
        train_examples = []
        self.env.reset()
        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = self.env.env.last()
            pi = self.mcts.get_action_probs(observation)
            train_examples.append()




env = Environment()
nnet = NeuralNetwork()

learner = AlphaZero(env, nnet)
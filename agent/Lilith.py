from torch import optim

from agent.DuelingDDQN import DuelingDDQN
from agent.FCDuelingQ import FCDuelingQ
from agent.ReplayBuffer import ReplayBuffer
from agent.strategies.EGreedyExpStrategy import EGreedyExpStrategy
from agent.strategies.GreedyStrategy import GreedyStrategy


class Lilith:
    def __init__(self):
        value_model_fn = lambda nS, nA: FCDuelingQ(nS, nA, hidden_dims=(128, 128, 32))
        value_optimizer_fn = lambda net, lr: optim.RMSprop(net.parameters(), lr=lr)
        training_strategy_fn = lambda: EGreedyExpStrategy(init_epsilon=1.0,
                                                          min_epsilon=0.3,
                                                          decay_steps=100)
        evaluation_strategy_fn = lambda: GreedyStrategy()
        replay_buffer_fn = lambda: ReplayBuffer(max_size=500, batch_size=64)

        self.agent = DuelingDDQN(replay_buffer_fn,
                                 value_model_fn,
                                 value_optimizer_fn,
                                 training_strategy_fn,
                                 evaluation_strategy_fn,
                                 value_optimizer_lr=0.001,
                                 max_gradient_norm=float('inf'),
                                 n_warmup_batches=5,
                                 update_target_every_steps=1,
                                 tau=0.1)

    def train(self, env):
        final_eval_score = self.agent.train(gamma=0.99, max_episodes=1000, env=env)
        print(final_eval_score)

    def demonstration(self, env):
        result, history = self.agent.evaluate_once(self.agent.online_model, step_limit=500, eval_env=env)
        return history

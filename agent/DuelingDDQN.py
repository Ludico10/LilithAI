import gc

import numpy as np
import torch
from itertools import count


class DuelingDDQN:
    def __init__(self,
                 replay_buffer_fn,
                 value_model_fn,
                 value_optimizer_fn,
                 training_strategy_fn,
                 evaluation_strategy_fn,
                 value_optimizer_lr,
                 max_gradient_norm,
                 n_warmup_batches,
                 update_target_every_steps,
                 tau):
        self.episode_exploration = None
        self.episode_reward = None
        self.episode_timestep = None

        self.gamma = 1
        self.evaluation_strategy = None
        self.training_strategy = None
        self.replay_buffer = None
        self.value_optimizer = None
        self.online_model = None
        self.target_model = None

        self.replay_buffer_fn = replay_buffer_fn
        self.value_model_fn = value_model_fn
        self.value_optimizer_fn = value_optimizer_fn
        self.value_optimizer_lr = value_optimizer_lr
        self.max_gradient_norm = max_gradient_norm
        self.training_strategy_fn = training_strategy_fn
        self.evaluation_strategy_fn = evaluation_strategy_fn
        self.n_warmup_batches = n_warmup_batches
        self.update_target_every_steps = update_target_every_steps
        self.tau = tau

    def optimize_model(self, experiences):
        states, actions, rewards, next_states, is_terminals = experiences
        batch_size = len(is_terminals)

        argmax_a_q_sp = self.online_model(next_states).max(1)[1]
        q_sp = self.target_model(next_states).detach()
        max_a_q_sp = q_sp[
            np.arange(batch_size), argmax_a_q_sp].unsqueeze(1)
        target_q_sa = rewards + (self.gamma * max_a_q_sp * (1 - is_terminals))
        q_sa = self.online_model(states).gather(1, actions)

        td_error = q_sa - target_q_sa
        value_loss = td_error.pow(2).mul(0.5).mean()
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_model.parameters(),
                                       self.max_gradient_norm)
        self.value_optimizer.step()

    def interaction_step(self, state, env):
        action = self.training_strategy.select_action(self.online_model, state)
        new_state, reward, is_terminal, info = env.step(action)
        is_truncated = 'TimeLimit.truncated' in info and info['TimeLimit.truncated']
        is_failure = is_terminal and not is_truncated
        experience = (state, action, reward, new_state, float(is_failure))
        self.replay_buffer.store(experience)
        self.episode_reward[-1] += reward
        self.episode_timestep[-1] += 1
        self.episode_exploration[-1] += int(self.training_strategy.exploratory_action_taken)
        return new_state, is_terminal

    def update_network(self, tau=None):
        tau = self.tau if tau is None else tau
        for target, online in zip(self.target_model.parameters(),
                                  self.online_model.parameters()):
            target_ratio = (1.0 - tau) * target.data
            online_ratio = tau * online.data
            mixed_weights = target_ratio + online_ratio
            target.data.copy_(mixed_weights)

    def train(self, gamma, max_episodes, env):

        nS, nA = env.observation_space, env.action_space
        self.gamma = gamma
        self.episode_timestep = []
        self.episode_reward = []
        self.episode_exploration = []

        self.target_model = self.value_model_fn(nS, nA)
        self.online_model = self.value_model_fn(nS, nA)
        self.update_network(tau=0.2)

        self.value_optimizer = self.value_optimizer_fn(self.online_model,
                                                       self.value_optimizer_lr)

        self.replay_buffer = self.replay_buffer_fn()
        self.training_strategy = self.training_strategy_fn()
        self.evaluation_strategy = self.evaluation_strategy_fn()

        for episode in range(1, max_episodes + 1):

            state, is_terminal = env.reset(), False
            self.episode_reward.append(0.0)
            self.episode_timestep.append(0.0)
            self.episode_exploration.append(0.0)

            for _ in count():
                state, is_terminal = self.interaction_step(state, env)

                min_samples = self.replay_buffer.batch_size * self.n_warmup_batches
                if len(self.replay_buffer) > min_samples:
                    experiences = self.replay_buffer.sample()
                    experiences = self.online_model.load(experiences)
                    self.optimize_model(experiences)

                if np.sum(self.episode_timestep) % self.update_target_every_steps == 0:
                    self.update_network()

                if is_terminal:
                    gc.collect()
                    break

            print(str(self.episode_reward[-1]) + '   ' +
                  str(self.episode_timestep[-1]) + '   ' +
                  str(self.episode_exploration))

        final_eval_score = self.evaluate(self.online_model, 10 * max_episodes, env, n_episodes=10)
        print('Training complete.')

        return final_eval_score

    def evaluate(self, eval_policy_model, step_limit, eval_env, n_episodes=1):
        rs = []
        for _ in range(n_episodes):
            gc.collect()
            r, _ = self.evaluate_once(eval_policy_model, step_limit, eval_env)
            rs.append(r)
        return np.mean(rs)

    def evaluate_once(self, eval_policy_model, step_limit, eval_env):
        state, d = eval_env.reset(), False
        reward_sum = 0
        state_history = [(state[len(state) // 2:], reward_sum)]
        for i in count():
            a = self.evaluation_strategy.select_action(eval_policy_model, state)
            state, r, d, _ = eval_env.step(a)
            reward_sum += r
            state_history.append((state[len(state) // 2:], reward_sum))
            if d or i > step_limit:
                break
        return reward_sum, state_history

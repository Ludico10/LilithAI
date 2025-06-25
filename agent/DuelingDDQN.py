import gc

import tensorflow as tf
import numpy as np

from agent.DuelingCNN import DuelingCNN
from agent.PrioritizedReplayBuffer import PrioritizedReplayBuffer
from agent.strategies.EGreedyExpStrategy import EGreedyExpStrategy
from agent.strategies.GreedyStrategy import GreedyStrategy


class DuelingDDQN:
    def __init__(self, outputs, name="Lilith"):
        self.online_model = DuelingCNN([128, 64, 64], outputs)
        self.target_model = DuelingCNN([128, 64, 64], outputs)
        self.target_model.set_weights(self.online_model.get_weights())

        self.buffer = PrioritizedReplayBuffer(10000)
        self.train_strat = EGreedyExpStrategy()
        self.test_strat = GreedyStrategy()

        self.optimizer = None
        self.loss_fn = None

        self.actions_cnt = outputs

    def compile(self, optimizer, loss_fn):
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    @staticmethod
    def _split_transitions(transitions):
        states = tf.convert_to_tensor([t[0] for t in transitions], dtype=tf.float32)
        actions = tf.convert_to_tensor([t[1] for t in transitions], dtype=tf.int32)
        rewards = tf.convert_to_tensor([t[2] for t in transitions], dtype=tf.float32)
        next_states = tf.convert_to_tensor([t[3] for t in transitions], dtype=tf.float32)
        dones = tf.convert_to_tensor([t[4] for t in transitions], dtype=tf.float32)
        return states, actions, rewards, next_states, dones

    def train_step(self, exp, gamma_value=0.995):
        transitions, indices, weights = exp
        states, actions, rewards, next_states, dones = self._split_transitions(transitions)
        weights = tf.convert_to_tensor(weights, dtype=tf.float32)

        with tf.GradientTape() as tape:
            q = self.online_model(states)
            actual_q = tf.reduce_sum(tf.one_hot(actions, self.actions_cnt) * q, axis=1)

            next_q_online = self.online_model(next_states)
            next_q_target = self.target_model(next_states)
            best_actions = tf.argmax(next_q_online, axis=1)
            next_q = tf.reduce_sum(tf.one_hot(best_actions, self.actions_cnt) * next_q_target, axis=1)

            gamma = tf.cast(gamma_value, next_q.dtype)
            one = tf.constant(1.0, dones.dtype)
            target_q = rewards + gamma * (one - dones) * next_q

            td_errors = actual_q - target_q
            loss = self.loss_fn(target_q, actual_q) * weights
            total_loss = tf.reduce_mean(loss)

        grads = tape.gradient(total_loss, self.online_model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.online_model.trainable_variables))

        return total_loss, td_errors, indices

    def soft_update(self, tau=0.005):
        target_weights = self.target_model.get_weights()
        online_weights = self.online_model.get_weights()
        new_weights = [
            tau * ow + (1.0 - tau) * tw
            for ow, tw in zip(online_weights, target_weights)
        ]
        self.target_model.set_weights(new_weights)

    def interaction_step(self, strategy, state, env):
        state_input = tf.expand_dims(tf.convert_to_tensor(state, dtype=tf.float32), axis=0)
        action = strategy.select_action(self.online_model, state_input)
        next_state, reward, done, _ = env.step(action)
        self.buffer.add((state, action, reward, next_state, done))
        return next_state, reward, done

    def fit(self, env, gamma=0.998, tau=0.01, batch_size=64, episodes=1000, max_len=5000, update_period=9):
        for _ in range(episodes):
            state = env.reset()
            episode_reward = 0
            for step in range(max_len):
                state, reward, done = self.interaction_step(self.train_strat, state, env)
                step += 1
                episode_reward += reward
                if len(self.buffer) > batch_size:
                    exp = self.buffer.sample(batch_size)
                    total_loss, td_errors, indices = self.train_step(exp, gamma)
                    self.buffer.update_priorities(indices, tf.abs(td_errors))

                    if step % update_period == 0:
                        self.soft_update(tau)

                if done:
                    print(episode_reward, step)
                    break

            gc.collect()

        return self.online_model

    def evaluate(self, env, episodes=10, max_len=5000):
        rewards = []
        for _ in range(episodes):
            _, episode_reward = self.predict(env, max_len)
            rewards.append(episode_reward)
            gc.collect()

        return np.mean(rewards)

    def predict(self, env, max_len=2500):
        state = env.reset()
        history = []
        total_reward = 0
        for step in range(max_len):
            state, reward, done = self.interaction_step(self.test_strat, state, env)
            total_reward += reward
            history.append((state, total_reward))
            if done:
                break

        return history, total_reward

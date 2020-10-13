"""
Using training code and agent from https://www.kaggle.com/garethjns/deep-q-learner-starter-code
and https://www.kaggle.com/garethjns/convolutional-deep-q-learner. Derived from code in
https://github.com/garethjns/reinforcement-learning-keras
"""

import collections
import os
import pickle
import zlib
from dataclasses import dataclass
from typing import Tuple, Any, List, Union, Callable, Iterable, Dict

import gym
import numpy as np
from reinforcement_learning_keras.agents.models.split_layer import SplitLayer
from tensorflow import keras


class SMMFrameProcessWrapper(gym.Wrapper):
    """
    Wrapper for processing frames from SMM observation wrapper from football env.

    Input is (72, 96, 4), where last dim is (team 1 pos, team 2 pos, ball pos,
    active player pos). Range 0 -> 255.
    Output is (72, 96, 4) as difference to last frame for all. Range -1 -> 1
    """

    def __init__(self, env: Union[None, gym.Env] = None,
                 obs_shape: Tuple[int, int] = (72, 96, 4)) -> None:
        """
        :param env: Gym env, or None. Allowing None here is unusual,
                    but we'll reuse the buffer functunality later in
                    the submission, when we won't be using the gym API.
        :param obs_shape: Expected shape of single observation.
        """
        if env is not None:
            super().__init__(env)
        self._buffer_length = 2
        self._obs_shape = obs_shape
        self._prepare_obs_buffer()

    @staticmethod
    def _normalise_frame(frame: np.ndarray):
        return frame / 255.0

    def _prepare_obs_buffer(self) -> None:
        """Create buffer and preallocate with empty arrays of expected shape."""

        self._obs_buffer = collections.deque(maxlen=self._buffer_length)

        for _ in range(self._buffer_length):
            self._obs_buffer.append(np.zeros(shape=self._obs_shape))

    def build_buffered_obs(self) -> np.ndarray:
        """
        Iterate over the last dimenion, and take the difference between this obs
        and the last obs for each.
        """
        agg_buff = np.empty(self._obs_shape)
        for f in range(self._obs_shape[-1]):
            agg_buff[..., f] = self._obs_buffer[1][..., f] - self._obs_buffer[0][..., f]

        return agg_buff

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[Any, Any]]:
        """Step env, add new obs to buffer, return buffer."""
        obs, reward, done, info = self.env.step(action)

        obs = self._normalise_frame(obs)
        self._obs_buffer.append(obs)

        return self.build_buffered_obs(), reward, done, info

    def reset(self) -> np.ndarray:
        """Add initial obs to end of pre-allocated buffer.

        :return: Buffered observation
        """
        self._prepare_obs_buffer()
        obs = self.env.reset()
        self._obs_buffer.append(obs)

        return self.build_buffered_obs()


class SplitterConvNN:

    def __init__(self, observation_shape: List[int], n_actions: int,
                 output_activation: Union[None, str] = None,
                 unit_scale: int = 1, learning_rate: float = 0.0001,
                 opt: str = 'Adam') -> None:
        """
        :param observation_shape: Tuple specifying input shape.
        :param n_actions: Int specifying number of outputs
        :param output_activation: Activation function for output. Eg.
                                  None for value estimation (off-policy methods).
        :param unit_scale: Multiplier for all units in FC layers in network
                           (not used here at the moment).
        :param opt: Keras optimiser to use. Should be string.
                    This is to avoid storing TF/Keras objects here.
        :param learning_rate: Learning rate for optimiser.

        """
        self.observation_shape = observation_shape
        self.n_actions = n_actions
        self.unit_scale = unit_scale
        self.output_activation = output_activation
        self.learning_rate = learning_rate
        self.opt = opt

    @staticmethod
    def _build_conv_branch(frame: keras.layers.Layer, name: str) -> keras.layers.Layer:
        conv1 = keras.layers.Conv2D(16, kernel_size=(8, 8), strides=(4, 4),
                                    name=f'conv1_frame_{name}', padding='same',
                                    activation='relu')(frame)
        mp1 = keras.layers.MaxPooling2D(pool_size=2, name=f"mp1_frame_{name}")(conv1)
        conv2 = keras.layers.Conv2D(24, kernel_size=(4, 4), strides=(2, 2),
                                    name=f'conv2_frame_{name}', padding='same',
                                    activation='relu')(mp1)
        mp2 = keras.layers.MaxPooling2D(pool_size=2, name=f"mp2_frame_{name}")(conv2)
        conv3 = keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1),
                                    name=f'conv3_frame_{name}', padding='same',
                                    activation='relu')(mp2)
        mp3 = keras.layers.MaxPooling2D(pool_size=2, name=f"mp3_frame_{name}")(conv3)

        flatten = keras.layers.Flatten(name=f'flatten_{name}')(mp3)

        return flatten

    def _model_architecture(self) -> Tuple[keras.layers.Layer, keras.layers.Layer]:
        n_units = 512 * self.unit_scale

        frames_input = keras.layers.Input(name='input', shape=self.observation_shape)
        frames_split = SplitLayer(split_dim=3)(frames_input)
        conv_branches = []
        for f, frame in enumerate(frames_split):
            conv_branches.append(self._build_conv_branch(frame, name=str(f)))

        concat = keras.layers.concatenate(conv_branches)
        fc1 = keras.layers.Dense(units=int(n_units), name='fc1',
                                 activation='relu')(concat)
        fc2 = keras.layers.Dense(units=int(n_units / 2), name='fc2',
                                 activation='relu')(fc1)
        action_output = keras.layers.Dense(units=self.n_actions, name='output',
                                           activation=self.output_activation)(fc2)

        return frames_input, action_output

    def compile(self, model_name: str = 'model',
                loss: Union[str, Callable] = 'mse') -> keras.Model:
        """
        Compile a copy of the model using the provided loss.

        :param model_name: Name of model
        :param loss: Model loss. Default 'mse'. Can be custom callable.
        """
        # Get optimiser
        if self.opt.lower() == 'adam':
            opt = keras.optimizers.Adam
        elif self.opt.lower() == 'rmsprop':
            opt = keras.optimizers.RMSprop
        else:
            raise ValueError(f"Invalid optimiser {self.opt}")

        state_input, action_output = self._model_architecture()
        model = keras.Model(inputs=[state_input], outputs=[action_output],
                            name=model_name)
        model.compile(optimizer=opt(learning_rate=self.learning_rate),
                      loss=loss)

        return model

    def plot(self, model_name: str = 'model') -> None:
        keras.utils.plot_model(self.compile(model_name),
                               to_file=f"{model_name}.png", show_shapes=True)
        # plt.show()


@dataclass
class ContinuousBuffer:
    buffer_size: int = 50

    def __post_init__(self) -> None:
        self._state_queue = collections.deque(maxlen=self.buffer_size)
        self._other_queue = collections.deque(maxlen=self.buffer_size)

        self.queue = collections.deque(maxlen=self.buffer_size)

    def __len__(self) -> int:
        return self.n if (self.n > 0) else 0

    @property
    def full(self) -> bool:
        return len(self._state_queue) == self.buffer_size

    @property
    def n(self) -> int:
        return len(self._state_queue) - 1

    def append(self, items: Tuple[Any, int, float, bool]) -> None:
        """
        :param items: Tuple containing (s, a, r, d).
        """
        self._state_queue.append(items[0])
        self._other_queue.append(items[1::])

    def get_batch(self, idxs: Iterable[int]) -> Tuple[List[np.ndarray],
                                                      List[np.ndarray],
                                                      List[float],
                                                      List[bool],
                                                      List[np.ndarray]]:
        ss = [self._state_queue[i] for i in idxs]
        ss_ = [self._state_queue[i + 1] for i in idxs]

        ard = [self._other_queue[i] for i in idxs]
        aa = [a for (a, _, _) in ard]
        rr = [r for (_, r, _) in ard]
        dd = [d for (_, _, d) in ard]

        return ss, aa, rr, dd, ss_

    def sample_batch(self, n: int) -> Tuple[List[np.ndarray],
                                            List[np.ndarray],
                                            List[float],
                                            List[bool],
                                            List[np.ndarray]]:
        if n > self.n:
            raise ValueError

        idxs = np.random.randint(0, self.n, n)
        return self.get_batch(idxs)


@dataclass
class EpsilonGreedy:
    """
    Handles epsilon-greedy action selection, decay of epsilon during training.

    :param eps_initial: Initial epsilon value.
    :param decay: Decay rate in percent (should be positive to decay).
    :param decay_schedule: 'linear' or 'compound'.
    :param eps_min: The min value epsilon can fall to.
    :param state: Random state, used to pick between the greedy or random options.
    """
    eps_initial: float = 0.2
    decay: float = 0.0001
    decay_schedule: str = 'compound'
    eps_min: float = 0.01
    state = None

    def __post_init__(self) -> None:
        self._step: int = 0
        self.eps_current = self.eps_initial

        valid_decay = ('linear', 'compound')
        if self.decay_schedule.lower() not in valid_decay:
            raise ValueError(f"Invalid decay schedule {self.decay_schedule}. "
                             "Pick from {valid_decay}.")

        self._set_random_state()

    def _set_random_state(self) -> None:
        self._state = np.random.RandomState(self.state)

    def _linear_decay(self) -> float:
        return self.eps_current - self.decay

    def _compound_decay(self) -> float:
        return self.eps_current - self.eps_current * self.decay

    def _decay(self):
        new_eps = np.nan
        if self.decay_schedule.lower() == 'linear':
            new_eps = self._linear_decay()

        if self.decay_schedule.lower() == 'compound':
            new_eps = self._compound_decay()

        self._step += 1

        return max(self.eps_min, new_eps)

    def select(self, greedy_option: Callable, random_option: Callable,
               training: bool = False) -> Any:
        """
        Apply epsilon greedy selection.

        If training, decay epsilon, and return selected option.
        If not training, just return greedy_option.

        Use of lambdas is to avoid unnecessarily picking between
        two pre-computed options.

        :param greedy_option: Function to evaluate if random option
                              is NOT picked.
        :param random_option: Function to evaluate if random option
                              IS picked.
        :param training: Bool indicating if call is during training
                         and to use epsilon greedy and decay.
        :return: Evaluated selected option.
        """
        if training:
            self.eps_current = self._decay()
            if self._state.random() < self.eps_current:
                return random_option()

        return greedy_option()


@dataclass
class DeepQAgent:
    replay_buffer: ContinuousBuffer
    eps: EpsilonGreedy
    model_architecture: SplitterConvNN
    name: str = 'DQNAgent'
    double: bool = False
    noisy: bool = False
    gamma: float = 0.99
    replay_buffer_samples: int = 75
    final_reward: Union[float, None] = None

    def __post_init__(self) -> None:
        self._build_model()

    def _build_model(self) -> None:
        """
        Prepare two of the same model.

        The action model is used to pick actions and the target model
        is used to predict value of Q(s', a). Action model
        weights are updated on every buffer sample + training step.
        The target model is never directly trained, but it's
        weights are updated to match the action model at the end of
        each episode.
        """
        self._action_model = self.model_architecture.compile(
            model_name='action_model', loss='mse')
        self._target_model = self.model_architecture.compile(
            model_name='target_model', loss='mse')

    def transform(self, s: np.ndarray) -> np.ndarray:
        """Check input shape, add Row dimension if required."""

        if len(s.shape) < len(self._action_model.input.shape):
            s = np.expand_dims(s, 0)

        return s

    def update_experience(self, s: np.ndarray, a: int,
                          r: float, d: bool) -> None:
        """
        First the most recent step is added to the buffer.

        Note that s' isn't saved because there's no need.
        It'll be added next step. s' for any s is always index + 1 in
        the buffer.
        """

        # Add s, a, r, d to experience buffer
        self.replay_buffer.append((s, a, r, d))

    def update_model(self) -> None:
        """
        Sample a batch from the replay buffer, calculate targets using
        target model, and train action model.

        If the buffer is below its minimum size, no training is done.

        If the buffer has reached its minimum size, a training batch
        from the replay buffer and the action model is updated.

        This update samples random (s, a, r, s') sets from the buffer
        and calculates the discounted reward for each set.
        The value of the actions at states s and s' are predicted from
        the value model. The action model is updated using these value
        predictions as the targets. The value of performed action is
        updated with the discounted reward (using its value prediction
        at s'). ie. x=s, y=[action value 1, action value 2].
        """

        # If buffer isn't full, don't train
        if not self.replay_buffer.full:
            return

        # Else sample batch from buffer
        ss, aa, rr, dd, ss_ = self.replay_buffer.sample_batch(
            self.replay_buffer_samples)

        # Calculate estimated S,A values for current states and next states.
        # These are stacked together first to avoid making two separate
        # predict calls (which is slow on GPU).
        ss = np.array(ss)
        ss_ = np.array(ss_)
        y_now_and_future = self._target_model.predict_on_batch(np.vstack((ss, ss_)))
        # Separate again
        y_now = y_now_and_future[0:self.replay_buffer_samples]
        y_future = y_now_and_future[self.replay_buffer_samples::]

        # Update rewards where not done with y_future predictions
        dd_mask = np.array(dd, dtype=bool).squeeze()
        rr = np.array(rr, dtype=float).squeeze()

        # Gather max action indexes and update relevant actions in y
        if self.double:
            # If using double dqn select best actions using the action model,
            # but the value of those action using the
            # target model (already have in y_future).
            y_future_action_model = self._action_model.predict_on_batch(ss_)
            selected_actions = np.argmax(y_future_action_model[~dd_mask, :],
                                         axis=1)
        else:
            # If normal dqn select targets using target model,
            # and value of those from target model too
            selected_actions = np.argmax(y_future[~dd_mask, :],
                                         axis=1)

        # Update reward values with estimated values (where not done)
        # and final rewards (where done)
        rr[~dd_mask] += y_future[~dd_mask, selected_actions]
        if self.final_reward is not None:
            # If self.final_reward is set, set done cases to this value.
            # Else leave as observed reward.
            rr[dd_mask] = self.final_reward
        aa = np.array(aa, dtype=int)
        np.put_along_axis(y_now, aa.reshape(-1, 1), rr.reshape(-1, 1), axis=1)

        # Fit model with updated y_now values
        self._action_model.train_on_batch(ss, y_now)

    def get_best_action(self, s: np.ndarray) -> np.ndarray:
        """
        Get best action(s) from model - the one with the highest predicted value.

        :param s: A single or multiple rows of state observations.
        :return: The selected action.
        """
        preds = self._action_model.predict(self.transform(s))

        return np.argmax(preds)

    def get_action(self, s: np.ndarray, training: bool = False) -> int:
        """
        Get an action using get_best_action or epsilon greedy.

        Epsilon decays every time a random action is chosen.

        :param s: The raw state observation.
        :param training: Bool to indicate whether or not to use this
                         experience to update the model. If False, just
                         returns best action.
        :return: The selected action.
        """
        action = self.eps.select(greedy_option=lambda: self.get_best_action(s),
                                 random_option=lambda: self.env.action_space.sample(),
                                 training=training)

        return action

    def update_target_model(self) -> None:
        """
        Update the value model with the weights of the action model
        (which is updated each step).

        The value model is updated less often to aid stability.
        """
        self._target_model.set_weights(self._action_model.get_weights())

    def after_episode_update(self) -> None:
        """Value model synced with action model at the end of each episode."""
        self.update_target_model()

    def _discounted_reward(self, reward: float,
                           estimated_future_action_rewards: np.ndarray) -> float:
        """
        Use this to define the discounted reward for unfinished episodes,
        default is 1 step TD.
        """
        return reward + self.gamma * np.max(estimated_future_action_rewards)

    def _get_reward(self, reward: float,
                    estimated_future_action_rewards: np.ndarray,
                    done: bool) -> float:
        """
        Calculate discounted reward for a single step.

        :param reward: Last real reward.
        :param estimated_future_action_rewards: Estimated future values
                                                of actions taken on next step.
        :param done: Flag indicating if this is the last step on an episode.
        :return: Reward.
        """

        if done:
            # If done, reward is just this step. Can finish because agent has won or lost.
            return self._final_reward(reward)
        else:
            # Otherwise, it's the reward plus the predicted max value of next action
            return self._discounted_reward(reward,
                                           estimated_future_action_rewards)


def play_episode(env: gym.Env, agent: DeepQAgent, n_steps: int = 10, pr: bool = False,
                 training: bool = False) -> Tuple[List[float], List[int]]:
    episode_rewards = []
    episode_actions = []

    obs = env.reset()
    done = False
    for s in range(n_steps):
        if done:
            break

        # Select action
        action = agent.get_action(obs)
        episode_actions.append(action)

        # Take action
        prev_obs = obs
        obs, reward, done, info = env.step(action)
        episode_rewards.append(reward)

        # Update model
        if training:
            agent.update_experience(s=prev_obs, a=action,
                                    r=reward, d=done)
            agent.update_model()

        if pr:
            print(f"Step {s}: Action taken {action}, "
                  f"reward received {reward}")

        last_step = s

    if training:
        agent.after_episode_update()

    return episode_rewards, episode_actions


def run_multiple_episodes(env: gym.Env, agent, n_episodes: int = 10,
                          n_steps: int = 10, pr: bool = False,
                          training: bool = False) -> List[float]:
    total_episode_rewards = []
    for ep in range(n_episodes):

        episode_rewards, _ = play_episode(env, agent, n_steps,
                                          pr=False, training=training)

        total_episode_rewards.append(sum(episode_rewards))

        if pr:
            print(f"Episode {ep} finished after {len(episode_rewards)} "
                  f"steps, total reward: {sum(episode_rewards)}")

    return total_episode_rewards


def train_agent(output_path: str = 'saved_model/'):
    import gfootball  # noqa
    env = gym.make("GFootball-11_vs_11_kaggle-SMM-v0")

    agent = DeepQAgent(name='test',
                       replay_buffer=ContinuousBuffer(buffer_size=1000),
                       eps=EpsilonGreedy(),
                       model_architecture=SplitterConvNN(observation_shape=(72, 96, 4),
                                                         n_actions=19))

    run_multiple_episodes(env, agent, n_episodes=5, n_steps=10)

    agent._action_model.save(os.path.join(output_path))

    return agent


if __name__ == "__main__":
    agent_ = train_agent('test_model/')

    weights_str = zlib.compress(pickle.dumps(agent_._action_model.get_weights()))

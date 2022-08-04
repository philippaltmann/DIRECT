from typing import abc
import numpy as np; import torch as th

class DiscrimNet(th.nn.Module, abc.ABC):
    """Abstract base class for discriminator, used in AIRL and GAIL."""

    def predict_reward_train(
        self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray, done: np.ndarray,
    ) -> np.ndarray:
        """Vectorized reward for training an imitation learning algorithm.

        Args:
            state: The observation input. Its shape is
                `(batch_size,) + observation_space.shape`.
            action: The action input. Its shape is
                `(batch_size,) + action_space.shape`. The None dimension is
                expected to be the same as None dimension from `obs_input`.
            next_state: The observation input. Its shape is
                `(batch_size,) + observation_space.shape`.
            done: Whether the episode has terminated. Its shape is `(batch_size,)`.
        Returns:
            The rewards. Its shape is `(batch_size,)`.
        """
        return self._eval_reward(
            is_train=True, state=state, action=action, next_state=next_state, done=done
        )

    def predict_reward_test(
        self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray, done: np.ndarray,
    ) -> np.ndarray:
        """Vectorized reward for training an expert during transfer learning.

        Args:
            state: The observation input. Its shape is
                `(batch_size,) + observation_space.shape`.
            act: The action input. Its shape is
                `(batch_size,) + action_space.shape`. The None dimension is
                expected to be the same as None dimension from `obs_input`.
            next_state: The observation input. Its shape is
                `(batch_size,) + observation_space.shape`.
            done: Whether the episode has terminated. Its shape is `(batch_size,)`.
        Returns:
            The rewards. Its shape is `(batch_size,)`.
        """
        return self._eval_reward(
            is_train=False, state=state, action=action, next_state=next_state, done=done
        )

    def _eval_reward(
        self, is_train: bool, state: th.Tensor, action: th.Tensor, next_state: th.Tensor, done: th.Tensor,
    ) -> np.ndarray:
        (state_th, action_th, next_state_th, done_th, ) = rewards_common.disc_rew_preprocess_inputs(
            observation_space=self.observation_space, action_space=self.action_space,state=state,     action=action,     next_state=next_state,     done=done,     device=self.device(),     scale=self.scale, )

        with th.no_grad():
            if is_train:
                rew_th = self.reward_train(state_th, action_th, next_state_th, done_th)
            else:
                rew_th = self.reward_test(state_th, action_th, next_state_th, done_th)

        rew = rew_th.detach().cpu().numpy() 
        assert rew.shape == (len(state),)

        return rew

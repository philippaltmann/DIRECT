import gym
import numpy as np
import torch as th
from stable_baselines3.ppo import PPO
# from stable_baselines3.common import logger, utils
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import MaybeCallback
from typing import Type, Union, Any, Dict, List, Tuple, Optional

from tqdm import tqdm

from torch.utils.tensorboard.writer import SummaryWriter

from . import DirectBuffer, Discriminator

import util.evaluation as eval

# from sklearn import discriminant_analysis

#TODO replace self.logger.info oä
class DIRECT(PPO):
    """
    Discriminative Reward Co-Training (DIRECT)

    Extending the PPO Implementation by stable baselines 3
    for further Hyperparameters refer to the base class

    :param env: The environment to learn from (if registered in Gym, can be str)
    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param kappa: (int) k-Best Trajectories to be stored in the reward buffer
    :param omega: (float) The frequency to perform updates to the discriminator. 
        [1/1 results in concurrent training of policy and discriminator]
    :param chi: (float): The mixture parameter determining the mixture of real and discriminative reward
        [1 => purely discriminateive reward, 0 => no discriminateive reward, just real reward (pure PPO)]
    :param disc_kwargs: (dict) parameters to be passed the discriminator on creation
        see Discriminator class for full description
    # :param ppo_kwargs: (dict) additional parameters to be passed to ppo on creation
    :param seed: Seed for the pseudo random generators
    :param tb_path: (str) the log location for tensorboard (if None, no logging)
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance

    DIRECT(env, "MlpPolicy", chi=, kappa=, omega=)
    """

    def __init__(
        self, env: Union[gym.Env, str], policy: Union[str, Type[ActorCriticPolicy]] = "MlpPolicy",
        chi: float = 1.0, kappa: int = 512, omega: float = 1/1, normalize: bool = False, 
        device: Union[th.device, str] = "auto", disc_kwargs: Dict[str, Any] = {}, ppo_kwargs: Dict[str, Any] = {}, 
        seed: Optional[int] = None, tb_path: Optional[str] = None, _init_setup_model: bool = True,
    ):
        env = VecNormalize(env) if normalize else env 
        self.tb_path= tb_path
        self.progress_bar = None
        self.hidden_rewards, self.observed_rewards, self.discounts = None, None, None

        self.chi = chi; assert chi <= 1.0
        self.kappa = kappa; assert kappa > 0
        self.omega = omega; assert 0 < omega < 10
        self.buffer = None
        self.discriminator = None        
        disc_kwargs.setdefault('hidden_size', [32,32])
        # disc_kwargs.setdefault('optimizer_class', th.optim.Adam)
        self.disc_kwargs = disc_kwargs


        # TODO Add support for variable algorithm inheriting from "OnPolicyAlgorithm"
        self.ppo_kwargs = ppo_kwargs
        super().__init__(policy, env, device=device, seed=seed, _init_setup_model=False, **self.ppo_kwargs)
        print(f"Using {self.device} {f'v{th.version.cuda} on {th.cuda.get_device_name(0)}' if th.cuda.is_available() else''} with seed {self.seed}")
        if _init_setup_model: self._setup_model()

    def _setup_model(self) -> None:
        super(DIRECT, self)._setup_model()
        self.writer = SummaryWriter(log_dir=self.tb_path) if self.tb_path else None
        self.disc_kwargs.setdefault('batch_size', min(int(self.n_steps*self.n_envs*self.omega), self.n_steps*self.n_envs))
        self.buffer = DirectBuffer(buffer_size=self.kappa, parent=self)
        self.discriminator = Discriminator(chi=self.chi, parent=self, **self.disc_kwargs).to(self.device)
        assert False, "Setup direct model"

    def _excluded_save_params(self) -> List[str]:
        """
        Returns the names of the parameters that should be excluded from being
        saved by pickling. E.g. replay buffers are skipped by default
        as they take up a lot of space. PyTorch variables should be excluded
        with this so they can be stored with ``th.save``.

        :return: List of parameters that should be excluded from being saved with pickle.
        """
        exclude = super(DIRECT, self)._excluded_save_params()
        exclude.extend(['buffer', 'writer', 'progress_bar'])
        return exclude

     #'TODO'
    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        """
        Get the name of the torch variables that will be saved with
        PyTorch ``th.save``, ``th.load`` and ``state_dicts`` instead of the default
        pickling strategy. This is to handle device placement correctly.

        Names can point to specific variables under classes, e.g.
        "policy.optimizer" would point to ``optimizer`` object of ``self.policy``
        if this object.

        :return:
            List of Torch variables whose state dicts to save (e.g. th.nn.Modules),
            and list of other Torch variables to store with ``th.save``.
        """
        # print(self.discriminator.state_dict())
        # print(self.discriminator._get_constructor_parameters())
        torch, vars = super(DIRECT, self)._get_torch_save_params()
        torch.extend(["discriminator"])
        return torch, vars
        # state_dicts = ["policy"]
        # # assert False, "TODO"
        # return state_dicts, []
        # def save(self, path: str) -> None:
        # """
        # Save model to a given location.

        # :param path:
        # """
        # th.save({"state_dict": self.state_dict(), "data": self._get_constructor_parameters()}, path)


    # TODO custom eval env/ flow
    # TODO log_omterval not needed? 
    def learn(
        self, total_timesteps: int, callback: MaybeCallback = None, log_interval: int = 1, 
        eval_env: Optional[gym.Env] = None, eval_freq: int = -1, n_eval_episodes: int = 5, 
        eval_log_path: Optional[str] = None, reset_num_timesteps: bool = True #tb_log_name: str = "DIRECT", 
    ) -> "DIRECT":
        self.progress_bar = tqdm(
            total=total_timesteps, desc=f"Training DIRECT(χ={self.chi}, κ={self.kappa}, ω={self.omega})", unit="steps", 
            postfix=[0], bar_format="{desc}[R: {postfix[0]:4.2f}][{bar}]({percentage:3.0f}%)[{n_fmt}/{total_fmt}@{rate_fmt}]"
        )
        self.hidden_rewards, self.observed_rewards, self.discounts =  [[] for _ in range(self.n_envs)], [[] for _ in range(self.n_envs)], [[] for _ in range(self.n_envs)]

        super(DIRECT, self).learn( #model =  
            total_timesteps=total_timesteps, callback=callback, log_interval=log_interval,
            eval_env=eval_env, eval_freq=eval_freq, n_eval_episodes=n_eval_episodes, 
            tb_log_name="", eval_log_path=eval_log_path, reset_num_timesteps=reset_num_timesteps #tb_log_name
        )
        self.progress_bar.close()

    def train(self) -> None:
        step = self.num_timesteps
        mean_return = safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer])
        mean_length = safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer])
        self.progress_bar.postfix = [mean_return]
        self.progress_bar.update(self.n_steps * self.env.num_envs)

        # Force Typing for used Objects 
        discriminator: Discriminator = self.discriminator
        rollout: RolloutBuffer = self.rollout_buffer
        real_rewards = rollout.rewards.copy()        

        # Update Buffer (previous: self.action_probs as Bias)
        self.buffer.extend(rollout=rollout)
        self.buffer.write_metrics()

        # init buffer to hold rollout trajectories
        policy_buffer = discriminator.process(rollout=rollout)

        # Update Returns from  Discriminative Rewards for updating the policy
        with th.no_grad(): _, values, _ = self.policy.forward(obs_as_tensor(self._last_obs, self.device))
        rollout.compute_returns_and_advantage(last_values=values, dones=self._last_episode_starts)
        disc_rewards = rollout.rewards.copy()        
        
        # Update Discriminator
        disc_updates = int(self.num_timesteps // (self.omega * self.n_steps * self.n_envs))
        self.logger.info("Performing {:d} discriminator updates with batch[{}] from rollout[{}]".format(disc_updates, discriminator.batch_size, policy_buffer.size()))
        disc_stats = [discriminator.train(direct_buffer=self.buffer,policy_buffer=policy_buffer) for _ in range(disc_updates)]
        
        # TODO eval.write_stats(disc_stats, self.writer)
        # TODO eval.write_ci(real_rewards, self.writer,....)

        [self.writer.add_scalar(key, value, step) for stats in disc_stats for key, value in stats.items() if isinstance(value, (float, int))]
        [self.writer.add_histogram(key, values, step) for stats in disc_stats for key, value in stats.items() if isinstance(value, (th.Tensor, np.ndarray))]
        # [self.writer.add_scalars('', stats, step) for stats in disc_stats]
        # [print("Adding: {} | {} at {}".format(key, value, step)) for stats in disc_stats for key, value in stats.items()]
        
        # Train PPO
        super(DIRECT, self).train()

        [self.writer.add_scalar(key, value, step) for key, value in self.logger.name_to_value.items() if isinstance(value, (float, int))]
        assert len([key for key, value in self.logger.name_to_value.items() if isinstance(value, (th.Tensor, np.ndarray))]) == 0, "missed writing hist to tb"
        self.writer.flush()
        
        # TODO write PPO logs to TB when verbose=0 -> logging forced off

        # TODO: returns vs rewards ??
        # print(len(real_rewards))
        self.writer.add_scalar("rewards/100-mean-return", mean_return, step)
        self.writer.add_scalar("rewards/100-mean-length", mean_length, step)
        
        #TODO: cumulate disc & env rewards in similar manner to additional safety metrics
        eval.write_ci(real_rewards, self.writer, "rewards", "environment", self.num_timesteps, self.n_steps, 0.2)
        eval.write_ci(disc_rewards, self.writer, "rewards", "discriminator", self.num_timesteps, self.n_steps, 0.2)
        
        # print(self.num_timesteps)
        # print("Trained")
        # assert False

        # Log Rewards & dump logs
        # self.logger.record("rewards/enironment", float(real_rewards.mean()))
        # self.logger.record("rewards/discriminator", float(disc_rewards.mean()))
        # self.logger.dump(step=self.num_timesteps)
        # # ideas: 
        # # self.logger.log("_current_progress_remaining: {}".format(model._current_progress_remaining))
        # # self.logger.log("_episode_num: {}".format(model._episode_num))
        # Previous:
        #  self.log_probs
        # _prob = self.model.action_probability(observation)
        # self.action_probs.append(np.var(_prob[0])) # TODO for each env
    
    # Moved to /util/hparams for universal usage with PPO and other algorithms
    # def get_hparams(self):
    #     """ Fetches, filters & falttens own hyperparameters
    #     :return: Dict of Hyperparameters containing numbers and strings only
    #     """
    #     #_n_updates=num_timesteps/(n_envs*n_steps)/n_epochs
    #     # TODO: return list with discretes (for hparams)
    #     # additional = [ ]
    #     'ppo_kwargs' not param anymore
    #     exclude = ['discriminator', 'disc_kwargs',  'ppo_kwargs', 'policy',  'policy_kwargs', 'start_time', 'policy_class', 'discounts'] \
    #         + ['buffer','ep_info_buffer', 'ep_success_buffer', 'rollout_buffer', 'lr_schedule', 'device', 'hidden_rewards', 'observed_rewards'] \
    #         + ['env', '_vec_normalize_env', 'eval_env', 'observation_space', 'action_space', 'action_noise','_last_obs', '_last_original_obs' ] \
    #         + ['verbose', 'writer', '_logger', '_custom_logger', 'tensorboard_log',  'tb_path', 'progress_bar', '_last_episode_starts' ] \
    #         + ['sde_sample_freq', '_current_progress_remaining', '_episode_num', '_n_updates', 'clip_range', 'clip_range_vf', 'target_kl']
        
    #     hparams = {k : v.__name__ if isinstance(v, type) else v for k, v in vars(self).items() if k not in exclude and not k.startswith('_')}        
    #     hparams = {**hparams, **self.discriminator.get_hparams(prefix="disc_"), 'device': self.device.type}
    #     hparams = {k: v if isinstance(v, (int, float, str, bool, th.Tensor)) else str(v) for k,v in hparams.items()}
    #     return hparams

    def _update_info_buffer(self, infos: List[Dict[str, Any]], dones: Optional[np.ndarray] = None) -> None:
        """
        Retrieve reward, episode length, episode success and update the buffer
        if using Monitor wrapper or a GoalEnv.

        :param infos: List of additional information about the transition.
        :param dones: Termination signals
        """
        # TODO: move writing to callback?
        super(DIRECT, self)._update_info_buffer(infos, dones)
        for i, info in enumerate(infos):
          if info.get("episode"):
            if (len(self.hidden_rewards[i])): self.writer.add_scalar( 'rewards/hidden_reward', 
              sum(np.multiply(self.hidden_rewards[i],self.discounts[i]).copy()), self.num_timesteps)
            if (len(self.observed_rewards[i])): self.writer.add_scalar('rewards/observed_reward', 
              sum(np.multiply(self.observed_rewards[i],self.discounts[i]).copy()), self.num_timesteps)
            self.hidden_rewards[i], self.observed_rewards[i], self.discounts[i] = [],[],[]
          else:
            if info.get('hidden_reward'): self.hidden_rewards[i].append(info.get('hidden_reward'))
            if info.get('observed_reward'): self.observed_rewards[i].append(info.get('observed_reward'))
            if info.get('discount'): self.discounts[i].append(info.get('discount'))

       

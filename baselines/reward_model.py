import numpy as np; import torch as th; import gymnasium as gym
import torch.nn as nn; import torch.nn.functional as F


class RewardModel:
  def __init__(self, env, ensemble_size=3, device='auto', lr=3e-4, activation=nn.Tanh, n_updates=100,
    mb_size=128, large_batch=10, size_segment=50,  capacity=5e5, label_margin=0.0, teacher='oracle'):
    """ Reward model class for Preference-Based RL, simulating human teacher feedback to learn a reward
    :param ensemble_size (int): number of models in the ensemble (default: 3)
    :param lr (float): learning rate of the reward model (default: 3e-4)
    :param activation (th.nn): Activation function of the reward model (default: tanh)
    :param n_updates (int): number of updates per training step (default: 100)
    :param mb_size (int): number of samples per batch (default: 128)
    :param large_batch (int): sacling of batch size for disgreement sampling (default: 10)
    :param size_segment (int): Size of a simulated feedback segment (default: 50)
    :param capacity (int): Capacity of the experience buffer for training the reward model (default: 5e5)
    :param label_margin (float): For Soft Cross Entropy Updates (default: 0.0)
    :param teacher (str): Teacher Mode ['oracle'|'mistake'|'noisy'|'skip'|'myopic'|'equal']"""

    # Determine Input / Output dimensions (|S]+|A| -> 1)
    obs = env.envs[0].observation_space.shape[0]
    act = 1 if isinstance(env.envs[0].action_space, gym.spaces.Discrete) else env.envs[0].action_space.shape[0]
    self.inputs = []; self.targets = []; self.input_size = obs+act;  

    # Construct Ensemble Model
    self.ensemble_size = ensemble_size; self.ensemble = []
    self.model = None; self.paramlst = []; self.device = device
    self.lr = lr; self.activation = activation; self.opt = None
    self.construct_ensemble()
    
    # Setup Training
    self.mb_size = mb_size; self.large_batch = large_batch 
    self.CEloss = nn.CrossEntropyLoss(); self.n_updates = n_updates
    
    # Init Experience Buffer
    self.capacity = int(capacity); self.buffer_index = 0; self.buffer_full = False
    self.size_segment = size_segment; self.max_ep_len = env.envs[0].unwrapped.max_episode_steps
    self.buffer_seg1 = np.empty((self.capacity, size_segment, self.input_size), dtype=np.float32)
    self.buffer_seg2 = np.empty((self.capacity, size_segment, self.input_size), dtype=np.float32)
    self.buffer_label = np.empty((self.capacity, 1), dtype=np.float32) # TODO: mby move in buffer?           

    """ Init Teacher Models 
    teacher_beta: rationality constant of stochastic preference model (default: -1 for perfectly rational model)
    teacher_gamma: discount factor to model myopic behavior (default: 1)
    teacher_eps_mistake: probability of making a mistake (default: 0)
    teacher_eps_skip: hyperparameters to control skip threshold (\in [0,1])
    teacher_eps_equal: hyperparameters to control equal threshold (\in [0,1])"""
    self.label_margin = label_margin; self.label_target = 1 - 2 * self.label_margin
    self.teacher = teacher; [ setattr(self, f'teacher_{k}', v) for k,v in {
        'oracle':  {'beta': -1, 'gamma': 1,   'eps_mistake': 0,   'eps_equal': 0,   'eps_skip': 0   },
        'mistake': {'beta': -1, 'gamma': 1,   'eps_mistake': 0.1, 'eps_equal': 0,   'eps_skip': 0   },
        'noisy':   {'beta':  1, 'gamma': 1,   'eps_mistake': 0,   'eps_equal': 0,   'eps_skip': 0   },
        'skip':    {'beta': -1, 'gamma': 1,   'eps_mistake': 0,   'eps_equal': 0.1, 'eps_skip': 0   },
        'myopic':  {'beta': -1, 'gamma': 0.9, 'eps_mistake': 0,   'eps_equal': 0,   'eps_skip': 0   },
        'equal':   {'beta': -1, 'gamma': 1,   'eps_mistake': 0,   'eps_equal': 0,   'eps_skip': 0.1 },
      }[teacher].items() ]


  def construct_ensemble(self):
    gen_net = lambda sizes: [l for i,o in zip(sizes, sizes[1:]) for l in [nn.Linear(i, o), nn.LeakyReLU()]][:-1]
    for i in range(self.ensemble_size):
      model = nn.Sequential(*gen_net([self.input_size, *(3 * [256]), 1]), self.activation()).float().to(self.device)
      self.ensemble.append(model); self.paramlst.extend(model.parameters())    
    self.opt = th.optim.Adam(self.paramlst, lr = self.lr)


  def r_hat_member(self, x, member=-1):
    # the network parameterizes r hat in eqn 1 from the paper
    return self.ensemble[member](th.from_numpy(x).float().to(self.device))
    

  def r_hat_batch(self, x):
    return np.mean(np.array(
      [self.r_hat_member(x, member=member).detach().cpu().numpy() for member in range(self.ensemble_size)]
    ), axis=0)


  def add_data(self, ep):
    S, A = ep['history']['states'], ep['history']['actions']
    if len(S.shape) != len(A.shape): A = A[:, np.newaxis]
    R = np.full((ep['history']['rewards'].shape[0], 1), ep['r'])
    # R = ep['rewards'][:, np.newaxis]
    self.inputs.append(np.concatenate([S, A], axis=-1))
    self.targets.append(R)


  def add_data_batch(self, obses, rewards):
    num_env = obses.shape[0]
    for index in range(num_env):
      self.inputs.append(obses[index])
      self.targets.append(rewards[index])


  def update_margin(self, mean_return):
    """Update Teacher Thresolds for Skip and Equal with current mean_return"""
    new_margin = mean_return * (self.size_segment  / self.max_ep_len)
    self.teacher_thres_skip = new_margin * self.teacher_eps_skip
    self.teacher_thres_equal = new_margin * self.teacher_eps_equal

    
  def get_queries(self, large=False):
    """Orginial Code written for unifrom episode lengths, adapted to potentially different"""
    mb_size = self.mb_size * (self.large_batch if large else 1) 
    # 2-Step procedure: sample b episodes from buffer, sample s step segments from episodes
    sample_batch = lambda i, o, b, s: (                  # (ep, tau, SxA|R) => (mb_size, segment_size, SxA|R)
      np.array(batch).reshape((b,s,len(batch[0])))                          # Convert to (b,s,_) arrays
        for batch in zip(*[(i[ep][se], o[ep][se])                           # Unpack input and target
          for ep in np.random.choice(len(i), size=b, replace=True)          # Sample Ep Batch
            for se in np.random.choice(len(i[ep]), size=s, replace=True)])) # Sample Segment

    sa_1,r_1 = sample_batch(self.inputs, self.targets, mb_size, self.size_segment)
    sa_2,r_2 = sample_batch(self.inputs, self.targets, mb_size, self.size_segment)
    return sa_1, sa_2, r_1, r_2


  def get_label(self, sa1, sa2, r1, r2):
    """Simulates asking a teacher for preference between sa1 and sa2"""
    sum_r1, sum_r2 = np.sum(r1, axis=1), np.sum(r2, axis=1)
    
    # skip the query
    if self.teacher_thres_skip > 0: 
      max_index = (np.maximum(sum_r1, sum_r2) > self.teacher_thres_skip).reshape(-1)
      if sum(max_index) == 0: return None, None, None, None, []
      sa1, sa2, r1, r2 = sa1[max_index], sa2[max_index], r1[max_index], r2[max_index]
      sum_r1, sum_r2 = np.sum(r1, axis=1), np.sum(r2, axis=1)

    # equally preferable
    margin_index = (np.abs(sum_r1 - sum_r2) < self.teacher_thres_equal).reshape(-1)
    
    # perfectly rational
    temp_r1, temp_r2 = r1.copy(), r2.copy()
    for index in range(r1.shape[1] - 1):
      temp_r1[:,:index+1] *= self.teacher_gamma; temp_r2[:,:index+1] *= self.teacher_gamma
    sum_r1, sum_r2 = np.sum(r1, axis=1), np.sum(r2, axis=1)
    
    if self.teacher_beta > 0: # Bradley-Terry rational model
      r_hat = th.cat([th.Tensor(sum_r1), th.Tensor(sum_r2)], axis=-1)*self.teacher_beta
      labels = th.bernoulli(F.softmax(r_hat, dim=-1)[:, 1]).int().numpy().reshape(-1, 1)
    else: labels = 1 * (sum_r1 < sum_r2) # rational_labels
    
    # making a mistake
    rand_num = np.random.rand(labels.shape[0])
    noise_index = rand_num <= self.teacher_eps_mistake
    labels[noise_index] = 1 - labels[noise_index]

    # equally preferable
    labels[margin_index] = -1 
    return sa1, sa2, r1, r2, labels


  def put_queries(self, sa1, sa2, labels):
    """Store queries in the experience buffer"""
    total_sample = sa1.shape[0]; next_index = self.buffer_index + total_sample
    def insert(d1,d2,t1,t2):
      np.copyto(self.buffer_seg1[d1:d2], sa1[t1:t2])
      np.copyto(self.buffer_seg2[d1:d2], sa2[t1:t2])
      np.copyto(self.buffer_label[d1:d2], labels[t1:t2])

    if next_index >= self.capacity:
      self.buffer_full = True
      maximum_index = self.capacity - self.buffer_index
      insert(self.buffer_index, self.capacity, None, maximum_index)
      self.buffer_index  = total_sample - (maximum_index)
      if self.buffer_index  > 0: insert(0, self.remain, maximum_index, None)
    else:
      insert(self.buffer_index, next_index, None, None)
      self.buffer_index = next_index


  def sample(self, disagreement=False):
    """Sample Uniform or Disagreement-Based from inputs and targets"""
    sa_1, sa_2, r_1, r_2 = self.get_queries(large=disagreement) # Get Queries 

    if disagreement: # get final queries based on uncertainty
      def p_hat_member(xs, m=-1): # softmaxing to get the probabilities according to eqn 1
        with th.no_grad(): r_hat = th.cat([self.r_hat_member(x, m).sum(axis=1) for x in xs], axis=-1)
        return F.softmax(r_hat, dim=-1)[:,0] # taking 0 index for probability x_1 > x_2
      probs = np.array([p_hat_member([sa_1, sa_2], member).cpu().numpy() for member in range(self.ensemble_size)])
      top_k_index = (-np.std(probs, axis=0)).argsort()[:self.mb_size]
      r_1, sa_1, r_2, sa_2 = r_1[top_k_index], sa_1[top_k_index], r_2[top_k_index], sa_2[top_k_index]        
      assert sa_1.shape[0] == self.mb_size; sa_2.shape[0] == self.mb_size

    # get labels
    sa_1, sa_2, r_1, r_2, labels = self.get_label(sa_1, sa_2, r_1, r_2)
    if len(labels) > 0: self.put_queries(sa_1, sa_2, labels)
    return len(labels)


  def softXEnt_loss(self, input, labels):
    labels[labels == -1] = 0
    target = th.zeros_like(input).scatter(1, labels.unsqueeze(1), self.label_target) 
    target += self.label_margin
    if sum(labels == -1) > 0: target[labels == -1] = 0.5
    logprobs = th.nn.functional.log_softmax(input, dim = 1)
    return  -(target * logprobs).sum() / input.shape[0]

    
  def train_reward(self, soft=False):
    ensemble_losses = [[] for _ in range(self.ensemble_size)]
    ensemble_acc = np.array([0 for _ in range(self.ensemble_size)])
    max_len = self.capacity if self.buffer_full else self.buffer_index
    total_batch_index = [np.random.permutation(max_len) for _ in range(self.ensemble_size)]
    
    num_epochs = int(np.ceil(max_len/self.mb_size)); total = 0
    
    for epoch in range(num_epochs):
      self.opt.zero_grad(); loss = 0.0
      
      last_index = (epoch + 1) * self.mb_size
      if last_index > max_len: last_index = max_len
          
      for member in range(self.ensemble_size):
        # get random batch
        idxs = total_batch_index[member][epoch*self.mb_size:last_index]
        sa_1, sa_2 = self.buffer_seg1[idxs], self.buffer_seg2[idxs]
        labels = th.from_numpy(self.buffer_label[idxs].flatten()).long().to(self.device)
        if member == 0: total += labels.size(0)
        
        r_hat = th.cat([  # get logits
           self.r_hat_member(sa_1, member).sum(axis=1), 
           self.r_hat_member(sa_2, member).sum(axis=1)], axis=-1)

        # compute loss
        if soft: loss = self.softXEnt_loss(r_hat, labels)
        else: loss = self.CEloss(r_hat, labels)
        ensemble_losses[member].append(loss.item())

        # compute acc
        _, predicted = th.max(r_hat.data, 1)
        correct = (predicted == labels).sum().item()
        ensemble_acc[member] += correct
          
      loss.backward()
      self.opt.step()
    
    return ensemble_acc / total
    

  def train(self):
    train_acc = []
    for _ in range(self.n_updates): # update reward
      if self.teacher_eps_equal > 0: train_acc.append(self.train_reward(soft=True))
      else: train_acc.append(self.train_reward(soft=False))
      if np.mean(train_acc) > 0.97: break;
    return np.mean(train_acc)
                            
              
  def save(self, model_dir): #, step
    [th.save(self.ensemble[member].state_dict(), f'{model_dir}/reward_model_{member}.pt') #_{step}
      for member in range(self.ensemble_size)]
    
          
  def load(self, model_dir): #, step
    [self.ensemble[member].load_state_dict(th.load(f'{model_dir}/reward_model_{member}.pt')) #_{step}
      for member in range(self.ensemble_size)]
    
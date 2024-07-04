import numpy as np; import torch as th
def square(x): return x**2

class BNNLayer(th.nn.Module):
  """Probabilistic layer that uses Gaussian weights.
  Each weight has two parameters: mean and standard deviation (std). """

  def __init__(self,num_inputs, num_units, nonlinearity=th.nn.ReLU(), prior_sd=None, use_reparametrization_trick=True):
    super(BNNLayer, self).__init__()
    self.nonlinearity, self.prior_sd = nonlinearity, prior_sd
    self.num_inputs, self.num_units = num_inputs, num_units
    self.use_reparametrization_trick = use_reparametrization_trick
    prior_rho = th.log(th.exp(self.prior_sd) - 1) # Reverse log_to_std transformation.
    self.W = th.Tensor(np.random.normal(0., prior_sd, (self.num_inputs, self.num_units))) 
    self.b = th.Tensor(np.zeros((self.num_units,)))

    # Set weight and bias priors 
    mu, b_mu = th.Tensor(self.num_inputs, self.num_units), th.Tensor(self.num_units)
    th.nn.init.normal_(mu, mean=0., std=1.); th.nn.init.normal_(b_mu, mean=0., std=1.)
    self.mu, self.b_mu = th.nn.Parameter(mu), th.nn.Parameter(b_mu)
    rho, b_rho = th.Tensor(self.num_inputs, self.num_units), th.Tensor(self.num_units)
    th.nn.init.constant_(rho, prior_rho.item()); th.nn.init.constant_(b_rho, prior_rho.item())
    self.rho, self.b_rho = th.nn.Parameter(rho), th.nn.Parameter(b_rho)

    # Backup params for KL calculations.
    self.mu_old = th.Tensor(np.zeros((self.num_inputs, self.num_units)))
    self.rho_old = th.Tensor(np.ones((self.num_inputs, self.num_units)))
    self.b_mu_old = th.Tensor(np.zeros((self.num_units,)))
    self.b_rho_old = th.Tensor(np.ones((self.num_units,)))


  def log_to_std(self, rho):
    """Transformation for allowing rho in \mathbb{R}, rather than \mathbb{R}_+
    This makes sure that we don't get negative stds. However, a downside might be
    that we have little gradient on close to 0 std (= -inf using this transformation)."""
    return th.log(1 + th.exp(rho))


  def forward(self, input):
    if input.ndim > 2: input = input.view(-1, self.num_inputs)  # if the input has more than two dimensions, flatten it into a batch of feature vectors.
    if self.use_reparametrization_trick:  # According to Kingma et al., "Variational Dropout and the Local Reparametrization Trick", 2015
      gamma = th.mm(input, self.mu) + self.b_mu.expand(input.size()[0], self.num_units)
      delta = th.mm(square(input), square(self.log_to_std(self.rho))) + square(self.log_to_std(self.b_rho)).expand(input.size()[0], self.num_units)
      epsilon = th.Tensor(self.num_units, ); th.nn.init.normal_(epsilon, mean=0., std=1.)
      return self.nonlinearity(gamma + th.sqrt(delta) * epsilon)
    else:  # Calculate weights based on shifting and rescaling according to mean and variance (paper step 2)
      eps_W = th.Tensor(self.num_inputs, self.num_units); th.nn.init.normal_(eps_W, mean=0., std=1.)
      eps_b = th.Tensor(self.num_units, ); th.nn.init.normal_(eps_b, mean=0., std=1.)
      self.W, self.b = self.mu + self.log_to_std(self.rho) * eps_W, self.b_mu + self.log_to_std(self.b_rho) * eps_b
      return self.nonlinearity(th.mm(input, self.W) + self.b.expand(input.size()[0], self.num_units))


  def save_old_params(self):
    """Save old parameter values for KL calculation."""
    self.mu_old, self.b_mu_old = self.mu.clone(), self.b_mu.clone()
    self.rho_old, self.b_rho_old = self.rho.clone(), self.b_rho.clone()


  def reset_to_old_params(self):
    """Reset to old parameter values for KL calculation."""
    self.mu.data, self.b_mu.data = self.mu_old.data, self.b_mu_old.data
    self.rho.data, self.b_rho.data = self.rho_old.data, self.b_rho_old.data


  def kl_div_p_q(self, p_mean, p_std, q_mean, q_std):
    """KL divergence D_{KL}[p(x)||q(x)] for a fully factorized Gaussian"""
    numerator = square(p_mean - q_mean) + square(p_std) - square(q_std)
    denominator = 2 * square(q_std) + 1e-8
    return th.sum(numerator / denominator + th.log(q_std) - th.log(p_std))


  def kl_div_new_old(self):
    kl_div = self.kl_div_p_q(self.mu, self.log_to_std(self.rho), self.mu_old, self.log_to_std(self.rho_old))
    kl_div += self.kl_div_p_q(self.b_mu, self.log_to_std(self.b_rho), self.b_mu_old, self.log_to_std(self.b_rho_old))
    return kl_div


  def kl_div_old_new(self):
    kl_div = self.kl_div_p_q(self.mu_old, self.log_to_std(self.rho_old), self.mu, self.log_to_std(self.rho))
    kl_div += self.kl_div_p_q(self.b_mu_old, self.log_to_std(self.b_rho_old), self.b_mu, self.log_to_std(self.b_rho))
    return kl_div


  def kl_div_new_prior(self):
    kl_div = self.kl_div_p_q(self.mu, self.log_to_std(self.rho), th.tensor([0.]), self.prior_sd)
    kl_div += self.kl_div_p_q(self.b_mu, self.log_to_std(self.b_rho), th.tensor([0.]), self.prior_sd)
    return kl_div


  def kl_div_prior_new(self):
    kl_div = self.kl_div_p_q(th.tensor([0.]), self.prior_sd, self.mu,  self.log_to_std(self.rho))
    kl_div += self.kl_div_p_q(th.tensor([0.]), self.prior_sd, self.b_mu, self.log_to_std(self.b_rho))
    return kl_div



class BNN(th.nn.Module):
  """Bayesian neural network (BNN) based on Blundell2016."""

  def __init__(self, input_size, output_size, hidden_dim=[32], activation=th.nn.ReLU(), n_batches=32, n_samples=10,
    prior_sd=0.5, use_reverse_kl_reg=False, reverse_kl_reg_factor=0.1, likelihood_sd=5.0,
    compression=False, information_gain=True, learning_rate=0.0001, device='auto'):

    super(BNN, self).__init__()

    self.input_size, self.output_size = input_size, output_size
    self.hidden_dim, self.activation = hidden_dim, activation
    self.n_batches, self.n_samples = n_batches, n_samples

    self.prior_sd = th.Tensor([prior_sd])
    self.use_reverse_kl_reg = use_reverse_kl_reg
    self.reverse_kl_reg_factor = reverse_kl_reg_factor
    self.likelihood_sd = th.Tensor([likelihood_sd])
    self.compression = compression
    self.information_gain = information_gain
    assert self.information_gain or self.compression

    self.build_network() # Build network architecture.
    self.opt = th.optim.Adam(self.parameters(), lr=learning_rate)
    self.to(device)


  def build_network(self):
    gen_net = lambda sizes,activation: [BNNLayer(i, o, a, self.prior_sd) for i,o,a in zip(sizes, sizes[1:], activation)]
    self.layers = th.nn.ModuleList([ *gen_net(
      [self.input_size, *self.hidden_dim, self.output_size],
      [*( len(self.hidden_dim) + 1 ) * [self.activation], th.nn.Identity()]
    )])


  def save_old_params(self): [layer.save_old_params() for layer in self.layers]
  def reset_to_old_params(self): [layer.reset_to_old_params() for layer in self.layers]
          

  def surprise(self):
    surpr = 0.
    if self.compression: surpr += sum(l.kl_div_old_new() for l in self.layers)  # KL divergence KL[old_param||new_param]
    if self.information_gain: surpr += sum(l.kl_div_new_old() for l in self.layers)  # KL divergence KL[new_param||old_param]
    return surpr


  def _log_prob_normal(self, input, mu=th.Tensor([0.]), sigma=th.Tensor([1.])):
      return th.sum(- th.log(sigma) - th.log(th.sqrt(2 * th.Tensor([np.pi]))) - square(input - mu) / (2 * square(sigma)))


  def forward(self, x, **kwargs):
    output = x.float()
    for _, l in enumerate(self.layers): output = l(output, **kwargs)
    return output


  def loss(self, input, target):
    log_p_D_given_w = sum([self._log_prob_normal(target, self(input), self.likelihood_sd) for _ in range(self.n_samples)])
    kl = sum(l.kl_div_new_prior() for l in self.layers) # Calculate variational posterior log(q(w)) and prior log(p(w)).
    if self.use_reverse_kl_reg: kl += self.reverse_kl_reg_factor * sum(l.kl_div_prior_new() for l in self.layers)
    return kl / self.n_batches - log_p_D_given_w / self.n_samples # Calculate loss function.


  def loss_last_sample(self, input, target):
    """The difference with the original loss is that we only update based on the latest sample.
    This means that instead of using the prior p(w), we use the previous approximated posterior
    q(w) for the KL term in the objective function: KL[q(w)|p(w)] becomems KL[q'(w)|q(w)]. """
    log_p_D_given_w = sum([self._log_prob_normal(target, self(input), self.likelihood_sd) for _ in range(self.n_samples)])
    return sum(l.kl_div_new_old() for l in self.layers) - log_p_D_given_w / self.n_samples


  def fast_kl_div(self, step_size):
    assert(step_size is not None); kl_component = []
    for m in self.modules():
      if isinstance(m, BNNLayer):
        invH = square(th.log(1 + th.exp(m.rho_old)))  # compute kl for mu
        kl_component.append((square(step_size) * square(m.mu.grad.data) * invH).sum())
        invH = square(th.log(1 + th.exp(m.b_rho_old)))  # compute kl for b_mu
        kl_component.append((square(step_size) * square(m.b_mu.grad.data) * invH).sum())        
        rho, b_rho = m.rho.data, m.b_rho.data  # compute kl for rho and b_rho
        invH = 1. / 2. * (th.exp(2 * rho)) / square(1. + th.exp(rho)) / square(th.log(1. + th.exp(rho)))
        invH = 1. / 2. * (th.exp(2 * b_rho)) / square(1. + th.exp(b_rho)) / square(th.log(1. + th.exp(b_rho)))
        kl_component.append((square(step_size) * square(m.rho.grad.data) * invH).sum())
        kl_component.append((square(step_size) * square(m.b_rho.grad.data) * invH).sum())
    return sum(kl_component)


  def train_update_fn(self, input, target, second_order_update, step_size=None):
    self.opt.zero_grad(); loss = self.loss_last_sample(input, target); loss.backward()
    if second_order_update: return self.fast_kl_div(step_size)
    self.opt.step()
    return float(self.surprise().detach()) 


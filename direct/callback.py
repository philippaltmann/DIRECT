from stable_baselines3.common.callbacks import BaseCallback

class DirectCallback(BaseCallback):
  def __init__(self): 
    super(DirectCallback, self).__init__() 

  def _on_training_start(self) -> None:
    pass
      
  def _on_step(self) -> bool: 
    return True

  def _on_rollout_end(self) -> None:
    pass

"""Networks package — the shared recurrent (GRU) Q-net encoder/head trunk."""

from src.marl.nets.agent_net import RecurrentQNet
from src.marl.nets.dims import action_dim, hidden_dim, obs_dim, state_dim

__all__ = ["RecurrentQNet", "action_dim", "hidden_dim", "obs_dim", "state_dim"]

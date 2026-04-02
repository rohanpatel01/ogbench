import copy
from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import GCActor, GCValue, LogParam, CNNEncoder, MLP

class ACROAgent(flax.struct.PyTreeNode):
    """MLP head for predicting action from state transitions (inverse dynamics model)."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()


    @jax.jit
    def loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""

        obs_t = batch['observations_t']
        obs_t_k = batch['observations_t_k']
        actions_t = batch['actions_t']

        z_t = self.network.select('encoder')(obs_t, params=grad_params)
        z_t_k = self.network.select('encoder')(obs_t_k, params=grad_params)

        action_preds = self.network.select('inverse_dynamics')(jnp.concatenate([z_t, z_t_k], axis=-1), params=grad_params)

        # losses
        info = {}
        loss = 0
        
        if self.config['discrete']:
            action_loss = optax.losses.softmax_cross_entropy_with_integer_labels(
                action_preds, actions_t.squeeze(-1)
            ).mean()
        else:
            action_loss = ((action_preds - actions_t) ** 2).mean()


        loss += action_loss 
        info['action_loss'] = action_loss

        return loss, info


    
    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)

        return self.replace(network=new_network, rng=new_rng), info



    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)
        
        action_dim = ex_actions.shape[-1]
        
        ex_acro_pred_input = jnp.zeros((1, 2 * config['rep_dim']))


        # Define networks
        encoder_def = CNNEncoder(
            output_dim=config['rep_dim'],
            num_conv_filters=config['num_conv_filters'],
            kernel_size=config['kernel_size']
        )

        inverse_dynamics_def = MLP(
            hidden_dims=(*config['hidden_dims'], action_dim),
            activations=nn.activation.relu
        )
        
        # TODO: 
        network_info = dict(
            encoder=(encoder_def, (ex_observations)),
            inverse_dynamics=(inverse_dynamics_def, (ex_acro_pred_input))
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        # params = network.params
        # params['modules_target_critic'] = params['modules_critic']

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))
        


# TODO
def get_config():
    config = ml_collections.ConfigDict(
        dict(

            # Agent hyperparameters.
            agent_name='acro',  # Agent name.
            hidden_dims=(256, 256),  # Width of hidden layers for MLP for inverse dynamics model
            discrete=False,
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            layer_norm=False,  # Whether to use layer normalization.
            acro_k_step=15,  # acro k step.
            rep_dim=50,  # ACRO representation dimension.
            num_conv_filters=32,
            kernel_size=3,    
            # Dataset hyperparameters.
            dataset_class='ACRODataset',  # Dataset class name.
            p_aug=0.0,  # Probability of applying image augmentation.
            frame_stack=ml_collections.config_dict.placeholder(int),
        )
    )
    return config
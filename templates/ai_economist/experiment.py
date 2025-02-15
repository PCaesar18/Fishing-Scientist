from environment.economy import EconomyEnv
from util.logging import *
from util.util_functions import get_pareto_skill_dists
from util.callbacks import *
from environment.jax_base_env_and_wrappers import LogWrapper
import jax
import chex
import jax.numpy as jnp
import equinox as eqx
import argparse
import wandb
import numpy as np
import os
import time
import torch
import json
import optax
from typing import List, Tuple, Dict
from functools import partial
from typing import NamedTuple, Union
import distrax
from typing import Dict
from jax_tqdm import scan_tqdm

from environment.jax_base_env_and_wrappers import JaxBaseEnv, TimeStep
from gymnax.environments import spaces
from dataclasses import replace, asdict
import optax
from util.spaces import MultiDiscrete
from util.util_functions import get_gini

WANDB_PROJECT_NAME = "EconoJax"


# --- BEGIN Networks.py ---
class CustomLinear(eqx.nn.Linear):
    """ eqx.nn.Linear with optional orthogonal initialization """
    def __init__(self, orth_init, orth_scale, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if orth_init:
            weight_shape = self.weight.shape
            orth_init = jax.nn.initializers.orthogonal(orth_scale)
            self.weight = orth_init(kwargs["key"], weight_shape)
            if self.bias is not None:
                self.bias = jnp.zeros(self.bias.shape)


class ActorNetwork(eqx.Module):
    """Actor network"""

    layers: list

    def __init__(self, key, in_shape, hidden_features: List[int], num_actions, orthogonal_init=True, **kwargs):
        keys = jax.random.split(key, len(hidden_features))
        self.layers = [
            CustomLinear(orthogonal_init, np.sqrt(2), in_shape, hidden_features[0], key=keys[0])
        ] + [
            CustomLinear(orthogonal_init, np.sqrt(2), hidden_features[i], hidden_features[i+1], key=keys[i+1])
            for i in range(len(hidden_features)-1)
        ] + [
            CustomLinear(orthogonal_init, 0.01, hidden_features[-1], num_actions, key=keys[-1])
        ]

    def __call__(self, x):
        if isinstance(x, dict):
            action_mask = x["action_mask"]
            x = x["observation"]
        else: action_mask = None
        for layer in self.layers[:-1]:
            x = jax.nn.tanh(layer(x))
        logits = self.layers[-1](x)
        if action_mask is not None:
            logit_mask = jnp.ones_like(logits) * -1e8
            logits_mask = logit_mask * (1 - action_mask)
            logits = logits + logits_mask
        return logits
    
class ActorNetworkMultiDiscrete(eqx.Module):
    """'
    Actor network for a multidiscrete output space
    """

    layers: list
    output_heads: list

    def __init__(self, key, in_shape, hidden_features, actions_nvec, orthogonal_init=True, **kwargs):
        keys = jax.random.split(key, len(hidden_features))
        self.layers = [CustomLinear(orthogonal_init, np.sqrt(2), in_shape, hidden_features[0], key=keys[0])]
        for i, feature in enumerate(hidden_features[:-1]):
            self.layers.append(
                CustomLinear(orthogonal_init, np.sqrt(2), feature, hidden_features[i + 1], key=keys[i])
            )

        multi_discrete_heads_keys = jax.random.split(keys[-1], len(actions_nvec))
        try: actions_nvec = actions_nvec.tolist() # convert to list if numpy array
        except AttributeError: pass
        self.output_heads = [
            CustomLinear(orthogonal_init, np.sqrt(2), hidden_features[-1], action, key=multi_discrete_heads_keys[i])
            for i, action in enumerate(actions_nvec)
        ]
        if len(set(actions_nvec)) == 1:  # all output shapes are the same, vmap
            self.output_heads = jax.tree_util.tree_map(
                lambda *v: jnp.stack(v), *self.output_heads
            )
        else:
            raise NotImplementedError(
                "Different output shapes are not supported"
            )

    def __call__(self, x):
        if isinstance(x, dict):
            action_mask = x["action_mask"]
            x = x["observation"]
        else: action_mask = None

        def forward(head, x):
            return head(x)

        for layer in self.layers:
            x = jax.nn.tanh(layer(x))
        logits = jax.vmap(forward, in_axes=(0, None))(self.output_heads, x)

        if action_mask is not None:  # mask the logits
            logit_mask = jnp.ones_like(logits) * -1e8
            logit_mask = logit_mask * (1 - action_mask)
            logits = logits + logit_mask

        return logits
    
class ValueNetwork(eqx.Module):
    """
        Value (critic) network with a single output
        Used to output V when given a state
    """
    layers: list

    def __init__(self, key, in_shape, hidden_layers: List[int], orthogonal_init=True, **kwargs):
        keys = jax.random.split(key, len(hidden_layers))
        self.layers = [
            CustomLinear(orthogonal_init, np.sqrt(2), in_shape, hidden_layers[0], key=keys[0])
        ] + [
            CustomLinear(orthogonal_init, np.sqrt(2), hidden_layers[i], hidden_layers[i+1], key=keys[i+1])
            for i in range(len(hidden_layers)-1)
        ] + [
            CustomLinear(orthogonal_init, 0.01, hidden_layers[-1], 1, key=keys[-1])
        ]

    def __call__(self, x):
        if isinstance(x, dict):
            x = x["observation"]
        for layer in self.layers[:-1]:
            x = jax.nn.tanh(layer(x))
        return jnp.squeeze(self.layers[-1](x), axis=-1)
# --- END networks.py ---



# --- BEGIN ppo_trainer.py ---
@chex.dataclass(frozen=True)
class PpoTrainerParams:
    num_envs: int = 6
    total_timesteps: int = 1e6
    trainer_seed: int = 0
    backend: str = "cpu" #"gpu"  # or "gpu"
    num_log_episodes_after_training: int = 1
    debug: bool = True
    out_dir: str = "run_0"

    learning_rate: float = 0.0005
    anneal_lr: bool = True
    gamma: float = 0.999
    gae_lambda: float = 0.95
    max_grad_norm: float = 1.0
    clip_coef: float = 0.20
    clip_coef_vf: float = 10.0  # Depends on the reward scaling !
    ent_coef_start_pop: float = .1
    ent_coef_start_gov: float = .1
    vf_coef: float = 0.25

    num_steps: int = 150  # steps per environment
    num_minibatches: int = 6  # Number of mini-batches
    update_epochs: int = 6  # K epochs to update the policy
    # shared_policies: bool = True
    share_policy_nets: bool = True
    share_value_nets: bool = True
    network_size_pop_policy: list = eqx.field(default_factory=lambda: [128, 128])
    network_size_pop_value: list = eqx.field(default_factory=lambda: [128, 128])
    network_size_gov: list = eqx.field(default_factory=lambda: [128, 128])

    # to be filled in runtime in at init:
    batch_size: int = 0  # batch size (num_envs * num_steps)
    num_iterations: int = (
        0  # number of iterations (total_timesteps / num_steps / num_envs)
    )

    def __post_init__(self):
        object.__setattr__(
            self,
            "num_iterations",
            int(self.total_timesteps // self.num_steps // self.num_envs),
        )
        object.__setattr__(
            self, "batch_size", int(self.num_envs * self.num_steps)
        )

@chex.dataclass(frozen=True)
class Transition:
    observation: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array
    value: chex.Array
    log_prob: chex.Array
    info: chex.Array
    # next_observation: chex.Array

class AgentState(NamedTuple):
    actor: ActorNetwork
    critic: ValueNetwork
    opt_state_policy: optax.OptState
    opt_state_value: optax.OptState

class TrainState(NamedTuple):
    population: AgentState
    government: AgentState

# Jit the returned function, not this function
def build_ppo_trainer(
    env: EconomyEnv,
    trainer_params: dict,
    load_model: str = None,
):
    config = PpoTrainerParams(**trainer_params)
    eval_env = eqx.tree_at(lambda x: x.create_info, env, True)
    env = LogWrapper(env)

    pop_observation_space = env.observation_space(agent="population")
    pop_action_space = env.action_space(agent="population")
    gov_observation_space = env.observation_space(agent="government")
    gov_action_space = env.action_space(agent="government")
    num_agents = env.num_population

    # rng keys
    rng = jax.random.PRNGKey(config.trainer_seed)
    rng, pop_network_key_policy, pop_network_key_value, gov_network_key_policy, gov_network_key_value, reset_key = jax.random.split(rng, 6)

    # networks
    if config.share_policy_nets:
        pop_network_key_policy = jnp.expand_dims(pop_network_key_policy, axis=0)
    else:
        pop_network_key_policy = jax.random.split(pop_network_key_policy, num_agents)
    if config.share_value_nets:
        pop_network_key_value = jnp.expand_dims(pop_network_key_value, axis=0)
    else:
        pop_network_key_value = jax.random.split(pop_network_key_value, num_agents)
    # convert possible list of strings to list of ints
    population_actor = jax.vmap(ActorNetwork, in_axes=(0, None, None, None))(pop_network_key_policy, pop_observation_space.shape[-1], config.network_size_pop_policy, pop_action_space.n)
    population_critic = jax.vmap(ValueNetwork, in_axes=(0, None, None))(pop_network_key_value, pop_observation_space.shape[-1], config.network_size_pop_value)
    government_actor = ActorNetworkMultiDiscrete(gov_network_key_policy, gov_observation_space.shape, config.network_size_gov, gov_action_space.nvec)
    government_critic = ValueNetwork(gov_network_key_value, gov_observation_space.shape, config.network_size_gov)

    number_of_update_steps = (
        config.num_iterations * config.num_minibatches * config.update_epochs
    )
    learning_rate_schedule = optax.linear_schedule(
        init_value=config.learning_rate,
        end_value=0.00000001,
        transition_steps=number_of_update_steps,
    )
    ent_coef_schedule = {
        "population": optax.linear_schedule(
            init_value=config.ent_coef_start_pop,
            end_value=0.0,
            transition_steps=int(number_of_update_steps * 0.9),
        ),
        "government": optax.linear_schedule(
            init_value=config.ent_coef_start_gov,
            end_value=0.0,
            transition_steps=int(number_of_update_steps * 0.9),
        ),
    }

    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(
            learning_rate=(
                learning_rate_schedule if config.anneal_lr else config.learning_rate
            ),
            eps=1e-5,
        ),
    )
    population_opt_state_policy = jax.vmap(optimizer.init)(population_actor)
    population_opt_state_value = jax.vmap(optimizer.init)(population_critic)
    government_opt_state_policy = optimizer.init(government_actor)
    government_opt_state_value = optimizer.init(government_critic)

    try:
        population_actor, population_opt_state_policy = jax.tree.map(
            lambda x: jnp.squeeze(x, axis=0), (population_actor, population_opt_state_policy)
        )
    except ValueError: # Squeezing not required (individual agents)
        pass
    try:
        population_critic, population_opt_state_value = jax.tree.map(
            lambda x: jnp.squeeze(x, axis=0), (population_critic, population_opt_state_value)
        )
    except ValueError: # Squeezing not required (individual agents)
        pass
    

    train_state = TrainState(
        population=AgentState(
            actor=population_actor,
            critic=population_critic,
            opt_state_policy=population_opt_state_policy,
            opt_state_value=population_opt_state_value,
        ),
        government=AgentState(
            actor=government_actor,
            critic=government_critic,
            opt_state_policy=government_opt_state_policy,
            opt_state_value=government_opt_state_value,
        ),
    )
    if load_model:
        train_state = eqx.tree_deserialise_leaves(
            load_model, train_state
        )

    reset_key = jax.random.split(reset_key, config.num_envs)
    obs_v, env_state_v = jax.vmap(env.reset, in_axes=(0))(reset_key)

    def get_action_logits_dict(observation, train_state: Union[TrainState, eqx.Module], agent_name: str = None):
        assert isinstance(train_state, TrainState) or (isinstance(train_state, eqx.Module) and agent_name is not None)

        if isinstance(train_state, eqx.Module): # When called from the loss function
            if agent_name == "population" and config.share_policy_nets:
                return jax.vmap(train_state)(observation)
            return train_state(observation)
        
        if config.share_policy_nets:
            population_logits_fn = jax.vmap(train_state.population.actor)
        else:
            population_logits_fn = partial(jax.vmap(lambda net, obs: net(obs)), train_state.population.actor)
        government_logits_fn = train_state.government.actor
        return {
            "population": population_logits_fn(observation["population"]),
            "government": government_logits_fn(observation["government"]),
        }

    def get_value_dict(observation, train_state: Union[TrainState, eqx.Module], agent_name: str = None):
        assert isinstance(train_state, TrainState) or (isinstance(train_state, eqx.Module) and agent_name is not None)

        if isinstance(train_state, eqx.Module): # When called from the loss function
            if agent_name == "population" and config.share_value_nets:
                return jax.vmap(train_state)(observation)
            return train_state(observation)
        if config.share_value_nets:
            population_logits_fn = jax.vmap(train_state.population.critic)
        else:
            population_logits_fn = partial(jax.vmap(lambda net, obs: net(obs)), train_state.population.critic)
        government_logits_fn = train_state.government.critic
        return {
            "population": population_logits_fn(observation["population"]),
            "government": government_logits_fn(observation["government"]),
        }


    @partial(jax.jit, backend=config.backend)
    def eval_func(key: chex.PRNGKey, train_state: TrainState):
        def step_env(carry, _):
            rng, obs_v, env_state, done, episode_reward = carry
            rng, step_key, sample_key = jax.random.split(rng, 3)

            action_logits = get_action_logits_dict(obs_v, train_state)
            action_dist = jax.tree.map(distrax.Categorical, action_logits)
            actions = jax.tree.map(lambda dist: dist.sample(seed=sample_key), action_dist, is_leaf=lambda x: isinstance(x, distrax.Distribution))

            (obs_v, reward, terminated, truncated, info), env_state = eval_env.step(
                step_key, env_state, actions
            )
            episode_reward += reward["population"]

            done = jnp.any(jnp.logical_or(terminated["population"], truncated["population"]))

            if "terminal_observation" in info.keys():
                info.pop("terminal_observation")

            return (rng, obs_v, env_state, done, episode_reward), info

        rng, reset_key = jax.random.split(key)
        obs, env_state = eval_env.reset(reset_key)
        done = False
        episode_reward = jnp.zeros(num_agents)

        # we know the episode length is fixed, so lets scan
        carry, episode_stats = jax.lax.scan(
            step_env,
            (rng, obs, env_state, done, episode_reward),
            None,
            eval_env.max_steps_in_episode,
        )

        return carry[-1], episode_stats

    @partial(jax.jit, backend=config.backend)
    def train_func(rng: chex.PRNGKey = rng):

        # functions prepended with _ are called in jax.lax.scan of train_step
        aggregator = [] 
        def _env_step(runner_state, _):
            train_state, env_state, last_obs, rng = runner_state
            rng, sample_key, step_key = jax.random.split(rng, 3)

            action_logits = jax.vmap(get_action_logits_dict, in_axes=(0, None))(last_obs, train_state)
            action_dist = jax.tree.map(distrax.Categorical, action_logits)
            actions = jax.tree.map(lambda dist: dist.sample(seed=sample_key), action_dist, is_leaf=lambda x: isinstance(x, distrax.Distribution))
            log_prob = jax.tree.map(lambda dist, action: dist.log_prob(action), action_dist, actions, is_leaf=lambda x: isinstance(x, distrax.Distribution))
            value = jax.vmap(get_value_dict, in_axes=(0, None))(last_obs, train_state)  

            step_keys = jax.random.split(step_key, config.num_envs)
            (obsv, reward, terminated, truncated, info), env_state = jax.vmap(
                env.step, in_axes=(0, 0, 0)
            )(step_keys, env_state, actions)

            # SB3 hack: (would like something else, but lets leave it for now)
            next_values = jax.vmap(get_value_dict, in_axes=(0, None))(info["terminal_observation"], train_state)
            next_value = jax.tree.map(lambda v: config.gamma * v, next_values)
            reward = jax.tree.map(lambda r, v, t: r + (v * t), reward, next_value, terminated)
            
            done = jax.tree.map(lambda te, tr: jnp.logical_or(te, tr), terminated, truncated)
            transition = Transition(
                observation=last_obs,
                action=actions,
                reward=reward,
                done=done,
                value=value,
                log_prob=log_prob,
                info=info,
            )

            runner_state = (train_state, env_state, obsv, rng)
            return runner_state, transition

        def _calculate_gae(gae_and_next_values, transition):
            gae, next_value = gae_and_next_values
            value, reward, done = (
                transition.value,
                transition.reward,
                transition.done,
            )
            delta = jax.tree.map(
                lambda r, v_next, d, v: r + config.gamma * v_next * (1 - d) - v,
                reward, next_value, done, value
            )
            gae = jax.tree.map(
                lambda de, d, g: de + config.gamma * config.gae_lambda * (1 - d) * g,
                delta, done, gae
            )
            returns = jax.tree.map(jnp.add, gae, value)
            return (gae, value), (gae, returns)

        def _update_epoch(update_state, _):
            """Do one epoch of update"""

            @eqx.filter_value_and_grad(has_aux=True)
            def __ppo_policy_loss_fn(params, trajectory_minibatch, agent_name: str):
                (observations, actions, init_log_prob, init_value, advantages, returns) = trajectory_minibatch

                action_logits = jax.vmap(get_action_logits_dict, in_axes=(0, None, None))(observations, params, agent_name)
                action_dist = distrax.Categorical(logits=action_logits)
                log_prob = action_dist.log_prob(actions)
                entropy = action_dist.entropy().mean()
                if agent_name == "government": # Multidiscrete action space
                    log_prob = log_prob.sum(axis=-1)
                    init_log_prob = init_log_prob.sum(axis=-1)

                # actor loss
                ratio = jnp.exp(log_prob - init_log_prob)
                _advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )
                actor_loss1 = _advantages * ratio
                actor_loss2 = (
                    jnp.clip(ratio, 1.0 - config.clip_coef, 1.0 + config.clip_coef)
                    * _advantages
                )
                actor_loss = -jnp.minimum(actor_loss1, actor_loss2).mean()

                count = getattr(train_state, agent_name).opt_state_policy[1][0].count
                ent_coef = ent_coef_schedule[agent_name](count)
                if ent_coef.size > 1:
                    ent_coef = ent_coef[0]
                # Total loss
                total_loss = (
                    actor_loss - ent_coef * entropy
                )
                return total_loss, (actor_loss, entropy)
            
            @eqx.filter_value_and_grad(has_aux=True)
            def __ppo_critic_loss_fn(params, trajectory_minibatch, agent_name: str):
                (observations, actions, init_log_prob, init_value, advantages, returns) = trajectory_minibatch
                value = jax.vmap(get_value_dict, in_axes=(0, None, None))(observations, params, agent_name)

                # critic loss
                value_pred_clipped = init_value + (
                    jnp.clip(
                        value - init_value, -config.clip_coef_vf, config.clip_coef_vf
                    )
                )
                value_losses = jnp.square(value - returns)
                value_losses_clipped = jnp.square(value_pred_clipped - returns)
                value_loss = jnp.maximum(value_losses, value_losses_clipped).mean()

                # Total loss
                total_loss = (config.vf_coef * value_loss)
                return total_loss, (value_loss)

            def __update_over_minibatch(train_state: TrainState, minibatch):
                trajectory_mb, advantages_mb, returns_mb = minibatch
                minibatch = (
                    trajectory_mb.observation,
                    trajectory_mb.action,
                    trajectory_mb.log_prob,
                    trajectory_mb.value,
                    advantages_mb,
                    returns_mb,
                )

                for agent_state, agent_name in zip(train_state, train_state._fields):
                    if agent_name == "government" and not env.enable_government:
                        continue
                    agent_minibatch = jax.tree.map(lambda x: x[agent_name], minibatch, is_leaf=lambda i: isinstance(i, dict))
                    if agent_name == "population":
                        if not config.share_policy_nets:
                            policy_loss_fn = jax.vmap(__ppo_policy_loss_fn, in_axes=(0, 1, None))
                            policy_update_fn = jax.vmap(optimizer.update)
                        else:
                            policy_loss_fn = __ppo_policy_loss_fn
                            policy_update_fn = optimizer.update
                        if not config.share_value_nets:
                            value_loss_fn = jax.vmap(__ppo_critic_loss_fn, in_axes=(0, 1, None))
                            value_update_fn = jax.vmap(optimizer.update)
                        else:
                            value_loss_fn = __ppo_critic_loss_fn
                            value_update_fn = optimizer.update
                    else:
                        policy_loss_fn = __ppo_policy_loss_fn
                        value_loss_fn = __ppo_critic_loss_fn
                        policy_update_fn = optimizer.update
                        value_update_fn = optimizer.update

                    # update policy
                    (total_loss, (actor_loss, entropy)), grads = policy_loss_fn(
                        agent_state.actor, agent_minibatch, agent_name
                    )
                    updates, new_policy_opt_state = policy_update_fn(
                        grads, agent_state.opt_state_policy
                    )
                    new_policy_networks = optax.apply_updates(
                        agent_state.actor, updates
                    )
                    # update value
                    (total_loss, (value_loss)), grads = value_loss_fn(
                        agent_state.critic, agent_minibatch, agent_name
                    )
                    updates, new_value_opt_state = value_update_fn(
                        grads, agent_state.opt_state_value
                    )
                    new_value_networks = optax.apply_updates(
                        agent_state.critic, updates
                    )

                    train_state = train_state._replace(
                        **{agent_name: AgentState(
                            actor=new_policy_networks,
                            critic=new_value_networks,
                            opt_state_policy=new_policy_opt_state,
                            opt_state_value=new_value_opt_state,
                        )}
                    )

                return train_state, (total_loss, actor_loss, value_loss, entropy)

            train_state, trajectory_batch, advantages, returns, rng = update_state
            rng, key = jax.random.split(rng)

            batch_idx = jax.random.permutation(key, config.batch_size)
            batch = (trajectory_batch, advantages, returns)

            # reshape (flatten)
            batch = jax.tree_util.tree_map(
                lambda x: x.reshape((config.batch_size,) + x.shape[2:]), batch
            )
            # take from the batch in a new order (the order of the randomized batch_idx)
            shuffled_batch = jax.tree_util.tree_map(
                lambda x: jnp.take(x, batch_idx, axis=0), batch
            )
            # split in minibatches
            minibatches = jax.tree_util.tree_map(
                lambda x: x.reshape((config.num_minibatches, -1) + x.shape[1:]),
                shuffled_batch,
            )
            # update over minibatches
            train_state, losses = jax.lax.scan(
                __update_over_minibatch, train_state, minibatches
            )
            update_state = (train_state, trajectory_batch, advantages, returns, rng)
            return update_state, losses

        def train_step(runner_state, _):
            # Do rollout of single trajactory (num_steps)
            train_state, env_state, last_obs, rng, aggregator = runner_state
            # 1) run environment rollouts for config.num_steps
            runner_state_no_agg = (train_state, env_state, last_obs, rng)
            runner_state_no_agg, trajectory_batch = jax.lax.scan(
                _env_step, runner_state_no_agg, None, config.num_steps
            )
            (train_state, env_state, last_obs, rng) = runner_state_no_agg

            # calculate gae
            last_value = jax.vmap(get_value_dict, in_axes=(0, None))(last_obs, train_state)
            _, (advantages, returns) = jax.lax.scan(
                _calculate_gae,
                (jax.tree.map(jnp.zeros_like, last_value), last_value),
                trajectory_batch,
                reverse=True,
                unroll=16,
            )

            # Do update epochs
            update_state = (train_state, trajectory_batch, advantages, returns, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config.update_epochs
            )

            train_state = update_state[0]    
            rng = update_state[-1]
            
            metric = trajectory_batch.info
            metric["loss_info"] = loss_info
            metric["num_envs"] = config.num_envs
            
            train_dir = os.path.join(config.out_dir, "training_logs")
            os.makedirs(train_dir, exist_ok=True)
            
            jax.debug.callback(log_callback, metric, train_dir, aggregator) 
            if wandb.run:
                print("DEBUG: we get into the WANDB callback loop")
                jax.debug.callback(wandb_callback, metric, aggregator)  # probs don't need the aggregator anymore
            else:
                #print("DEBUG: we get into the NORMAL callback loop")
                jax.debug.callback(logwrapper_callback, metric, train_dir, aggregator) 
            
            runner_state = (train_state, env_state, last_obs, rng, aggregator)
            return runner_state, metric

        rng, key = jax.random.split(rng)
        runner_state = (train_state, env_state_v, obs_v, key, aggregator)
        if not config.debug:
            train_step = scan_tqdm(config.num_iterations)(train_step)
        runner_state, metrics = jax.lax.scan(
            train_step, runner_state, jnp.arange(config.num_iterations)
        )
        trained_train_state = runner_state[0]
        rng = runner_state[-2]
        aggregator = runner_state[-1] 

        return {
            "train_state": trained_train_state,
            "train_metrics": aggregator,
        }

    return train_func, eval_func

# --- END ppo_trainer.py ---

# ---BEGIN economy.py ---


# disable jit
# jax.config.update("jax_disable_jit", True)

@chex.dataclass(frozen=True)
class EnvState:
    
    utility: dict[str, chex.Array]
    productivity: float
    equality: float

    inventory_coin: chex.Array  # shape: (num_population,)
    inventory_labor: chex.Array # shape: (num_population,)
    inventory_resources: chex.Array # shape: (num_population, num_resources)

    # temp inventory for trading
    escrow_coin: chex.Array # shape: (num_population,)
    escrow_resources: chex.Array # shape: (num_population, num_resources)

    skills_craft: chex.Array  # shape: (num_population,)
    skills_gather_resources: chex.Array # shape: (num_population, num_resources)

    market_orders: chex.Array # shape: (expiry_time, (num_resources * 2), num_population) # 2 for buy and sell
    trade_price_history: chex.Array # shape: (expiry_time, num_resources) 

    tax_rates: chex.Array # shape: (len(tax_bracket_cutoffs) - 1, )
    start_year_inventory_coin: chex.Array # shape: (num_population,) # to calculate income of the current year
    income_this_period_pre_tax: chex.Array # shape: (num_population,)
    income_prev_period_pre_tax: chex.Array # shape: (num_population,)
    marginal_income: chex.Array # shape: (num_population,) # income per earned coin after tax
    net_tax_payed_prev_period: chex.Array # shape: (num_population,) 

    timestep: int = 0


DEFAULT_NUM_POPULATION = 12
DEFAULT_NUM_RESOURCES = 2 # e.g. stone and wood
class EconomyEnv(JaxBaseEnv):

    name: str = "economy_env"
    seed: int = 0

    num_population: int = DEFAULT_NUM_POPULATION # Excluding the government
    num_resources: int = DEFAULT_NUM_RESOURCES
    max_steps_in_episode: int = 1000
    tax_period_length: int = 100
    enable_government: bool = True # Enable setting tax rates
    allow_noop: bool = True
    create_info: bool = False
    insert_agent_ids: bool = False

    starting_coin: int = 15
    init_craft_skills: np.ndarray = None
    init_gather_skills: np.ndarray = None
    base_skill_development_multiplier: float = .0 # Allow skills to improve by performing actions (0 == no improvement)
    max_skill_level: float = 5
    
    trade_expiry_time: int = 30
    max_orders_per_agent: int = 15
    possible_trade_prices: np.ndarray = eqx.field(converter=np.asarray, default_factory=lambda: np.arange(1, 11, step=2, dtype=np.int8))
    
    coin_per_craft: int = 20 # fixed multiplier of the craft skill
    gather_labor_cost: int = 1
    craft_labor_cost: int = 1
    trade_labor_cost: int = 0.05
    craft_diff_resources_required: int = 2 # 0 = log2(num_resources) rounded down
    craft_num_resource_required: int = 2 # Requirements per resource

    tax_bracket_cutoffs: np.ndarray = eqx.field(converter=np.asarray, default_factory=lambda: np.array([0, 380.980, 755.188, np.inf])) # Dutch tax bracket (scaled down by 100)

    isoelastic_eta: float = 0.27
    labor_coefficient: float = 1
        
    @property
    def num_agents(self):
        return self.num_population + 1
    @property
    def trade_actions_per_resource(self):
        # return 2 * (self.trade_price_ceiling - self.trade_price_floor + 1) # 2 for buy and sell
        return 2 * len(self.possible_trade_prices)
    @property
    def trade_actions_total(self):
        return self.num_resources * self.trade_actions_per_resource
    def gather_resource_action_index(self, resource_index: int): # 0-indexed
        return self.trade_actions_total + resource_index
    @property
    def craft_action_index(self):
        return self.trade_actions_total + self.num_resources
    
    def __post_init__(self):
        if self.craft_diff_resources_required == 0:
            diff_required = int(np.log2(self.num_resources))
            diff_required = np.clip(diff_required, 0, self.num_resources)
            self.__setattr__("craft_diff_resources_required", int(diff_required))
        
        key = jax.random.PRNGKey(self.seed)
        if self.init_craft_skills is None:
            init_craft_skills = jax.random.normal(key, shape=(self.num_population,)) + 1
            init_craft_skills = jnp.clip(init_craft_skills, 0, self.max_skill_level)
            self.__setattr__("init_craft_skills", init_craft_skills)
        if self.init_gather_skills is None:
            init_gather_skills = jax.random.normal(key, shape=(self.num_population, self.num_resources)) + 1
            init_gather_skills = jnp.clip(init_gather_skills, 0, self.max_skill_level)
            self.__setattr__("init_gather_skills", init_gather_skills)

    def __check_init__(self):
        assert self.name
        assert self.num_population > 0
        assert self.max_steps_in_episode > 0
        assert 1 <= self.tax_period_length < self.max_steps_in_episode
        assert len(self.init_gather_skills) == self.num_population
        assert len(self.init_craft_skills) == self.num_population
        # assert (self.trade_actions_per_resource / 2) % 2 == 0, "The number of sell and buy actions per resource should be even"
        assert self.tax_bracket_cutoffs[0] == 0, "The first tax bracket should start at 0"
        assert self.tax_bracket_cutoffs[-1] == np.inf, "The last tax bracket should be infinity"
        assert np.all(np.diff(self.tax_bracket_cutoffs) > 0), "Tax brackets should be sorted in ascending order"
        assert self.craft_diff_resources_required >= 0 and self.craft_diff_resources_required <= self.num_resources
        assert np.all(self.possible_trade_prices > 0), "Trade prices should be positive"
        assert np.all(np.diff(self.possible_trade_prices) > 0), "Trade prices should be sorted in ascending order"

    @eqx.filter_jit
    def reset_env(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], EnvState]:
        start_coin = jnp.ones(self.num_population, dtype=jnp.int32) * self.starting_coin
        state = EnvState(
            utility={
                "population": jnp.zeros(self.num_population),
                "government": 0.
            },
            inventory_coin=start_coin,
            inventory_labor=jnp.zeros(self.num_population),
            inventory_resources=jnp.zeros((self.num_population, self.num_resources), dtype=jnp.int16),
            escrow_coin=jnp.zeros(self.num_population, dtype=jnp.int32),
            escrow_resources=jnp.zeros((self.num_population, self.num_resources), dtype=jnp.int16),
            skills_craft=self.init_craft_skills,
            skills_gather_resources=self.init_gather_skills,
            market_orders=jnp.zeros((self.trade_expiry_time, self.num_resources, 2, self.num_population), dtype=jnp.int8), # 2 for buy and sell
            trade_price_history=jnp.zeros((self.trade_expiry_time, self.num_resources), dtype=jnp.float16),
            tax_rates=jnp.zeros(len(self.tax_bracket_cutoffs) - 1, dtype=jnp.float32),
            start_year_inventory_coin=start_coin,
            income_this_period_pre_tax=jnp.zeros(self.num_population, dtype=jnp.int32),
            income_prev_period_pre_tax=jnp.zeros(self.num_population, dtype=jnp.int32),
            marginal_income=jnp.ones(self.num_population, dtype=jnp.float32),
            net_tax_payed_prev_period=jnp.zeros(self.num_population, dtype=jnp.int32),
            timestep=0,
            productivity=0.,
            equality=0.
        )
        state = self.calculate_utilities(state)
        observations = self.get_observations_and_action_masks(state)
        return observations, state
    
    @eqx.filter_jit
    def step_env(self, rng: chex.PRNGKey, state: EnvState, actions: Dict[str, chex.Array]) -> Tuple[TimeStep, EnvState]:

        new_state = self.update_state(state, actions, rng)
        observations = self.get_observations_and_action_masks(new_state)
        reward = self.get_rewards(state, new_state)
        terminated, truncated = self.get_terminal_truncated(new_state)
        info = self.get_info(new_state, actions)

        return TimeStep(observations, reward, terminated, truncated, info), new_state
    
    def get_observations_and_action_masks(self, state: EnvState) -> Dict[str, chex.Array]:
        obs = self.get_observations(state)
        action_masks = self.get_action_masks(state)
        observations = {
            key: {"observation": obs[key], "action_mask": action_masks[key]} 
            for key in obs
        }
        return observations

    
    @eqx.filter_jit
    def get_observations(self, state: EnvState) -> Dict[str, chex.Array]:
        """
            Should take in a state and return an observation array of shape (num_population, num_features)
            Also returns the action mask of shape (num_population, num_actions)
        """

        is_tax_day = state.timestep % self.tax_period_length == 0 if self.enable_government else 0

        ### Market info
        # state.market_orders = of shape (expiry_time, num_resources, 2, num_population)
        # for each resource we need the highest buy per agent and lowest sell per agent
        highest_order_per_resource_per_agent = jnp.max(state.market_orders, axis=0)
        lowest_order_per_resource_per_agent = jnp.min(state.market_orders, axis=0)
        highest_buy_order_per_resource_per_agent = highest_order_per_resource_per_agent[:, 0, :]
        lowest_sell_order_per_resource_per_agent = lowest_order_per_resource_per_agent[:, 1, :]
        highest_buy_order_per_resource = jnp.max(highest_buy_order_per_resource_per_agent, axis=1)
        lowest_sell_order_per_resource = jnp.min(lowest_sell_order_per_resource_per_agent, axis=1)
        average_prices = jnp.nan_to_num(
            jnp.mean(state.trade_price_history, axis=0, where=state.trade_price_history != 0),
            nan=0
        )

        private_observations = [ 
            state.inventory_coin / 1000,
            state.inventory_resources / 10,
            state.inventory_labor / 100,
            state.escrow_coin / 1000,
            state.escrow_resources,
            state.skills_craft,
            state.skills_gather_resources,
            highest_buy_order_per_resource_per_agent.T,
            lowest_sell_order_per_resource_per_agent.T,
            state.marginal_income,
            state.income_this_period_pre_tax / 1000,
            state.income_prev_period_pre_tax / 1000,
        ]
        if self.insert_agent_ids:
            agent_ids = np.arange(self.num_population)
            binary_agent_ids = ((agent_ids[:, None] & (1 << np.arange(self.num_population.bit_length()))) > 0).astype(int)[:, ::-1]
            private_observations.append(binary_agent_ids)
        private_observations = jnp.column_stack(private_observations)

        global_observations = jnp.hstack([
            # state.timestep,
            state.tax_rates,
            highest_buy_order_per_resource,
            lowest_sell_order_per_resource,
            average_prices,
            is_tax_day,
        ])
        global_observations = jnp.broadcast_to(global_observations, (self.num_population, global_observations.shape[0]))

        observations_population = jnp.hstack([private_observations, global_observations])

        
        # the number of agents that are in each of the brackets:
        highest_brackets_per_agent = jnp.digitize(state.income_this_period_pre_tax, self.tax_bracket_cutoffs) - 1
        count_per_bracket = jnp.bincount(highest_brackets_per_agent, length=len(self.tax_bracket_cutoffs)-1)

        observation_government = jnp.concatenate([
            jnp.array([
                is_tax_day, 
                state.inventory_coin.mean() / 1000,
                state.inventory_coin.std() / 1000,
                jnp.median(state.inventory_coin) / 1000,
                state.start_year_inventory_coin.mean() / 1000,
                state.start_year_inventory_coin.std() / 1000,
                jnp.median(state.start_year_inventory_coin) / 1000,
                state.income_this_period_pre_tax.mean() / 1000,
                state.income_this_period_pre_tax.std() / 1000,
                jnp.median(state.income_this_period_pre_tax) / 1000,
            ]),
            count_per_bracket / self.num_population,
            state.tax_rates,
            average_prices / 10,
        ]).flatten()

        return {
            "population": observations_population,
            "government": observation_government
        }

    @eqx.filter_jit
    def get_action_masks(self, state: EnvState) -> chex.Array:
        # For convinience, trade actions will be the first actions, this is helpful in the trade_action_processing function
        coin_inventory = state.inventory_coin
        resources_inventory = state.inventory_resources 

        ### trade
        # prices = np.arange(self.trade_price_floor, self.trade_price_ceiling + 1)
        prices = self.possible_trade_prices
        num_trade_actions = int(self.trade_actions_per_resource / 2)
        # Buy orders (need enough coin)
        buy_resource_masks = (coin_inventory[:, None] >= prices).astype(jnp.bool)
        buy_resource_masks = jnp.expand_dims(buy_resource_masks, axis=1).repeat(self.num_resources, axis=1)
        # Sell orders (need at least 1 resource)
        sell_resource_masks = (resources_inventory >= 1).astype(jnp.bool)[:, :, None].repeat(num_trade_actions, axis=-1)
        trade_masks = jnp.concatenate([buy_resource_masks, sell_resource_masks], axis=-1).reshape(self.num_population, -1)

        # agent can't trade if they reached the max number of orders
        num_orders_per_agent = jnp.count_nonzero(state.market_orders, axis=(0, 1, 2))
        orders_remaining = (num_orders_per_agent <= self.max_orders_per_agent)[:, None]
        trade_masks = trade_masks * orders_remaining

        ### gather: always available
        gather_masks = jnp.ones((self.num_population, self.num_resources), dtype=jnp.int16)

        ### craft
        # craft_masks = (resources_inventory >= self.craft_num_resource_required).all(axis=1)[:, None].astype(jnp.bool)
        craft_masks = (
            (resources_inventory >= self.craft_num_resource_required).sum(axis=1) >= self.craft_diff_resources_required
        )[:, None].astype(jnp.bool)

        population_masks = jnp.concatenate([trade_masks, gather_masks, craft_masks], axis=1)

        if self.allow_noop:
            do_nothing_masks = jnp.ones((self.num_population, 1), dtype=jnp.bool)
            population_masks = jnp.concatenate([population_masks, do_nothing_masks], axis=1)

        # Gov actions: won't have any effect when not a tax day
        gov_action_space_nvec = self.action_space("government").nvec
        government_masks = jnp.ones((len(gov_action_space_nvec), gov_action_space_nvec[0]), dtype=jnp.bool)

        return {
            "population": population_masks,
            "government": government_masks
        }
    

    @eqx.filter_jit
    def get_rewards(self, old_state: EnvState, new_state: EnvState) -> chex.Array:
        """ Returns the difference in utility for a timestep as the reward """
        return jax.tree.map(
            lambda x, y: y - x, old_state.utility, new_state.utility
        )
        
    @eqx.filter_jit
    def get_terminal_truncated(self, state: EnvState) -> Tuple[bool, bool]:
        terminated = False # no termination
        truncated = state.timestep >= self.max_steps_in_episode
        return ({
            "population": jnp.broadcast_to(terminated, (self.num_population,)),
            "government": terminated
        }, {
            "population": jnp.broadcast_to(truncated, (self.num_population,)),
            "government": truncated
        })
    
    @eqx.filter_jit
    def get_info(self, state: EnvState, actions) -> Dict[str, chex.Array]:
        if not self.create_info:
            return {
                "coin": state.inventory_coin,
                "labor": state.inventory_labor,
                "productivity": state.productivity,
                "equality": state.equality,
            }
        state_dict = asdict(state)
        state_dict.update({"population_actions": actions["population"]})
        state_dict.update({"government_actions": actions["government"]})
        state_dict.update({"population_utility": state.utility["population"]})
        state_dict.update({"government_utility": state.utility["government"]})
        info_keys = [
            # "escrow_coin",
            # "escrow_resources",
            "inventory_coin",
            "inventory_labor",
            # "inventory_resources",
            "skills_craft",
            # "skills_gather_resources",
            "population_utility",
            # "government_utility",
            "population_actions",
            # "government_actions",
            "timestep",
            "tax_rates",
            "trade_price_history",
            "productivity",
            "equality",
            # "income_this_period_pre_tax",
            # "income_prev_period_pre_tax",
            # "marginal_income",
            # "net_tax_payed_prev_period",
        ]
        info = {}
        for key in info_keys:
            if state_dict[key].shape == (self.num_population,):
                info[key] = {
                    i: state_dict[key][i] for i in range(self.num_population)
                }
            elif state_dict[key].shape == (self.num_population, self.num_resources):
                info[key] = {}
                for r in range(self.num_resources):
                    info[key][r] = {
                        i: state_dict[key][i, r] for i in range(self.num_population)
                    }
            elif key == "government_actions" or key == "tax_rates":
                info[key] = {
                    i: state_dict[key][i] for i in range(len(state_dict[key]))
                }
            elif key == "trade_price_history":
                info[key] = jnp.nan_to_num(
                    jnp.mean(state_dict[key], axis=0, where=state_dict[key] != 0),
                    nan=0
                )
                info[key] = {
                    i: info[key][i] for i in range(len(info[key]))
                }
            else:
                info[key] = state_dict[key]
        return info

    @eqx.filter_jit
    def update_state(self, state: EnvState, action: Dict[str, chex.Array], rng: chex.PRNGKey) -> EnvState:
        gather_key, trade_key = jax.random.split(rng, 2)

        # POPULATION
        population_actions = action["population"] # an array with a discrete action for each agent
        state = self.component_gather_and_craft(state, population_actions, gather_key)
        state = self.component_trading(state, population_actions, trade_key)

        # GOVERNMENT
        # NOTE: component taxation also updates parts of the observation that are used, even when government is disabled
        government_action = action["government"] # an array with a multi-discrete action for the government
        state = self.component_taxation(state, government_action)

        # Calculate utilities
        state = self.calculate_utilities(state)

        return state.replace(
            timestep=state.timestep + 1,
        )
    
    def calculate_utilities(self, state: EnvState) -> Dict[str, chex.Array]:
        """
            Utility functions per agent, dictating the rewards.
            Rewards are the difference in utility per timestep.
            These utility functions follow that of the AI-economist:
            https://github.com/salesforce/ai-economist/blob/master/ai_economist/foundation/scenarios/utils/rewards.py
        """
        agent_total_coin = state.inventory_coin + state.escrow_coin

        # Population utility: isoelastic utility minus labor
        # https://en.wikipedia.org/wiki/Isoelastic_utility
        util_consumption = (agent_total_coin ** (1 - self.isoelastic_eta) - 1) / (1 - self.isoelastic_eta)
        util_labor = state.inventory_labor * self.labor_coefficient
        util_population = util_consumption - util_labor

        # Government utility: productivity * equality
        EQUALITY_WEIGHT = 1.0 # 0 (U = prod) and 1 (U = prod * eq)
        productivity = jnp.sum(agent_total_coin) / self.num_population
        equality = 1 - get_gini(agent_total_coin)
        equality_weighted = EQUALITY_WEIGHT * equality + (1 - EQUALITY_WEIGHT)
        util_government = equality_weighted * productivity

        return state.replace(
            utility = {
                "population": util_population,
                "government": util_government
            },
            productivity=productivity,
            equality=equality
        )
    
    @eqx.filter_jit
    def component_gather_and_craft(self, state: EnvState, agent_actions: chex.Array, key: chex.PRNGKey) -> EnvState:
        gather_resource_action_indices = np.arange(self.num_resources) + self.gather_resource_action_index(0)

        key, luck_key = jax.random.split(key)
        skill_success = jax.random.uniform(luck_key, (self.num_population,), maxval=1.1)
        labor_inventories = state.inventory_labor
        resource_inventories = state.inventory_resources
        coin_inventories = state.inventory_coin
        
        gather_actions = (agent_actions == gather_resource_action_indices[:, None]).T # (num_population, num_resources)
        # gather_bonus = gather_actions * (jnp.floor(state.skills_gather_resources)).astype(jnp.int16)
        gather_bonus = gather_actions * (jnp.floor(state.skills_gather_resources + skill_success[:, None])).astype(jnp.int16)
        resource_inventories += (gather_bonus).astype(jnp.int16)
        labor_inventories += jnp.any(gather_actions, axis=1) * self.gather_labor_cost
        
        craft_actions = agent_actions == self.craft_action_index
        craft_actions = craft_actions # & (resource_inventories >= self.craft_num_resource_required).all(axis=1)
        craft_gains = craft_actions * (self.coin_per_craft * state.skills_craft).astype(jnp.int32)
        coin_inventories += craft_gains.astype(jnp.int32)

        # Crafting:
        can_craft = (resource_inventories >= self.craft_num_resource_required).sum(axis=1) >= self.craft_diff_resources_required
        will_craft = craft_actions * can_craft

        # Fixed arrays:
        highest_resource_indices = jnp.argsort(resource_inventories, axis=1, descending=True)
        resources_to_craft_with = jnp.zeros(resource_inventories.shape, dtype=bool)
        resources_to_craft_with = resources_to_craft_with.at[np.arange(self.num_population)[:, None], highest_resource_indices[:, :self.craft_diff_resources_required]].set(True)
        resource_changes = resources_to_craft_with * self.craft_num_resource_required * will_craft[:, None]

        resource_inventories -= resource_changes
        # resource_inventories -= (craft_actions * self.craft_num_resource_required)[:, None].astype(jnp.int16)
        labor_inventories += (will_craft * self.craft_labor_cost)

        # Skill development
        if self.base_skill_development_multiplier > 0:
            sched = optax.exponential_decay(
                init_value=self.base_skill_development_multiplier, 
                transition_steps=5, 
                decay_rate=0.005
            ) # learning with high skill is harder
            gather_multiplier = sched(state.skills_gather_resources)
            craft_multiplier = sched(state.skills_craft)
            gather_skill_development = (gather_actions * gather_multiplier) + 1
            skills_gather_resources = jnp.minimum(state.skills_gather_resources * gather_skill_development, self.max_skill_level)
            craft_skill_development = (will_craft * craft_multiplier) + 1
            skills_craft = jnp.minimum(state.skills_craft * craft_skill_development, self.max_skill_level)
            state = state.replace(
                skills_gather_resources=skills_gather_resources,
                skills_craft=skills_craft
            )

        return state.replace(
            inventory_coin=coin_inventories,
            inventory_labor=labor_inventories,
            inventory_resources=resource_inventories
        )

    @eqx.filter_jit
    def component_trading(self, state: EnvState, actions: chex.Array, key: chex.PRNGKey) -> EnvState:
        trade_time_index = state.timestep % self.trade_expiry_time
        # prices = np.arange(self.trade_price_floor, self.trade_price_ceiling + 1, dtype=jnp.int16)
        prices = self.possible_trade_prices
        num_trade_actions = self.trade_actions_total

        coin_inventory = state.inventory_coin
        coin_escrow = state.escrow_coin
        resource_inventory = state.inventory_resources
        resource_escrow = state.escrow_resources
        labor_inventory = state.inventory_labor
        market_orders = state.market_orders # (expiry_time, num_resources, 2, num_population)
        trade_price_history = state.trade_price_history

        ### Expire previous bids/asks: return escrow to inventory
        expired_orders = market_orders[trade_time_index]
        expired_buys = expired_orders[:, 0, :]
        expired_sells = expired_orders[:, 1, :]
        coin_inventory += jnp.sum(expired_buys, axis=0, dtype=jnp.int32)
        coin_escrow -= jnp.sum(expired_buys, axis=0, dtype=jnp.int32)
        resource_inventory += expired_sells.astype(jnp.bool).T
        resource_escrow -= expired_sells.astype(jnp.bool).T

        ### Process actions (new bids/asks)
        # create a one-hot encoded matrix from the actions array (the first num_actions in actions are trade actions)
        one_hot_market_actions = jax.nn.one_hot(actions, num_trade_actions, dtype=jnp.int8) # (num_population, num_trade_actions)
        one_hot_market_actions *= jnp.tile(prices, num_trade_actions // len(prices)) # correct for pricing
        market_actions = one_hot_market_actions.reshape(self.num_population, self.num_resources, 2, -1).sum(axis=-1, dtype=jnp.int8)
        market_actions = jnp.moveaxis(market_actions, 0, -1).astype(jnp.int8) # (num_resources, 2, num_population) (these are the action of this timestep)

        # update market state:
        market_orders = market_orders.at[trade_time_index].set(market_actions)

        # update inventories
        agents_spend_on_buys = jnp.sum(market_actions[:, 0, :], axis=0, dtype=jnp.int32) # (num_population,)
        agents_that_sell_per_resource = jnp.array(market_actions[:, 1, :], dtype=jnp.bool) # (num_resources, num_population)
        agents_that_traded = jnp.any(market_actions, axis=(0, 1)) # (num_population,)

        coin_inventory -= agents_spend_on_buys
        coin_escrow += agents_spend_on_buys
        resource_inventory -= agents_that_sell_per_resource.T
        resource_escrow += agents_that_sell_per_resource.T
        labor_inventory += agents_that_traded * self.trade_labor_cost
                     
        # Make sure the oldest orders are the first in the array (so that they are prioritized):
        market_orders_rolled = jnp.roll(market_orders, -(trade_time_index+1), axis=0)

        def process_resource_orders(resource_orders: chex.Array):
            reordered_orders = jnp.squeeze(resource_orders, axis=1)
            # randomize agent order such that final tie breaker is random
            random_agent_order = jax.random.permutation(key, np.arange(self.num_population))
            reordered_orders = reordered_orders[:, :, random_agent_order]
            reordered_orders = jnp.moveaxis(reordered_orders, 1, 0)
            reordered_orders = reordered_orders.reshape(2, -1)
            agent_ids = jnp.tile(random_agent_order, self.trade_expiry_time)
            trade_times = jnp.repeat(np.arange(self.trade_expiry_time), self.num_population)

            buy_orders = reordered_orders[0]
            sell_orders = reordered_orders[1]
            order_of_buys = jnp.argsort(buy_orders, descending=True)#[::-1]
            order_of_sells = jnp.argsort(jnp.where(sell_orders == 0, jnp.inf, sell_orders))
            buy_orders = buy_orders[order_of_buys]
            sell_orders = sell_orders[order_of_sells]
            buy_agent_ids = agent_ids[order_of_buys]
            sell_agent_ids = agent_ids[order_of_sells]
            buy_trade_times = trade_times[order_of_buys]
            sell_trade_times = trade_times[order_of_sells]

            valid_trades = (buy_orders >= sell_orders) & (buy_orders != 0) & (sell_orders != 0)
            trade_prices = (jax.lax.select(sell_trade_times > buy_trade_times, sell_orders, buy_orders)) * valid_trades
            buyers_discounts = (buy_orders - trade_prices) * valid_trades
            num_resources_traded = valid_trades.astype(jnp.int16)

            # where num_resources_trades == 0, the trade did not occur
            # set buy_agent_ids and sell_agent_ids to self.num_population + 2 (out of bounds)
            # JAX will ignore these indices
            buy_agent_ids += (self.num_population * 2) * ~valid_trades
            sell_agent_ids += (self.num_population * 2) * ~valid_trades

            coin_inventory_changes = jnp.zeros(self.num_population, dtype=jnp.int32)
            coin_escrow_changes = jnp.zeros(self.num_population, dtype=jnp.int32)
            resource_inventory_changes = jnp.zeros(self.num_population, dtype=jnp.int16)
            resource_escrow_changes = jnp.zeros(self.num_population, dtype=jnp.int16)

            coin_inventory_changes = coin_inventory_changes.at[sell_agent_ids].add(trade_prices)
            coin_inventory_changes = coin_inventory_changes.at[buy_agent_ids].add(buyers_discounts)
            coin_escrow_changes = coin_escrow_changes.at[buy_agent_ids].add(-(trade_prices + buyers_discounts))
            resource_inventory_changes = resource_inventory_changes.at[buy_agent_ids].add(num_resources_traded)
            resource_escrow_changes = resource_escrow_changes.at[sell_agent_ids].add(-num_resources_traded)

            inventory_changes = jnp.stack([
                coin_inventory_changes,
                coin_escrow_changes,
                resource_inventory_changes,
                resource_escrow_changes
            ], axis=1)

            # Price history
            avg_trade_price_this_step = jnp.nan_to_num(
                jnp.mean(trade_prices, where=trade_prices != 0), nan=0
            )

            # update the market itself, set occured trades to 0
            resource_orders = resource_orders.at[buy_trade_times, 0, 0, buy_agent_ids].set(0)
            resource_orders = resource_orders.at[sell_trade_times, 0, 1, sell_agent_ids].set(0)

            return (inventory_changes, avg_trade_price_this_step, resource_orders)

        per_resource_orders = jnp.split(market_orders_rolled, self.num_resources, axis=1)
        orders_out = jax.tree.map(process_resource_orders, per_resource_orders)

        inventory_changes = jnp.stack([x[0] for x in orders_out], axis=1)
        total_coin_inventory_changes = inventory_changes[:, :, 0].sum(axis=1)
        total_coin_escrow_changes = inventory_changes[:, :, 1].sum(axis=1)
        resource_inventory_changes = inventory_changes[:, :, 2].astype(jnp.int16)
        resource_escrow_changes = inventory_changes[:, :, 3].astype(jnp.int16)
        avg_trade_price_this_step = jnp.array([x[1] for x in orders_out], dtype=jnp.float16)
        market_orders_rolled = jnp.concatenate([x[2] for x in orders_out], axis=1)

        coin_inventory += total_coin_inventory_changes
        coin_escrow += total_coin_escrow_changes
        resource_inventory += resource_inventory_changes
        resource_escrow += resource_escrow_changes
        trade_price_history = trade_price_history.at[trade_time_index].set(avg_trade_price_this_step)

        # roll back to original order
        market_orders = jnp.roll(market_orders_rolled, (trade_time_index+1), axis=0)

        return state.replace(
            inventory_coin=coin_inventory,
            escrow_coin=coin_escrow,
            inventory_resources=resource_inventory,
            escrow_resources=resource_escrow,
            inventory_labor=labor_inventory,
            market_orders=market_orders,
            trade_price_history=trade_price_history
        )
    
    def component_taxation(self, state: EnvState, actions: chex.Array) -> EnvState:

        def process_tax_day(state: EnvState):
            inventory_coin = state.inventory_coin
            year_income_per_agent = income_prev_period_pre_tax
            tax_rates = state.tax_rates
            tax_bracket_cutoffs = self.tax_bracket_cutoffs

            income_in_tax_bracket = jnp.clip(year_income_per_agent[:, None] - tax_bracket_cutoffs[:-1], 0, tax_bracket_cutoffs[1:] - tax_bracket_cutoffs[:-1])
            tax_in_brackets_per_agent = income_in_tax_bracket * tax_rates
            total_tax_due_per_agent = tax_in_brackets_per_agent.sum(axis=-1)
            # can't pay more than you have in inventory (escrow is not considered)
            total_tax_due_per_agent = jnp.minimum(total_tax_due_per_agent, inventory_coin).astype(jnp.int32)
            total_tax_due = total_tax_due_per_agent.sum()
            taxes_to_distribute = (total_tax_due // self.num_population).astype(jnp.int32) # Uniform distribution of taxes

            # Collect taxes and redistribute
            inventory_coin -= total_tax_due_per_agent
            inventory_coin += taxes_to_distribute
            net_tax_payed = total_tax_due_per_agent - taxes_to_distribute

            # Now set the new tax rates according to the actions
            # actions is an array in [0, 20] with an element per bracket. e.g. action == 3: 15% (3 * 5%)
            new_tax_rates = jnp.array(actions, dtype=jnp.float32) * 0.05

            # Update the state along with the start_year_inventory_coin
            state = state.replace(
                inventory_coin=inventory_coin,
                tax_rates=new_tax_rates,
                net_tax_payed_prev_period=net_tax_payed
            )

            return state

        is_tax_day = state.timestep % self.tax_period_length == 0 if self.enable_government else False
        income_prev_period_pre_tax = jax.lax.select(
            is_tax_day,
            state.inventory_coin + state.escrow_coin - state.start_year_inventory_coin,
            state.income_prev_period_pre_tax
        )
        start_year_inventory_coin = jax.lax.select(
            is_tax_day,
            state.inventory_coin + state.escrow_coin,
            state.start_year_inventory_coin
        )
        income_this_period_pre_tax = state.inventory_coin + state.escrow_coin - start_year_inventory_coin
        
        state = jax.lax.cond(
            is_tax_day,
            process_tax_day,
            lambda state: state,
            state
        )

        # Marginal incomes
        highest_bracket_per_agent = jnp.digitize(income_this_period_pre_tax, self.tax_bracket_cutoffs) - 1
        applicable_tax_rates = state.tax_rates[highest_bracket_per_agent]
        return state.replace(
            marginal_income=1 - applicable_tax_rates,
            income_this_period_pre_tax=income_this_period_pre_tax,
            income_prev_period_pre_tax=income_prev_period_pre_tax,
            start_year_inventory_coin=start_year_inventory_coin
        )
    
    @eqx.filter_jit
    def obs_dict_to_neural_input(self, obs_dict: dict):
        return jnp.array(jax.tree.leaves(obs_dict))
    
    def observation_space(self, agent: str):
        obs, _ = self.reset(jax.random.PRNGKey(0))
        if agent == "population":
            obs = obs["population"]["observation"][0]
            return spaces.Box(
                low=jnp.full(obs.shape[-1], 0),
                high=jnp.full(obs.shape[-1], 1000),
                shape=(obs.shape[-1],),
                dtype=jnp.float32
            )
        elif agent == "government":
            obs = obs["government"]["observation"]
            return spaces.Box(
                low=jnp.full(obs.shape[0], 0),
                high=jnp.full(obs.shape[0], 1000),
                shape=obs.shape[0],
                dtype=jnp.float32
            )
        else:
            raise ValueError(f"Unknown agent: {agent}")
        
    def action_space(self, agent: str):
        if agent == "population":
            num_actions = 1 if self.allow_noop else 0 # do nothing
            num_actions += self.num_resources # gather
            num_actions += 1 # craft
            num_actions += self.trade_actions_total # trade (buy and sell)
            # for convenience, in a discrete action space, we assume the trade actions
            # to be the first (2 * self.trade_actions_total) actions
            return spaces.Discrete(num_actions)
        elif agent == "government":
            num_actions_per_bracket = 21 # every 5% up to 100% (incl. 0%)
            num_brackets = len(self.tax_bracket_cutoffs) - 1
            actions = np.full(num_brackets, num_actions_per_bracket)
            return MultiDiscrete(actions)
        else:
            raise ValueError(f"Unknown agent: {agent}")
        
# -- END econonmy.py --- 

def initialize_environment(args, env_parameters):
    craft_skills, gather_skills = None, None
    if args.init_learned_skills:
        craft_skills, gather_skills = get_pareto_skill_dists(args.population_seed, args.num_agents, args.num_resources)

    env = EconomyEnv(
        seed=args.population_seed,
        num_population=args.num_agents,
        num_resources=args.num_resources,
        init_craft_skills=craft_skills,
        init_gather_skills=gather_skills,
        enable_government=args.enable_government,
        possible_trade_prices=args.trade_prices,
        base_skill_development_multiplier=args.skill_multiplier,
        **env_parameters
    )
    print("skills\n", jnp.concatenate([env.init_craft_skills[:, None], env.init_gather_skills], axis=1))
    return env

def initialize_policy(args,env):
    config = PpoTrainerParams(
        total_timesteps=args.total_timesteps,
        num_envs=args.num_envs,
        debug=args.debug,
        trainer_seed=args.seed,
        share_policy_nets=not args.individual_policies,
        share_value_nets=not args.individual_value_nets,
        num_log_episodes_after_training=args.eval_runs,
        network_size_pop_policy=args.network_size_pop_policy,
        network_size_pop_value=args.network_size_pop_value,
        network_size_gov=args.network_size_gov,
        num_steps=args.rollout_length,
        out_dir=args.out_dir
    )
    merged_config = {**config.__dict__, **env.__dict__}
    merged_config = jax.tree.map(lambda x: x.tolist() if isinstance(x, np.ndarray) else x, merged_config)
    _, merged_config = eqx.partition(merged_config, eqx.is_array)

    train_func, eval_func = build_ppo_trainer(env, config, args.load_model)
    train_func_jit = eqx.filter_jit(train_func, backend=config.backend)
    return train_func_jit, eval_func, merged_config, config

        
### merge with old training loop above        
def main():
    argument_parser = argparse.ArgumentParser(description="Run experiment")
    argument_parser.add_argument("--out_dir", type=str, default="run_0", help="Output directory")
    
    argument_parser.add_argument("-w", "--wandb", action="store_true")
    argument_parser.add_argument("-s", "--seed", type=int, default=42)
    argument_parser.add_argument("-ps", "--population_seed", type=int, default=42)
    argument_parser.add_argument("-a", "--num_agents", type=int, default=32)
    argument_parser.add_argument("-e", "--num_envs", type=int, default=6)
    argument_parser.add_argument("-t", "--total_timesteps", type=int, default=20000)
    argument_parser.add_argument("-r", "--num_resources", type=int, default=2)
    argument_parser.add_argument("-d", "--debug", action="store_true")
    argument_parser.add_argument("-l", "--load_model", type=str, default=None)
    argument_parser.add_argument("-i", "--individual_policies", action="store_true")
    argument_parser.add_argument("-iv", "--individual_value_nets", action="store_true")
    argument_parser.add_argument("-g", "--enable_government", action="store_true")
    argument_parser.add_argument("-wg", "--wandb_group", type=str, default=None)
    argument_parser.add_argument("-npp", "--network_size_pop_policy", nargs="+", type=int, default=[128, 128])
    argument_parser.add_argument("-npv", "--network_size_pop_value", nargs="+", type=int, default=[128, 128])
    argument_parser.add_argument("-ng", "--network_size_gov", nargs="+", type=int, default=[128, 128])
    argument_parser.add_argument("--trade_prices", nargs="+", type=int, default=np.arange(1,11,step=2, dtype=int))
    argument_parser.add_argument("--eval_runs", type=int, default=3)
    argument_parser.add_argument("--rollout_length", type=int, default=150)
    argument_parser.add_argument("--init_learned_skills", action="store_true")
    argument_parser.add_argument("--skill_multiplier", type=float, default=0.0)
    args, extra_args = argument_parser.parse_known_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)
    
    env_parameters = {}
    for i in range(0, len(extra_args), 2):
        key = extra_args[i].lstrip("--")
        value = extra_args[i + 1]
        if value.lower() == "false":
            env_parameters[key] = False
        elif value.lower() == "true":
            env_parameters[key] = True
        elif "." in value:
            env_parameters[key] = float(value)
        else:
            env_parameters[key] = int(value)
            
    if args.wandb:
        wandb.init(
            project=WANDB_PROJECT_NAME,
            config=merged_config,
            group=args.wandb_group,
            dir=args.out_dir,
            name="experiment_logs"
        )     
            
    #initialize the environment
    env = initialize_environment(args, env_parameters)
    
    #initialize the policy (PPO in this case)
    train_func_jit, eval_func, merged_config, config = initialize_policy(args, env)
    
  
    print("Starting training...")
    start_time = time.time()
    
    result = train_func_jit()
    train_state = result["train_state"]
    metrics = result["train_metrics"]

    print("Training complete! Saving model...")
    os.makedirs(args.out_dir, exist_ok=True)
    model_path = os.path.join(args.out_dir, f"ppo_{int(time.time())}.eqx")
    eqx.tree_serialise_leaves(model_path, train_state)

    print(f"Model saved to {model_path}")
        
    print("Evaluating trained model...")
    if config.num_log_episodes_after_training > 0:
        rng = jax.random.PRNGKey(args.seed)
        rng, eval_key = jax.random.split(rng)
        eval_keys = jax.random.split(
            eval_key, config.num_log_episodes_after_training
        )
        # eval_rewards, eval_logs = jax.vmap(eval_func, in_axes=(0, None))(
        #     eval_keys, trained_agent
        # )
        # log_eval_logs_to_wandb(eval_logs, args, WANDB_PROJECT_NAME, merged_config, env)
        for i in range(len(eval_keys)):
            eval_rewards, eval_logs = eval_func(eval_key, train_state)
            eval_logs = jax.tree.map(lambda x: np.expand_dims(x, axis=0), eval_logs)
            #print(eval_logs.keys())
            if args.wandb:
                log_eval_logs_to_wandb(eval_logs, args, WANDB_PROJECT_NAME, merged_config, env, id=i)
            else:
                log_eval_logs_local(eval_logs, args, merged_config, env, id=i)
    print("Evaluation complete.")
    
    if args.wandb:
        wandb.finish()
    
    
    final_info = {
        "final_train_state": model_path,
        "total_train_time": time.time() - start_time,
        "means": metrics[-1] if metrics else {}
    }
    final_info = merge_summary(args.out_dir, final_info)
    
    run_id = f"{args.out_dir}"
    final_info = {run_id: final_info}
    final_info_path = os.path.join(args.out_dir, "final_info.json")
    with open(final_info_path, "w") as f:
        json.dump(final_info, f)
        
        
    print(f"Final info saved to {final_info_path}")
        

if __name__ == "__main__":
    main()

    

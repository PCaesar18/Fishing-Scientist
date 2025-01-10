from econojax.environment.economy import EconomyEnv
from econojax.util.logging import log_eval_logs_to_wandb
from econojax.util.util_functions import get_pareto_skill_dists
from econojax.util.callbacks import logwrapper_callback, wandb_callback
from econojax.environment.jax_base_env_and_wrappers import LogWrapper
import jax
import chex
import jax.numpy as jnp
import equinox as eqx
import argparse
import wandb
import numpy as np
import os
import time
import optax
from typing import List
from functools import partial
from typing import NamedTuple, Union
import wandb
import distrax
from typing import Dict
from jax_tqdm import scan_tqdm

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
    backend: str = "gpu"  # or "gpu"
    num_log_episodes_after_training: int = 1
    debug: bool = True

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
            runner_state, trajectory_batch = jax.lax.scan(
                _env_step, runner_state, None, config.num_steps
            )

            # calculate gae
            train_state, env_state, last_obs, rng = runner_state
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
            metric = trajectory_batch.info
            metric["loss_info"] = loss_info
            metric["num_envs"] = config.num_envs
            
            if wandb.run:
                jax.debug.callback(wandb_callback, metric)
            if config.debug:
                jax.debug.callback(logwrapper_callback, metric)
            else:
                metric = None # save memory

            rng = update_state[-1]
            runner_state = (train_state, env_state, last_obs, rng)
            return runner_state, metric

        rng, key = jax.random.split(rng)
        runner_state = (train_state, env_state_v, obs_v, key)
        if not config.debug:
            train_step = scan_tqdm(config.num_iterations)(train_step)
        runner_state, metrics = jax.lax.scan(
            train_step, runner_state, jnp.arange(config.num_iterations)
        )
        trained_train_state = runner_state[0]
        rng = runner_state[-1]

        return {
            "train_state": trained_train_state,
            "train_metrics": metrics,
        }

    return train_func, eval_func

if __name__ == "__main__":
    env_params = {
        "num_agents": 12,
    }
    trainer_params = {
        "debug": False,
    }
    env = EconomyEnv(env_params)
    train_func = build_ppo_trainer(env, trainer_params)
    train_func()

# --- END ppo_trainer.py ---

def initalize_environment(args, env_parameters):
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
        num_steps=args.rollout_length
    )
    merged_config = {**config.__dict__, **env.__dict__}
    merged_config = jax.tree.map(lambda x: x.tolist() if isinstance(x, np.ndarray) else x, merged_config)
    _, merged_config = eqx.partition(merged_config, eqx.is_array)

    train_func, eval_func = build_ppo_trainer(env, config, args.load_model)
    train_func_jit = eqx.filter_jit(train_func, backend=config.backend)
    return train_func_jit, eval_func, merged_config

def train(args, train_func_jit, eval_func, env, out_dir):
    print("Starting training...")
    start_time = time.time()
    result = train_func_jit()
    train_state = result["train_state"]
    metrics = result["train_metrics"]

    print("Training complete! Saving model...")
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, f"ppo_{int(time.time())}.eqx")
    eqx.tree_serialise_leaves(model_path, train_state)

    print(f"Model saved to {model_path}")

    final_info = {
        "final_train_state": model_path,
        "total_train_time": time.time() - start_time,
        "train_metrics": metrics
    }

    final_info_path = os.path.join(out_dir, "final_info.json")
    with open(final_info_path, "w") as f:
        json.dump(final_info, f)

    print("Evaluating trained model...")
    rng = jax.random.PRNGKey(args.seed)
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
            eval_rewards, eval_logs = eval_func(eval_key, trained_agent)
            eval_logs = jax.tree.map(lambda x: np.expand_dims(x, axis=0), eval_logs)
            log_eval_logs_to_wandb(eval_logs, args, WANDB_PROJECT_NAME, merged_config, env, id=i)
    print("Evaluation complete.")

    print(f"Final info saved to {final_info_path}")


        
### merge with old training loop above        
def main():
    argument_parser = argparse.ArgumentParser(description="Run experiment")
    argument_parser.add_argument("--out_dir", type=str, default="run_0", help="Output directory")
    
    argument_parser.add_argument("-w", "--wandb", action="store_true")
    argument_parser.add_argument("-s", "--seed", type=int, default=42)
    argument_parser.add_argument("-ps", "--population_seed", type=int, default=42)
    argument_parser.add_argument("-a", "--num_agents", type=int, default=32)
    argument_parser.add_argument("-e", "--num_envs", type=int, default=6)
    argument_parser.add_argument("-t", "--total_timesteps", type=int, default=5e6)
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
    
    wandb.require("core")
    if args.wandb:
    wandb.init(
        project="EconoJax"
        config=merged_config,
        group=args.wandb_group
    )
    
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
            
    #initialize the environment
    env = initialize_environment(args, env_parameters)
    
    #initialize the policy (PPO in this case)
    train_func_jit, eval_func, merged_config = initialize_policy(args, env)
    
    all_results = {}
    final_infos = {}
    
    #train model
    train(args, train_func_jit, eval_func, env, args.out_dir)

    
    
    with open(os.path.join(out_dir, "final_info.json"), "w") as f:
        json.dump(final_infos, f)

    with open(os.path.join(out_dir, "all_results.npy"), "wb") as f:
        np.save(f, all_results)
        

if __name__ == "__main__":
    main()

    

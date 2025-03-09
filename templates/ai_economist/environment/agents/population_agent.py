# Concrete implementation for population agents.
import jax
import jax.numpy as jnp
from base_agent import BaseAgent
from environment.economy import EconomyEnv, EnvState
from gymnax.environments import spaces


class PopulationAgent(BaseAgent):
    name: str = "population"

    def observation_space(self, env: EconomyEnv) -> spaces.Box:
        # Use the reset method to obtain a sample observation.
        obs, _ = env.reset(jax.random.PRNGKey(0))
        pop_obs = obs["population"]["observation"][0]
        return spaces.Box(
            low=jnp.full(pop_obs.shape, 0.0),
            high=jnp.full(pop_obs.shape, 1000.0),
            shape=pop_obs.shape,
            dtype=jnp.float32,
        )

    def action_space(self, env: EconomyEnv) -> spaces.Discrete:
        # For example, using your current logic:
        num_actions = (1 if env.allow_noop else 0) + env.num_resources + 1 + env.trade_actions_total
        return spaces.Discrete(num_actions)

    def get_observations(self, state: EnvState, env: EconomyEnv) -> jnp.ndarray:
        # Example: combine a few scaled state components.
        is_tax_day = jnp.where(
            env.enable_government, state.timestep % env.tax_period_length == 0, 0
        ).astype(jnp.float32)
        # Here we create a simple private observation by stacking some quantities.
        private_obs = jnp.column_stack([
            state.inventory_coin / 1000.0,
            jnp.mean(state.inventory_resources, axis=1, keepdims=True),
            state.inventory_labor / 100.0,
        ])
        # Global info could be appended; for simplicity, we add a single flag.
        global_obs = jnp.broadcast_to(is_tax_day, (private_obs.shape[0], 1))
        return jnp.hstack([private_obs, global_obs])

    def get_action_masks(self, state: EnvState, env: EconomyEnv) -> jnp.ndarray:
        # For demonstration, assume all actions are available for each population agent.
        n = self.action_space(env).n
        return jnp.ones((env.num_population, n), dtype=jnp.bool_)

    def compute_reward(
        self, old_state: EnvState, new_state: EnvState, env: EconomyEnv
    ) -> jnp.ndarray:
        # Compute reward as the change in utility for the population.
        return new_state.utility["population"] - old_state.utility["population"]
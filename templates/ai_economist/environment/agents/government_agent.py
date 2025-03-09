# Concrete implementation for the government agent.
import jax
import jax.numpy as jnp
from base_agent import BaseAgent
from environment.economy import EconomyEnv, EnvState
from gymnax.environments import spaces
class GovernmentAgent(BaseAgent):
    name: str = "government"

    def observation_space(self, env: EconomyEnv) -> spaces.Box:
        obs, _ = env.reset(jax.random.PRNGKey(0))
        gov_obs = obs["government"]["observation"]
        return spaces.Box(
            low=jnp.full(gov_obs.shape, 0.0),
            high=jnp.full(gov_obs.shape, 1000.0),
            shape=gov_obs.shape,
            dtype=jnp.float32,
        )

    def action_space(self, env: EconomyEnv) -> spaces.Space:
        # Government actions: using MultiDiscrete as in your current design.
        num_actions_per_bracket = 21  # e.g., each tax bracket has 21 options. # every 5% up to 100% (incl. 0%)
        num_brackets = len(env.tax_bracket_cutoffs) - 1
        from util.spaces import MultiDiscrete  # adjust the import as needed
        return MultiDiscrete(jnp.full((num_brackets,), num_actions_per_bracket, dtype=jnp.int32)) #double check here

    def get_observations(self, state: EnvState, env: EconomyEnv) -> jnp.ndarray:
        # For example, aggregate population statistics.
        avg_coin = jnp.mean(state.inventory_coin)
        std_coin = jnp.std(state.inventory_coin)
        # You can combine multiple pieces of information.
        return jnp.array(
            [state.timestep % env.tax_period_length, avg_coin / 1000.0, std_coin / 1000.0],
            dtype=jnp.float32,
        )

    def get_action_masks(self, state: EnvState, env: EconomyEnv) -> jnp.ndarray:
        gov_space = self.action_space(env)
        # Assume all government actions are available.
        return jnp.ones((gov_space.nvec.shape[0], gov_space.nvec[0]), dtype=jnp.bool_)

    def compute_reward(
        self, old_state: EnvState, new_state: EnvState, env: EconomyEnv
    ) -> jnp.ndarray:
        return new_state.utility["government"] - old_state.utility["government"]
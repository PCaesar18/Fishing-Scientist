import jax
import jax.numpy as jnp
import equinox as eqx
from gymnax.environments import spaces
from environment.economy import EconomyEnv, EnvState

# Base agent class: defines the API that all agents must implement.
class BaseAgent(eqx.Module):
    name: str  # e.g., "population" or "government"

    def observation_space(self, env: EconomyEnv) -> spaces.Space:
        """Return the observation space for the agent given the environment."""
        raise NotImplementedError

    def action_space(self, env: EconomyEnv) -> spaces.Space:
        """Return the action space for the agent given the environment."""
        raise NotImplementedError

    def get_observations(self, state: EnvState, env: EconomyEnv) -> jnp.ndarray:
        """
        Extract the agent-specific observation from the overall environment state.
        """
        raise NotImplementedError

    def get_action_masks(self, state: EnvState, env: EconomyEnv) -> jnp.ndarray:
        """
        Return an action mask for the agent given the environment state.
        """
        raise NotImplementedError

    def compute_reward(
        self, old_state: EnvState, new_state: EnvState, env: EconomyEnv
    ) -> jnp.ndarray:
        """
        Compute and return the reward for the agent given a state transition.
        """
        raise NotImplementedError

# Concrete implementation for population agents.
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


# Concrete implementation for the government agent.
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
        num_actions_per_bracket = 21  # e.g., each tax bracket has 21 options.
        num_brackets = len(env.tax_bracket_cutoffs) - 1
        from util.spaces import MultiDiscrete  # adjust the import as needed
        return MultiDiscrete(jnp.full((num_brackets,), num_actions_per_bracket, dtype=jnp.int32))

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

import wandb
import jax.numpy as jnp
import jax
import numpy as np
import json
import os
import time
import pickle
import matplotlib.pyplot as plt
        
        
def log_eval_logs_to_wandb(log, args, wandb_project_name, config, env, id=None, aggregator=None):
    if args.wandb_group is None:
        group_name = f"eval_logs_{int(time.time())}"
    else:
        group_name = args.wandb_group
    num_envs = log["timestep"].shape[0]
    num_steps_per_episode = log["timestep"].shape[1]
    for env_id in range(num_envs):
        wandb_dir = os.path.join(args.out_dir, "wandb_eval_logs")
        os.makedirs(wandb_dir, exist_ok=True)
        run = wandb.init(
            project=wandb_project_name,
            config=config,
            tags=["eval"],
            group=group_name,
            dir=wandb_dir,
            name="evaluation_logs" 
        )
        for step in range(0, num_steps_per_episode, 25):
            log_step = jax.tree.map(lambda x: x[env_id, step], log)
            run.log(log_step)

        # ACTION DISTRIBUTION
        bins = np.concatenate([
            np.arange(0, env.trade_actions_total, len(env.possible_trade_prices), dtype=int),
            np.arange(env.trade_actions_total, env.action_space("population").n + 1, dtype=int),
        ], dtype=int)

        resources = config["num_resources"]
        shared_policy = config["share_policy_nets"]
        shared_values = config["share_value_nets"]
        agent_ids = config["insert_agent_ids"]
        seed = config["trainer_seed"]
        pop_seed = config["seed"]
        wandb_id = run.id
        folder = "population_actions"
        if not os.path.exists(folder):
            os.makedirs(folder)

        # save the action distribution dict as a file
        name = f"{folder}/_r-{resources}_s-{shared_policy}_sv-{shared_values}_ai-{agent_ids}_seed-{seed}_popseed-{pop_seed}-{id}"
        with open(f"{name}_{wandb_id}.pkl", "wb") as f:
            pickle.dump(log["population_actions"], f)

        agent_action_dists = []
        for agent_id, agent_actions in log["population_actions"].items():
            counts, _ = np.histogram(agent_actions[env_id], bins=bins)
            agent_action_dists.append(counts)
        agent_action_dists = np.stack(agent_action_dists, axis=0)
        pairwise_diffs = np.abs(agent_action_dists[:, None] - agent_action_dists)
        total_diffs = pairwise_diffs.sum() // 2
        run.summary[f"Total action dist differences"] = total_diffs

        # Send action distribution as image to wandb:
        for agent_id, agent_actions in log["population_actions"].items():
            counts, _ = np.histogram(agent_actions[env_id], bins=bins)

            labels = [label for pair in zip([f"buy_{i}" for i in range(env.num_resources)], [f"sell_{i}" for i in range(env.num_resources)]) for label in pair] + [f"gather_{i}" for i in range(env.num_resources)] + ["craft"]
            if len(counts) > len(labels):
                labels.append("noop")
            action_dist = {
                label: count for label, count in zip(labels, counts)
            }
            total = sum(action_dist.values())
            action_dist = {k: v / total for k, v in action_dist.items()}
            fig = plt.figure()
            plt.bar(list(action_dist.keys()), action_dist.values())
            plt.ylabel("Percentage of actions")
            plt.ylim(0, 1)
            run.log({f"Action dist agent {agent_id}": wandb.Image(fig)}, commit=agent_id == len(log["population_actions"]) - 1)
            plt.close()

        run.finish()



def wandb_callback(info):
    if wandb.run is None:
        raise wandb.Error(
            """
                wandb logging is enabled, but wandb.run is not defined.
                Please initialize wandb before using this callback.
            """
        )
    num_envs = info["num_envs"]
    return_values = info["returned_episode_returns"][info["returned_episode"]]
    if len(return_values) == 0:
        return # no episodes finished
    equalities = info["equality"][info["returned_episode"]]
    productivities = info["productivity"][info["returned_episode"]]
    coins = info["coin"][info["returned_episode"]]
    labors = info["labor"][info["returned_episode"]]
    timesteps = info["timestep"][info["returned_episode"]] * num_envs

    try:
        losses = info["loss_info"]
        total_loss, actor_loss, value_loss, entropy = jax.tree.map(jnp.mean, losses)
    except KeyError:
        total_loss, actor_loss, value_loss, entropy = None, None, None, None
    episode_returns_averaged = np.mean(np.array(return_values), axis=0)
    coins_averaged = np.mean(np.array(coins), axis=0)
    labors_averaged = np.mean(np.array(labors), axis=0)
    wandb.log(
        {
            "per_agent_episode_return": {
                f"{agent_id}": episode_returns_averaged[agent_id]
                for agent_id in range(len(episode_returns_averaged)-1)
            },
            "mean_population_episode_return": np.mean(episode_returns_averaged),
            "government_reward": episode_returns_averaged[-1],
            "total_episode_return_sum": np.sum(episode_returns_averaged),
            "total_loss": total_loss,
            "actor_loss": actor_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "training timestep": timesteps[-1],
            "productivity": productivities.mean(),
            "equality": equalities.mean(),
            "per_agent_coin": {
                f"{agent_id}": coins_averaged[agent_id]
                for agent_id in range(len(coins_averaged))
            },
            "per_agent_labor": {
                f"{agent_id}": labors_averaged[agent_id]
                for agent_id in range(len(labors_averaged))
            },
            "mean_population_coin": np.mean(coins_averaged),
            "mean_population_labor": np.mean(labors_averaged),
            "median_population_coin": np.median(coins_averaged),
            "median_population_labor": np.median(labors_averaged),
            "median_population_episode_return": np.median(episode_returns_averaged),
        }
    )
        
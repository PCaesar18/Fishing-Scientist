import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import jax
import pickle
import os
import json

def logwrapper_callback(info, out_dir, aggregator=None):
    os.makedirs(out_dir, exist_ok=True)
    
    info = jax.tree_map(lambda x: np.asarray(x) if isinstance(x, jax.Array) else x, info)
    
    num_envs = info["num_envs"]
    returned_episode = info["returned_episode"]
    
    return_values = info["returned_episode_returns"][returned_episode]
    if return_values.size == 0:
        return  

    timesteps = info["timestep"][returned_episode] * num_envs
    equalities = np.asarray(info["equality"][returned_episode])
    productivities = np.asarray(info["productivity"][returned_episode])
    coins = np.asarray(info["coin"][returned_episode])
    labors = np.asarray(info["labor"][returned_episode])

    try:
        losses = info["loss_info"]
        total_loss = np.asarray(losses[0]) if losses[0].size > 0 else None
        actor_loss = np.asarray(losses[1]) if losses[1].size > 0 else None
        value_loss = np.asarray(losses[2]) if losses[2].size > 0 else None
        entropy = np.asarray(losses[3]) if losses[3].size > 0 else None
    except KeyError:
        total_loss = actor_loss = value_loss = entropy = None

    with open(os.path.join(out_dir,"per_step_logs.jsonl"), "a") as f:
        for step_idx in range(len(timesteps)):
            step_data = {
                "global_step": int(timesteps[step_idx]),
                "returns": return_values[step_idx].tolist(),
                "equality": float(equalities[step_idx]),
                "productivity": float(productivities[step_idx]),
                "coins": coins[step_idx].tolist(),
                "labors": labors[step_idx].tolist(),
            }

            if total_loss is not None:
                step_data.update({
                    "total_loss": float(total_loss[step_idx].mean()), #mean over the number of environments
                    "actor_loss": float(actor_loss[step_idx].mean()),
                    "value_loss": float(value_loss[step_idx].mean()),
                    "entropy": float(entropy[step_idx].mean()),
                })

            f.write(json.dumps(step_data) + "\n")

def log_eval_logs_local(log, args, config, env, id=None, aggregator=None):
    num_envs = log["timestep"].shape[0]
    num_steps_per_episode = log["timestep"].shape[1]
    eval_dir = os.path.join(args.out_dir, "eval_logs")
    os.makedirs(eval_dir, exist_ok=True)
    log_file_path = os.path.join(eval_dir, f"evaluation_logs_{id}.pkl")
    
    all_logs = []
    for env_id in range(num_envs):
        for step in range(0, num_steps_per_episode, 25):
            log_step = jax.tree.map(lambda x: x[env_id, step], log)
            all_logs.append(log_step)
    
    with open(log_file_path, "wb") as log_file:
        pickle.dump(all_logs, log_file)

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
        folder = os.path.join(eval_dir, "population_actions")
        if not os.path.exists(folder):
            os.makedirs(folder)
       #now stored in eval logs 
        name = f"{folder}/_r-{resources}_s-{shared_policy}_sv-{shared_values}_ai-{agent_ids}_seed-{seed}_popseed-{pop_seed}-{id}"
        with open(f"{name}_{id}.pkl", "wb") as f:
            pickle.dump(log["population_actions"], f)

        agent_action_dists = []
        for agent_id, agent_actions in log["population_actions"].items():
            counts, _ = np.histogram(agent_actions[env_id], bins=bins)
            agent_action_dists.append(counts)
        agent_action_dists = np.stack(agent_action_dists, axis=0)
        pairwise_diffs = np.abs(agent_action_dists[:, None] - agent_action_dists)
        total_diffs = pairwise_diffs.sum() // 2

        # Log total action distribution differences locally
        total_diffs_path = os.path.join(eval_dir, f"total_action_dist_differences_{id}.pkl")
        with open(total_diffs_path, "wb") as f:
            pickle.dump({"Total action dist differences": total_diffs}, f)

        # Save action distribution as image locally
        media_dir = os.path.join(eval_dir, "media")
        os.makedirs(media_dir, exist_ok=True)
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
            image_path = os.path.join(media_dir, f"action_dist_agent_{agent_id}_{id}.png")
            plt.savefig(image_path)
            plt.close()
        
    if aggregator is not None:
        #print("we get into the aggregator")
        aggregator["eval_logs"] = aggregator.get("eval_logs", [])
        aggregator["eval_logs"].append(log)
        
        
def log_callback(info, out_dir, aggregator=None):
    # print("DEBUG: log_callback was called!")
    num_envs = info["num_envs"]
    return_values = info["returned_episode_returns"][info["returned_episode"]]
    if len(return_values) == 0:
        return # no episodes finished
    
    equalities = info["equality"][info["returned_episode"]]
    productivities = info["productivity"][info["returned_episode"]]
    coins = info["coin"][info["returned_episode"]]
    labors = info["labor"][info["returned_episode"]]
    timesteps = info["timestep"][info["returned_episode"]] * num_envs
    #print("timesteps:", timesteps)

    try:
        losses = info["loss_info"]
        total_loss, actor_loss, value_loss, entropy = jax.tree.map(jnp.mean, losses)
    except KeyError:
        total_loss, actor_loss, value_loss, entropy = None, None, None, None
    episode_returns_averaged = np.mean(np.array(return_values), axis=0)
    coins_averaged = np.mean(np.array(coins), axis=0)
    labors_averaged = np.mean(np.array(labors), axis=0)
    summarized_log_data = {
        "per_agent_episode_return": {
            f"{agent_id}": float(episode_returns_averaged[agent_id])
            for agent_id in range(len(episode_returns_averaged) - 1)
        },
        "mean_population_episode_return": float(np.mean(episode_returns_averaged)),
        "government_reward": float(episode_returns_averaged[-1]),
        "total_episode_return_sum": float(np.sum(episode_returns_averaged)),
        "total_loss": float(total_loss) if total_loss else None,
        "actor_loss": float(actor_loss) if actor_loss else None,
        "value_loss": float(value_loss) if value_loss else None,
        "entropy": float(entropy) if entropy else None,
        "training timestep": int(timesteps[-1]),
        "productivity": float(productivities.mean()),
        "equality": float(equalities.mean()),
        "per_agent_coin": {
            f"{agent_id}": float(coins_averaged[agent_id])
            for agent_id in range(len(coins_averaged))
        },
        "per_agent_labor": {
            f"{agent_id}": float(labors_averaged[agent_id])
            for agent_id in range(len(labors_averaged))
        },
        "mean_population_coin": float(np.mean(coins_averaged)),
        "mean_population_labor": float(np.mean(labors_averaged)),
        "median_population_coin": float(np.median(coins_averaged)),
        "median_population_labor": float(np.median(labors_averaged)),
        "median_population_episode_return": float(np.median(episode_returns_averaged)),
    }
    

    # for step in range(len(timesteps)):
    #     equality = float(info["equality"][info["returned_episode"]][step])
    #     productivity = float(info["productivity"][info["returned_episode"]][step])
    #     total_loss = losses[0][step] if losses else None
    #     actor_loss = losses[1][step] if losses else None
    #     value_loss = losses[2][step] if losses else None
    #     entropy = losses[3][step] if losses else None

    #     print("lets see if we can save these in the numpy file",
    #           "equality:", equality, '\n',
    #           "productivity:", productivity, '\n',
    #           "total_loss:", total_loss, '\n',
    #           "actor_loss:", actor_loss, '\n',
    #           "value_loss:", value_loss, '\n',
    #           "entropy:", entropy, '\n')
    
    local_summary_file = os.path.join(out_dir,"run-summary.json")
    with open(local_summary_file, "a") as f:
        f.write(json.dumps(summarized_log_data) + "\n")
    

    np.save(os.path.join(out_dir,"all_results.npy"), summarized_log_data)


#json error here, lines in jsonl
def merge_summary(out_dir, final_info):
    summary_path = os.path.join(out_dir, "training_logs/run-summary.json")

    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r") as f:
                summaries = []
                for line in f:
                    line = line.strip()
                    if line:
                        summaries.append(json.loads(line))
                for summary in summaries:
                    final_info["means"].update(summary)
                #print("Successfully merged run-summary.json into 'means'.")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON in line: {e}")
    else:
        print("run-summary.json not found. Skipping merge.")
    
    return final_info
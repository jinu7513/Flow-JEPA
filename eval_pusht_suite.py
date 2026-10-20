"""Paired PushT deployment evaluation: default, red T, salt-and-pepper.

All conditions use native rendered current AND goal images. This suite does not
copy cached dataset pixels: a red goal is rendered with the same block color as
the red current observation. Dynamics, state targets and success rule are shared.
"""
import csv
import hashlib
import json
import time
from pathlib import Path

import hydra
import numpy as np
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from sklearn.preprocessing import StandardScaler

from eval import get_dataset, img_transform
from perturbations.salt_pepper import SaltPepperObservation


CONDITIONS = ("default", "red_t", "salt_pepper")


def select_trials(dataset, num_eval, goal_offset, seed):
    """Choose dataset rows once, including every valid row as a candidate."""
    column = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episodes = dataset.get_col_data(column)
    steps = dataset.get_col_data("step_idx")
    unique, inverse = np.unique(episodes, return_inverse=True)
    maximum_step = np.full(len(unique), -1, dtype=np.int64)
    np.maximum.at(maximum_step, inverse, steps)
    valid = np.flatnonzero(steps + goal_offset <= maximum_step[inverse])
    if num_eval < 1 or goal_offset < 1 or len(valid) < num_eval:
        raise ValueError("Not enough valid start/goal pairs for the requested evaluation")
    selected = np.sort(np.random.default_rng(seed).choice(valid, num_eval, replace=False))
    goal_rows = []
    for row in selected:
        matches = np.flatnonzero((episodes == episodes[row]) & (steps == steps[row] + goal_offset))
        if len(matches) != 1:
            raise ValueError("Dataset must contain exactly one row for each selected goal step")
        goal_rows.append(int(matches[0]))
    initial = dataset.get_row_data(selected)
    goals = dataset.get_row_data(np.asarray(goal_rows))
    seeds = np.asarray(initial.get("seed", np.arange(num_eval) + seed)).reshape(num_eval)
    return {
        "rows": selected, "goal_rows": np.asarray(goal_rows),
        "episodes": episodes[selected], "start_steps": steps[selected],
        "states": np.asarray(initial["state"]), "goal_states": np.asarray(goals["state"]),
        "seeds": seeds.astype(np.int64),
    }


def reset_trial_world(world, trials):
    """Set goal overlay and render both views at dataset-defined physical states."""
    for index, wrapper in enumerate(world.envs.envs):
        variations = wrapper.unwrapped.variation_space
        goal = trials["goal_states"][index]
        variations["goal"]["position"].set_init_value(np.asarray(goal[2:4], dtype=np.float64))
        variations["goal"]["angle"].set_init_value(float(goal[4]))
    options = [
        {"state": state.copy(), "goal_state": goal.copy()}
        for state, goal in zip(trials["states"], trials["goal_states"])
    ]
    world.reset(seed=trials["seeds"].tolist(), options=options)


def evaluate_condition(cfg, condition, model, process, trials, directory):
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    model.set_flow_seed(int(cfg.flow_seed))
    kwargs = OmegaConf.to_container(cfg.world, resolve=False)
    kwargs["num_envs"] = int(cfg.eval.num_eval)
    kwargs["max_episode_steps"] = 2 * int(cfg.eval.eval_budget)
    if condition == "red_t":
        kwargs["init_value"] = {"block.color": np.asarray(cfg.suite.red_t_rgb, dtype=np.uint8)}
    world = swm.World(**kwargs, image_shape=(int(cfg.eval.img_size),) * 2)
    try:
        solver = hydra.utils.instantiate(cfg.solver, model=model)
        policy = swm.policy.WorldModelPolicy(
            solver=solver, config=swm.PlanConfig(**cfg.plan_config), process=process,
            transform={"pixels": img_transform(cfg), "goal": img_transform(cfg)},
        )
        world.set_policy(policy)
        reset_trial_world(world, trials)
        np.save(directory / "initial_states.npy", world.infos["state"].copy())
        clean_goal = world.infos["goal"].copy()
        corruption = SaltPepperObservation(**OmegaConf.to_container(cfg.suite.salt_pepper, resolve=True)) if condition == "salt_pepper" else None
        alive = np.ones(world.num_envs, dtype=bool)
        successes = np.zeros(world.num_envs, dtype=bool)
        steps = np.zeros(world.num_envs, dtype=np.int64)
        current_frames = [[] for _ in range(world.num_envs)]
        goal_frames = [[] for _ in range(world.num_envs)]
        start = time.monotonic()
        for step in range(int(cfg.eval.eval_budget)):
            world.infos["goal"] = clean_goal.copy()
            info = corruption(world.infos) if corruption else world.infos
            if step == 0:
                Image.fromarray(info["pixels"][0, -1]).save(directory / "initial_observation.png")
                Image.fromarray(info["goal"][0, -1]).save(directory / "goal.png")
            if cfg.output.save_video:
                for index in range(world.num_envs):
                    current_frames[index].append(info["pixels"][index, -1].copy())
                    goal_frames[index].append(info["goal"][index, -1].copy())
            actions = world.policy.get_action(info)
            steps += alive
            _, world.rewards, world.terminateds, world.truncateds, world.infos = world.envs.step(
                actions, mask=None if alive.all() else alive,
            )
            successes |= alive & world.terminateds
            alive &= ~(world.terminateds | world.truncateds)
            if not alive.any():
                break
        elapsed = time.monotonic() - start
        world.infos["goal"] = clean_goal.copy()
        final_info = corruption(world.infos) if corruption else world.infos
        Image.fromarray(final_info["pixels"][0, -1]).save(directory / "final_observation.png")
        if cfg.output.save_video:
            for index in range(world.num_envs):
                current_frames[index].append(final_info["pixels"][index, -1].copy())
                goal_frames[index].append(final_info["goal"][index, -1].copy())
            from stable_worldmodel.plot import save_panel_videos
            save_panel_videos(directory / "videos", {"policy observation": current_frames, "policy goal": goal_frames}, fps=10)
        return {
            "condition": condition, "success_rate": float(100 * successes.mean()),
            "num_eval": world.num_envs, "successes": successes.tolist(),
            "steps": steps.tolist(), "evaluation_seconds": elapsed,
        }
    finally:
        world.close()


@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht_suite")
def run(cfg: DictConfig):
    if cfg.world.env_name != "swm/PushT-v1" or cfg.perturbation.enabled:
        raise ValueError("This suite supports PushT only; disable legacy Gaussian perturbation")
    conditions = list(cfg.suite.conditions)
    if not conditions or len(set(conditions)) != len(conditions) or any(c not in CONDITIONS for c in conditions):
        raise ValueError(f"Choose unique conditions from {CONDITIONS}")
    if int(cfg.plan_config.horizon) * int(cfg.plan_config.action_block) > int(cfg.eval.eval_budget):
        raise ValueError("Planning horizon exceeds the evaluation budget")
    checkpoint = Path(str(cfg.policy)).expanduser()
    if not checkpoint.is_absolute() or not checkpoint.is_file() or checkpoint.suffix != ".pt":
        raise ValueError("policy must be an absolute path to a trained .pt beside config.json")
    if not checkpoint.with_name("config.json").is_file():
        raise FileNotFoundError("Checkpoint requires an adjacent config.json")
    directory = Path(str(cfg.output.dir)).expanduser()
    if not directory.is_absolute():
        raise ValueError("output.dir must be an absolute new experiment directory")
    directory.mkdir(parents=True, exist_ok=False)
    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    trials = select_trials(dataset, int(cfg.eval.num_eval), int(cfg.eval.goal_offset_steps), int(cfg.seed))
    serial_trials = {key: value.tolist() for key, value in trials.items()}
    (directory / "trials.json").write_text(json.dumps(serial_trials, indent=2), encoding="utf-8")
    recorded_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    recorded_cfg.world.max_episode_steps = 2 * int(cfg.eval.eval_budget)
    OmegaConf.save(recorded_cfg, directory / "config.yaml", resolve=True)
    process = {}
    for column in cfg.dataset.keys_to_cache:
        if column == "pixels":
            continue
        values = dataset.get_col_data(column)
        process[column] = StandardScaler().fit(values[~np.isnan(values).any(axis=1)])
        if column != "action":
            process[f"goal_{column}"] = process[column]
    model = swm.wm.utils.load_pretrained(str(checkpoint)).to(cfg.solver.device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    digest = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    (directory / "checkpoint.json").write_text(json.dumps({"path": str(checkpoint), "sha256": digest.hexdigest()}, indent=2), encoding="utf-8")
    summaries = []
    for condition in conditions:
        condition_dir = directory / condition
        condition_dir.mkdir()
        print(f"Evaluating {condition} on {len(trials['rows'])} paired trials", flush=True)
        metrics = evaluate_condition(cfg, condition, model, process, trials, condition_dir)
        (condition_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        with (condition_dir / "episodes.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["row", "episode", "start_step", "goal_row", "success", "steps"])
            writer.writerows(zip(trials["rows"], trials["episodes"], trials["start_steps"],
                                 trials["goal_rows"], metrics["successes"], metrics["steps"]))
        summaries.append(metrics)
        with (directory / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["condition", "success_rate", "num_eval", "evaluation_seconds"])
            writer.writeheader()
            writer.writerows({key: item[key] for key in writer.fieldnames} for item in summaries)
        print(f"{condition}: {metrics['success_rate']:.1f}% success", flush=True)


if __name__ == "__main__":
    run()

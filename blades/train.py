#!/usr/bin/env python
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Optional

import ray
import typer
import yaml
from ray.rllib.common import CLIArguments as cli
from ray.rllib.common import SupportedFileType
from ray.rllib.common import download_example_file, get_file_type
from ray.tune.registry import register_trainable
from ray.tune.resources import resources_to_json, json_to_resources
from ray.tune.schedulers import create_scheduler
from ray.tune.tune import run_experiments


def _register_all():
    from blades.algorithms.registry import ALGORITHMS, _get_algorithm_class

    for key, get_trainable_class_and_config in ALGORITHMS.items():
        register_trainable(key, get_trainable_class_and_config()[0])
    for key in ["__fake", "__sigmoid_fake_data", "__parameter_tuning"]:
        register_trainable(key, _get_algorithm_class(key))


_register_all()
# Create the "train" Typer app
train_app = typer.Typer()


def _patch_path(path: str):
    """Patch a path to be relative to the current working directory.

    Args:
        path: relative input path.

    Returns: Patched path.
    """
    # This script runs in the fllib dir.
    fllib_dir = Path(__file__).parent
    if isinstance(path, list):
        return [_patch_path(i) for i in path]
    elif isinstance(path, dict):
        return {_patch_path(k): _patch_path(v) for k, v in path.items()}
    elif isinstance(path, str):
        if os.path.exists(path):
            return path
        else:
            abs_path = str(fllib_dir.absolute().joinpath(path))
            return abs_path if os.path.exists(abs_path) else path
    else:
        return path


def load_experiments_from_file(
    config_file: str,
    file_type: SupportedFileType,
    stop: Optional[str] = None,
    checkpoint_config: Optional[dict] = None,
) -> dict:
    """Load experiments from a file. Supports YAML and Python files.

    If you want to use a Python file, it has to have a 'config' variable
    that is an AlgorithmConfig object and - optionally - a `stop` variable defining
    the stop criteria.

    Args:
        config_file: The yaml or python file to be used as experiment definition.
            Must only contain exactly one experiment.
        file_type: One value of the `SupportedFileType` enum (yaml or python).
        stop: An optional stop json string, only used if file_type is python.
            If None (and file_type is python), will try to extract stop information
            from a defined `stop` variable in the python file, otherwise, will use {}.
        checkpoint_config: An optional checkpoint config to add to the returned
            experiments dict.

    Returns:
        The experiments dict ready to be passed into `tune.run_experiments()`.
    """

    # Yaml file.
    if file_type == SupportedFileType.yaml:
        with open(config_file) as f:
            experiments = yaml.safe_load(f)
            if stop is not None and stop != "{}":
                raise ValueError("`stop` criteria only supported for python files.")
    # Python file case (ensured by file type enum)
    else:
        module_name = os.path.basename(config_file).replace(".py", "")
        spec = importlib.util.spec_from_file_location(module_name, config_file)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        if not hasattr(module, "config"):
            raise ValueError(
                "Your Python file must contain a 'config' variable "
                "that is an AlgorithmConfig object."
            )
        algo_config = getattr(module, "config")
        if stop is None:
            stop = getattr(module, "stop", {})
        else:
            stop = json.loads(stop)

        # Note: we do this gymnastics to support the old format that
        # "run_fllib_experiments" expects. Ideally, we'd just build the config and
        # run the algo.
        config = algo_config.to_dict()
        experiments = {
            "default": {
                "run": algo_config.__class__.__name__.replace("Config", ""),
                "env": config.get("env"),
                "config": config,
                "stop": stop,
            }
        }

    for key, _ in experiments.items():
        experiments[key]["checkpoint_config"] = checkpoint_config or {}
    return experiments


@train_app.command()
def file(
    # File-based arguments.
    config_file: str = cli.ConfigFile,
    # stopping conditions
    stop: Optional[str] = cli.Stop,
    # Checkpointing
    checkpoint_freq: int = cli.CheckpointFreq,
    checkpoint_at_end: bool = cli.CheckpointAtEnd,
    keep_checkpoints_num: int = cli.KeepCheckpointsNum,
    checkpoint_score_attr: str = cli.CheckpointScoreAttr,
    # Additional config arguments used for overriding.
    v: bool = cli.V,
    vv: bool = cli.VV,
    trace: bool = cli.Trace,
    # WandB options.
    wandb_key: Optional[str] = cli.WandBKey,
    wandb_project: Optional[str] = cli.WandBProject,
    wandb_run_name: Optional[str] = cli.WandBRunName,
    # Ray cluster options.
    local_mode: bool = cli.LocalMode,
    ray_address: Optional[str] = cli.RayAddress,
    ray_ui: bool = cli.RayUi,
    ray_num_cpus: Optional[int] = cli.RayNumCpus,
    ray_num_gpus: Optional[int] = cli.RayNumGpus,
    ray_num_nodes: Optional[int] = cli.RayNumNodes,
    ray_object_store_memory: Optional[int] = cli.RayObjectStoreMemory,
    # Ray scheduling options.
    resume: bool = cli.Resume,
    scheduler: Optional[str] = cli.Scheduler,
    scheduler_config: str = cli.SchedulerConfig,
):
    """Train a reinforcement learning agent from file. The file argument is
    required to run this command.\n\n.

    Grid search example with the RLlib CLI:\n
      rllib train file tuned_examples/ppo/cartpole-ppo.yaml\n\n

    You can also run an example from a URL with the file content:\n
      rllib train file https://raw.githubusercontent.com/ray-project/ray/\
      master/rllib/tuned_examples/ppo/cartpole-ppo.yaml
    """
    # Attempt to download the file if it's not found locally.
    config_file, temp_file = download_example_file(
        example_file=config_file, base_url=None
    )

    checkpoint_config = {
        "checkpoint_frequency": checkpoint_freq,
        "checkpoint_at_end": checkpoint_at_end,
        "num_to_keep": keep_checkpoints_num,
        "checkpoint_score_attribute": checkpoint_score_attr,
    }

    file_type = get_file_type(config_file)

    experiments = load_experiments_from_file(
        config_file, file_type, stop, checkpoint_config
    )
    exp_name = list(experiments.keys())[0]
    algo = experiments[exp_name]["run"]

    # WandB logging support.
    callbacks = None
    # if we had to download the config file, remove the temp file.
    if temp_file:
        temp_file.close()

    run_fllib_experiments(
        experiments=experiments,
        v=v,
        vv=vv,
        trace=trace,
        ray_num_nodes=ray_num_nodes,
        ray_num_cpus=ray_num_cpus,
        ray_num_gpus=ray_num_gpus,
        ray_object_store_memory=ray_object_store_memory,
        ray_ui=ray_ui,
        ray_address=ray_address,
        local_mode=local_mode,
        resume=resume,
        scheduler=scheduler,
        scheduler_config=scheduler_config,
        algo=algo,
        callbacks=callbacks,
        # WandB options.
        wandb_key=wandb_key,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
    )


@train_app.callback(invoke_without_command=True)
def run(
    # Context object for subcommands
    ctx: typer.Context,
    # Config-based arguments.
    algo: str = cli.Algo,
    env: str = cli.Env,
    config: str = cli.Config,
    stop: str = cli.Stop,
    experiment_name: str = cli.ExperimentName,
    num_samples: int = cli.NumSamples,
    checkpoint_freq: int = cli.CheckpointFreq,
    checkpoint_at_end: bool = cli.CheckpointAtEnd,
    local_dir: str = cli.LocalDir,
    restore: str = cli.Restore,
    resources_per_trial: str = cli.ResourcesPerTrial,
    keep_checkpoints_num: int = cli.KeepCheckpointsNum,
    checkpoint_score_attr: str = cli.CheckpointScoreAttr,
    upload_dir: str = cli.UploadDir,
    # Additional config arguments used for overriding.
    v: bool = cli.V,
    vv: bool = cli.VV,
    trace: bool = cli.Trace,
    # Ray cluster options.
    local_mode: bool = cli.LocalMode,
    ray_address: str = cli.RayAddress,
    ray_ui: bool = cli.RayUi,
    ray_num_cpus: int = cli.RayNumCpus,
    ray_num_gpus: int = cli.RayNumGpus,
    ray_num_nodes: int = cli.RayNumNodes,
    ray_object_store_memory: int = cli.RayObjectStoreMemory,
    # Ray scheduling options.
    resume: bool = cli.Resume,
    scheduler: str = cli.Scheduler,
    scheduler_config: str = cli.SchedulerConfig,
):
    """Train a reinforcement learning agent from command line options. The
    options --env and --algo are required to run this command.

    Training example via RLlib CLI:\n     rllib train --algo DQN --env CartPole-v1\n\n
    """

    # If no subcommand is specified, simply run the following lines as the
    # "rllib train" main command.
    if ctx.invoked_subcommand is None:
        config = json.loads(config)
        resources_per_trial = json_to_resources(resources_per_trial)

        # Load a single experiment from configuration
        experiments = {
            experiment_name: {  # i.e. log to ~/ray_results/default
                "run": algo,
                "checkpoint_config": {
                    "checkpoint_frequency": checkpoint_freq,
                    "checkpoint_at_end": checkpoint_at_end,
                    "num_to_keep": keep_checkpoints_num,
                    "checkpoint_score_attribute": checkpoint_score_attr,
                },
                "local_dir": local_dir,
                "resources_per_trial": (
                    resources_per_trial and resources_to_json(resources_per_trial)
                ),
                "stop": json.loads(stop),
                "config": dict(config, env=env),
                "restore": restore,
                "num_samples": num_samples,
                "sync_config": {
                    "upload_dir": upload_dir,
                },
            }
        }

        run_fllib_experiments(
            experiments=experiments,
            v=v,
            vv=vv,
            trace=trace,
            ray_num_nodes=ray_num_nodes,
            ray_num_cpus=ray_num_cpus,
            ray_num_gpus=ray_num_gpus,
            ray_object_store_memory=ray_object_store_memory,
            ray_ui=ray_ui,
            ray_address=ray_address,
            local_mode=local_mode,
            resume=resume,
            scheduler=scheduler,
            scheduler_config=scheduler_config,
            algo=algo,
        )


def run_fllib_experiments(
    experiments: dict,
    v: cli.V,
    vv: cli.VV,
    trace: cli.Trace,
    ray_num_nodes: cli.RayNumNodes,
    ray_num_cpus: cli.RayNumCpus,
    ray_num_gpus: cli.RayNumGpus,
    ray_object_store_memory: cli.RayObjectStoreMemory,
    ray_ui: cli.RayUi,
    ray_address: cli.RayAddress,
    local_mode: cli.LocalMode,
    resume: cli.Resume,
    scheduler: cli.Scheduler,
    scheduler_config: cli.SchedulerConfig,
    algo: cli.Algo,
    callbacks=None,
    # WandB options.
    wandb_key: Optional[str] = cli.WandBKey,
    wandb_project: Optional[str] = cli.WandBProject,
    wandb_run_name: Optional[str] = cli.WandBRunName,
):
    """Main training function for the blades, whether you've loaded your
    experiments from a config file or from command line options."""

    # Override experiment data with command line arguments.
    verbose = 1
    for exp in experiments.values():
        if wandb_key is not None:
            exp_name = list(experiments.keys())[0]
            exp["config"]["wandb_api_key"] = wandb_key
            exp["config"]["wandb_project"] = wandb_project or exp_name
            exp["config"]["wandb_run_name"] = wandb_run_name or exp_name

        # Bazel makes it hard to find files specified in `args` (and `data`).
        # Look for them here.
        # NOTE: Some of our yaml files don't have a `config` section.
        input_ = exp.get("config", {}).get("input")
        if input_ and input_ != "sampler":
            exp["config"]["input"] = _patch_path(input_)
        if v:
            exp["config"]["log_level"] = "INFO"
            verbose = 3  # Print details on trial result
        if vv:
            exp["config"]["log_level"] = "DEBUG"
            verbose = 3  # Print details on trial result

    # Initialize the Ray cluster with the specified options.
    if ray_num_nodes:
        # Import this only here so that train.py also works with
        # older versions (and user doesn't use `--ray-num-nodes`).
        from ray.cluster_utils import Cluster

        cluster = Cluster()
        for _ in range(ray_num_nodes):
            cluster.add_node(
                num_cpus=ray_num_cpus or 1,
                num_gpus=ray_num_gpus or 0,
                object_store_memory=ray_object_store_memory,
            )
        ray.init(address=cluster.address)
    else:
        ray.init(
            # include_dashboard=ray_ui,
            # address=ray_address,
            # object_store_memory=ray_object_store_memory,
            # num_cpus=ray_num_cpus,
            # num_gpus=ray_num_gpus,
            # local_mode=local_mode,
        )

    # Run the Tune experiment and return the trials.
    scheduler_config = json.loads(scheduler_config)
    trials = run_experiments(
        experiments,
        scheduler=create_scheduler(scheduler, **scheduler_config),
        resume=resume,
        verbose=verbose,
        concurrent=True,
        callbacks=callbacks,
    )
    ray.shutdown()

    checkpoints = []
    for trial in trials:
        if trial.checkpoint.dir_or_data:
            checkpoints.append(trial.checkpoint.dir_or_data)

    if checkpoints:
        from rich import print
        from rich.panel import Panel

        print("\nYour training finished.")

        print("Best available checkpoint for each trial:")
        for cp in checkpoints:
            print(f"  {cp}")

        print(
            "\nYou can now evaluate your trained algorithm from any "
            "checkpoint, e.g. by running:"
        )
        print(Panel(f"[green]  fllib evaluate {checkpoints[0]} --algo {algo}"))


def main():
    """Run the CLI."""
    train_app()


if __name__ == "__main__":
    main()

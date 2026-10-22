from argparse import ArgumentParser
from typing import Optional
from os import getcwd, makedirs, environ, urandom
import shutil
import json

import random
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.profilers import PyTorchProfiler
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.progress.rich_progress import _RICH_AVAILABLE
from pytorch_lightning.loggers.wandb import _WANDB_AVAILABLE, WandbLogger

from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
    RichModelSummary,
    DeviceStatsMonitor,
    ModelSummary,
    TQDMProgressBar
)

from spanet import JetReconstructionModel, Options


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def generate_seed() -> int:
    """Draw a fresh seed from OS entropy (os.urandom), not from any of the
    RNGs set_seed() is about to touch -- so this can't be biased by, or
    confused with, a prior seeding call. Used whenever --random_seed isn't
    explicitly passed, so a run is still fully reproducible later even
    though nothing was chosen by hand: whatever value this draws gets
    printed and saved, and rerunning with --random_seed <that value>
    reproduces the model/optimization RNG state exactly."""
    return int.from_bytes(urandom(4), "big")


def main(
        event_file: str,
        training_file: str,
        validation_file: str,
        options_file: Optional[str],
        checkpoint: Optional[str],
        state_dict: Optional[str],
        freeze_state_dict: bool,

        log_dir: str,
        name: str,

        torch_script: bool,
        fp16: bool,
        verbose: bool,
        full_events: bool,

        profile: bool,
        gpus: Optional[int],
        epochs: Optional[int],
        time_limit: Optional[str],
        batch_size: Optional[int],
        limit_dataset: Optional[float],
        random_seed: int,
        dataset_seed: int,
    ):

    # Whether or not this script version is the master run or a worker
    master = True
    if "NODE_RANK" in environ:
        master = False

    # -------------------------------------------------------------------------------------------------------
    # Create options file and load any optional extra information.
    # -------------------------------------------------------------------------------------------------------
    options = Options(event_file, training_file, validation_file)

    if options_file is not None:
        with open(options_file, 'r') as json_file:
            options.update_options(json.load(json_file))

    # -------------------------------------------------------------------------------------------------------
    # Command line overrides for common option values.
    # -------------------------------------------------------------------------------------------------------
    options.verbose_output = verbose
    if master and verbose:
        print(f"Verbose output activated.")

    if full_events:
        if master:
            print(f"Overriding: Only using full events")
        options.partial_events = False
        options.balance_particles = False

    if gpus is not None:
        if master:
            print(f"Overriding GPU count: {gpus}")
        options.num_gpu = gpus

    if batch_size is not None:
        if master:
            print(f"Overriding Batch Size: {batch_size}")
        options.batch_size = batch_size

    if limit_dataset is not None:
        if master:
            print(f"Overriding Dataset Limit: {limit_dataset}%")
        options.dataset_limit = limit_dataset / 100

    if epochs is not None:
        if master:
            print(f"Overriding Number of Epochs: {epochs}")
        options.epochs = epochs

    # -------------------------------------------------------------------------------------------------------
    # Seeding. Two independent knobs, on purpose:
    #   - dataset_seed  -> options.dataset_randomization (train/val split, event ordering)
    #   - random_seed   -> python/numpy/torch/torch.cuda RNGs (init, dropout, etc. via set_seed())
    #
    # dataset_seed is only applied when explicitly passed (>0); left alone otherwise, so whatever
    # Options/options_file already set for dataset_randomization is preserved. random_seed is different:
    # we ALWAYS call set_seed(), and if random_seed isn't explicitly passed (0, the default) we draw a
    # fresh one from OS entropy first. That's what makes "set nothing" and "still reproducible" both true
    # at once -- previously, leaving --random_seed at 0 skipped set_seed() entirely, so python/numpy/torch
    # ended up seeded independently from unlogged OS/driver entropy with nothing printable to reproduce it.
    #
    # NOTE -- behavior change from before: dataset_randomization and the model/optimization seed used to
    # be driven off the SAME --random_seed value. They're independent now, and each follows the same rule:
    # set everything (pass both flags explicitly, for the hyperparameter scan, where every trial must be
    # fully reproducible and comparable) or set nothing (omit both, for the model-training seed sweep,
    # where every run should get its own fresh, auto-generated, auto-logged seed pair). There's no partial
    # state where one is fixed by hand and the other silently falls back to whatever's in the options file.
    # -------------------------------------------------------------------------------------------------------
    if dataset_seed > 0:
        dataset_randomization_seed = dataset_seed
    else:
        dataset_randomization_seed = generate_seed()
    options.dataset_randomization = dataset_randomization_seed

    if random_seed > 0:
        training_seed = random_seed
    else:
        training_seed = generate_seed()

    set_seed(training_seed)

    if master:
        print(f"Dataset randomization seed: {dataset_randomization_seed}"
              + ("" if dataset_seed > 0 else "  (auto-generated -- pass -ds/--dataset_seed "
                                              f"{dataset_randomization_seed} to reproduce this split)"))
        print(f"Model/training seed (python/numpy/torch/cuda): {training_seed}"
              + ("" if random_seed > 0 else "  (auto-generated -- pass -r/--random_seed "
                                             f"{training_seed} to reproduce this run)"))

    # -------------------------------------------------------------------------------------------------------
    # Print the full hyperparameter list
    # -------------------------------------------------------------------------------------------------------
    if master:
        options.display()

    # -------------------------------------------------------------------------------------------------------
    # Begin the training loop
    # -------------------------------------------------------------------------------------------------------

    # Create the initial model on the CPU
    model = JetReconstructionModel(options, torch_script)

    if state_dict is not None:
        if master:
            print(f"Loading state dict from: {state_dict}")

        state_dict = torch.load(state_dict, map_location="cpu")["state_dict"]
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        if master:
            print(f"Missing Keys: {missing_keys}")
            print(f"Unexpected Keys: {unexpected_keys}")

        if freeze_state_dict:
            for pname, parameter in model.named_parameters():
                if pname in state_dict:
                    parameter.requires_grad_(False)

    # Construct the logger for this training run. Logs will be saved in {logdir}/{name}/version_i
    log_dir = getcwd() if log_dir is None else log_dir
    logger = TensorBoardLogger(save_dir=log_dir, name=name)
    # logger = (
    #     WandbLogger(name=name, save_dir=log_dir)
    #     if _WANDB_AVAILABLE else
    #     TensorBoardLogger(save_dir=log_dir, name=name)
    # )

    # Create the checkpoints for this training run. Both track `validation_significance` (weighted S/sqrt(B) from the
    # detection scores, logged by JetReconstructionValidation) instead of validation_average_jet_accuracy.
    #  1) periodic: every 10 epochs, keep all (as before) -- the filename now carries the significance.
    #  2) best: every epoch, keep the 3 best by significance. Epoch-to-epoch values are noisy (a few heavy ttbar rows
    #     can move them), so look at the curve in TensorBoard as well before trusting a single "best" epoch.
    callbacks = [
        ModelCheckpoint(
            verbose=options.verbose_output,
            filename='{epoch}-{step}-{validation_significance:.5f}',
            monitor='validation_significance',
            save_top_k=-1,
            every_n_epochs=10,
            mode='max',
        ),
        ModelCheckpoint(
            verbose=options.verbose_output,
            filename='best-{epoch}-{step}-{validation_significance:.5f}',
            monitor='validation_significance',
            save_top_k=3,
            every_n_epochs=1,
            mode='max',
        ),
        LearningRateMonitor(),
        DeviceStatsMonitor(),
        RichProgressBar() if _RICH_AVAILABLE else TQDMProgressBar(),
        RichModelSummary(max_depth=1) if _RICH_AVAILABLE else ModelSummary(max_depth=1)
    ]

    epochs = options.epochs
    profiler = None
    if profile:
        epochs = 1
        profiler = PyTorchProfiler(emit_nvtx=True)

    # Create the final pytorch-lightning manager
    trainer = pl.Trainer(
        accelerator="gpu" if options.num_gpu > 0 else "auto",
        devices=options.num_gpu if options.num_gpu > 0 else "auto",
        strategy="ddp" if options.num_gpu > 1 else "auto",
        precision="16-mixed" if fp16 else "32-true",

        gradient_clip_val=options.gradient_clip if options.gradient_clip > 0 else None,
        max_epochs=epochs,
        max_time=time_limit,

        logger=logger,
        profiler=profiler,
        callbacks=callbacks
    )

    # Save the current hyperparameters to a json file in the checkpoint directory
    if master:
        print(f"Training Version {trainer.logger.version}")
        makedirs(trainer.logger.log_dir, exist_ok=True)

        # Record both seeds alongside the hyperparameters -- this is the persistent, per-run artifact,
        # so reproducing a run later doesn't depend on console/log output still being around.
        options_dict = dict(options.__dict__)
        options_dict["random_seed"] = training_seed
        options_dict["dataset_randomization_seed"] = options.dataset_randomization

        with open(f"{trainer.logger.log_dir}/options.json", 'w') as json_file:
            json.dump(options_dict, json_file, indent=4)

        shutil.copy2(options.event_info_file, f"{trainer.logger.log_dir}/event.yaml")

    trainer.fit(model, ckpt_path=checkpoint)
    # -------------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("-ef", "--event_file", type=str, default="",
                        help="Input file containing event symmetry information.")

    parser.add_argument("-tf", "--training_file", type=str, default="",
                        help="Input file containing training data.")

    parser.add_argument("-vf", "--validation_file", type=str, default="",
                        help="Input file containing Validation data. If not provided, will use training data split.")

    parser.add_argument("-of", "--options_file", type=str, default=None,
                        help="JSON file with option overloads.")

    parser.add_argument("-cf", "--checkpoint", type=str, default=None,
                        help="Optional checkpoint to load the training state from. "
                             "Fully restores model weights and optimizer state.")

    parser.add_argument("-sf", "--state_dict", type=str, default=None,
                        help="Load from checkpoint but only the model weights. "
                             "Can be partial as the weights don't have to match one-to-one.")

    parser.add_argument("-fsf", "--freeze_state_dict", action='store_true',
                        help="Freeze any weights that were loaded from the state dict. "
                             "Used for finetuning new layers.")

    parser.add_argument("-l", "--log_dir", type=str, default=None,
                        help="Output directory for the checkpoints and tensorboard logs. Default to current directory.")

    parser.add_argument("-n", "--name", type=str, default="spanet_output",
                        help="The sub-directory to create for this run and an identifier for WANDB.")

    parser.add_argument("-e", "--epochs", type=int, default=None,
                        help="Override number of epochs to train for")

    parser.add_argument("-t", "--time_limit", type=str, default=None,
                        help="Time limit for training, in the format DD:HH:MM:SS.")

    parser.add_argument("-g", "--gpus", type=int, default=None,
                        help="Override GPU count in hyperparameters.")

    parser.add_argument("-b", "--batch_size", type=int, default=None,
                        help="Override batch size in hyperparameters.")

    parser.add_argument("-f", "--full_events", action='store_true',
                        help="Limit training to only full events.")

    parser.add_argument("-p", "--limit_dataset", type=float, default=None,
                        help="Limit dataset to only the first L percent of the data (0 - 100).")

    parser.add_argument("-fp16", "--fp16", action="store_true",
                        help="Use Torch AMP for training.")

    parser.add_argument("-v", "--verbose", action='store_true',
                        help="Output additional information to console and log.")

    parser.add_argument("-r", "--random_seed", type=int, default=0,
                        help="Set the model/optimization random seed (python/numpy/torch/torch.cuda -- "
                             "init, dropout, batch shuffling, etc). If left at 0 (default), a fresh seed "
                             "is drawn from OS entropy each run, always applied, and printed/saved so the "
                             "run can be reproduced later by passing that value here. Independent of "
                             "--dataset_seed.")

    parser.add_argument("-ds", "--dataset_seed", type=int, default=0,
                        help="Set the dataset-randomization seed (train/val split, event ordering), "
                             "independent of --random_seed. If left at 0 (default), a fresh seed is "
                             "drawn from OS entropy each run, always applied, and printed/saved so the "
                             "split can be reproduced later by passing that value here. Set everything "
                             "(both this and --random_seed, same or different values) for the "
                             "hyperparameter scan, where every trial must be fixed and comparable; set "
                             "nothing (omit both) for the seed sweep, where every run should get its own "
                             "fresh, logged seed pair.")

    parser.add_argument("-ts", "--torch_script", action='store_true',
                        help="Compile the neural network using torchscript.")

    parser.add_argument("--profile", action='store_true',
                        help="Profile network for a single training epoch.")

    main(**parser.parse_args().__dict__)
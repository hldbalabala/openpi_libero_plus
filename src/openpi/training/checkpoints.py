from __future__ import annotations

import asyncio
import concurrent.futures as futures
import dataclasses
import logging
from typing import Protocol

from etils import epath
import flax.nnx as nnx
import flax.traverse_util
import jax
import orbax.checkpoint as ocp
import orbax.checkpoint.future as future
import orbax.checkpoint.transform_utils as transform_utils

from openpi.shared import array_typing as at
import openpi.shared.normalize as _normalize
import openpi.training.data_loader as _data_loader
import openpi.training.utils as training_utils


def initialize_checkpoint_dir(
    checkpoint_dir: epath.Path | str, *, keep_period: int | None, overwrite: bool, resume: bool
) -> tuple[ocp.CheckpointManager, bool]:
    checkpoint_dir = epath.Path(checkpoint_dir).resolve()
    resuming = False
    if checkpoint_dir.exists():
        if overwrite:
            checkpoint_dir.rmtree()
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Wiped checkpoint directory {checkpoint_dir}")
        elif resume:
            resuming = True
        else:
            raise FileExistsError(
                f"Checkpoint directory {checkpoint_dir} already exists. Use --overwrite or --resume "
                "to indicate how to handle it."
            )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    mngr = ocp.CheckpointManager(
        checkpoint_dir,
        item_handlers={
            "assets": CallbackHandler(),
            "train_state": ocp.PyTreeCheckpointHandler(),
            "params": ocp.PyTreeCheckpointHandler(),
        },
        options=ocp.CheckpointManagerOptions(
            max_to_keep=1,
            keep_period=keep_period,
            create=False,
            async_options=ocp.AsyncOptions(timeout_secs=7200),
        ),
    )

    # Special case: the checkpoint directory exists and the user requests to resume training, but the training run did
    # not get to the first checkpoint saved. In this case, we don't actually want the train script to try and restore a
    # checkpoint, since it will fail.
    if resuming and tuple(mngr.all_steps()) in [(), (0,)]:
        logging.info("Checkpoint directory exists, but does not contain any checkpoints. Aborting resume.")
        resuming = False

    return mngr, resuming


def save_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int,
):
    def save_assets(directory: epath.Path):
        # Save the normalization stats.
        data_config = data_loader.data_config()
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(directory / data_config.asset_id, norm_stats)

    # Split params that can be used for inference into a separate item.
    with at.disable_typechecking():
        train_state, params = _split_params(state)
    items = {
        "assets": save_assets,
        "train_state": train_state,
        "params": {"params": params},
    }
    checkpoint_manager.save(step, items)


def restore_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int | None = None,
) -> training_utils.TrainState:
    del data_loader

    with at.disable_typechecking():
        # Split params that can be used for inference into a separate item.
        train_state, params = _split_params(state)
        
        # Convert params to pure dict if it's a NNX State
        if isinstance(params, nnx.State):
            params = params.to_pure_dict()
        
        # Try to restore, but handle missing parameters gracefully
        try:
            restored = checkpoint_manager.restore(
                step,
                items={
                    "train_state": train_state,
                    "params": {"params": params},
                },
            )
            restored_params = restored["params"]["params"]
            # Convert to pure dict if it's a NNX State
            if isinstance(restored_params, nnx.State):
                restored_params = restored_params.to_pure_dict()
        except ValueError as e:
            # If restore fails due to structure mismatch, try partial restore
            if "tree structures do not match" in str(e) or "User-provided restore item" in str(e):
                logging.warning(
                    "Checkpoint structure mismatch detected. Attempting partial restore for missing parameters."
                )
                # Get the checkpoint path
                step_to_restore = step if step is not None else checkpoint_manager.latest_step()
                checkpoint_path = checkpoint_manager.directory / str(step_to_restore) / "params"
                
                # Try to restore only the matching parameters
                with ocp.PyTreeCheckpointer() as ckptr:
                    try:
                        # Get metadata to understand the checkpoint structure
                        checkpoint_metadata = ckptr.metadata(checkpoint_path)
                        checkpoint_params_shape = checkpoint_metadata.get("params", {})
                        
                        # Restore the checkpoint params (this will only restore what's in the checkpoint)
                        checkpoint_params_restored = ckptr.restore(
                            checkpoint_path,
                            ocp.args.PyTreeRestore(
                                item={"params": checkpoint_params_shape},
                            ),
                        )["params"]
                        
                        # Convert NNX State to pure dict if needed
                        if isinstance(checkpoint_params_restored, nnx.State):
                            checkpoint_params_restored = checkpoint_params_restored.to_pure_dict()
                        
                        # Merge checkpoint params with original params
                        # This will use checkpoint values where they exist, and keep original (randomly initialized) values for missing params
                        restored_params = _merge_checkpoint_params(params, checkpoint_params_restored)
                        logging.info("Successfully performed partial restore, keeping randomly initialized parameters for new layers.")
                    except Exception as restore_error:
                        logging.warning(
                            f"Failed to perform partial restore: {restore_error}. Using random initialization for new parameters."
                        )
                        restored_params = params
                
                # Still need to restore train_state
                # Try to restore train_state separately, handling potential structure mismatches
                step_to_restore = step if step is not None else checkpoint_manager.latest_step()
                restored_train_state = train_state
                try:
                    restored_train_state_dict = checkpoint_manager.restore(
                        step,
                        items={
                            "train_state": train_state,
                        },
                    )["train_state"]
                    # Check if step is a real value or ShapeDtypeStruct
                    if hasattr(restored_train_state_dict, "step"):
                        step_value = restored_train_state_dict.step
                        # If step is ShapeDtypeStruct, replace it with the actual step number
                        if isinstance(step_value, jax.ShapeDtypeStruct):
                            if step_to_restore is not None:
                                restored_train_state = dataclasses.replace(
                                    restored_train_state_dict,
                                    step=step_to_restore
                                )
                            else:
                                restored_train_state = restored_train_state_dict
                        else:
                            restored_train_state = restored_train_state_dict
                    else:
                        restored_train_state = restored_train_state_dict
                except Exception as train_state_error:
                    # If train_state restore fails, at least set the step from checkpoint path
                    logging.warning(f"Failed to restore train_state: {train_state_error}. Using step from checkpoint path.")
                    if step_to_restore is not None:
                        restored_train_state = dataclasses.replace(train_state, step=step_to_restore)
                    else:
                        restored_train_state = train_state
                
                restored = {"train_state": restored_train_state, "params": {"params": restored_params}}
            else:
                raise
        
        return _merge_params(restored["train_state"], restored["params"])


def load_norm_stats(assets_dir: epath.Path | str, asset_id: str) -> dict[str, _normalize.NormStats] | None:
    norm_stats_dir = epath.Path(assets_dir) / asset_id
    norm_stats = _normalize.load(norm_stats_dir)
    logging.info(f"Loaded norm stats from {norm_stats_dir}")
    return norm_stats


class Callback(Protocol):
    def __call__(self, directory: epath.Path) -> None: ...


class CallbackHandler(ocp.AsyncCheckpointHandler):
    """A CheckpointHandler for calling an arbitrary function asynchronously. Only for saving, not for restoring."""

    def save(self, directory: epath.Path, args: CallbackSave):
        if jax.process_index() == 0:
            args.callback(directory)

    async def async_save(self, directory: epath.Path, args: CallbackSave) -> list[futures.Future]:
        return [future.CommitFutureAwaitingContractedSignals(asyncio.to_thread(self.save, directory, args))]

    def restore(self, *args, **kwargs):
        raise NotImplementedError("CallbackHandler does not support restore")


@ocp.args.register_with_handler(CallbackHandler, for_save=True)
@dataclasses.dataclass
class CallbackSave(ocp.args.CheckpointArgs):
    callback: Callback


@ocp.args.register_with_handler(CallbackHandler, for_restore=True)
class CallbackRestore(ocp.args.CheckpointArgs): ...


def _split_params(state: training_utils.TrainState) -> tuple[training_utils.TrainState, at.Params]:
    if state.ema_params is not None:
        params = state.ema_params
        train_state = dataclasses.replace(state, ema_params=None)
    else:
        params = state.params
        train_state = dataclasses.replace(state, params={})
    return train_state, params


def _merge_params(train_state: training_utils.TrainState, params: dict[str, at.Params]) -> training_utils.TrainState:
    # Revert the logic inside `_split_params`. Assumes that existence of `params` means that EMA params were used during the split.
    if train_state.params:
        return dataclasses.replace(train_state, ema_params=params["params"])
    return dataclasses.replace(train_state, params=params["params"])


def _merge_checkpoint_params(original_params: at.Params, checkpoint_params: at.Params) -> at.Params:
    """Merge checkpoint parameters with original parameters.
    
    This allows new parameters (like image_out_proj) to keep their random initialization
    while restored parameters are loaded from checkpoint.
    
    Args:
        original_params: The current model parameters (with randomly initialized new layers)
        checkpoint_params: The parameters restored from checkpoint (may be missing some keys)
    
    Returns:
        Merged parameters: checkpoint values where available, original values for missing keys
    """
    flat_original = flax.traverse_util.flatten_dict(original_params, sep="/")
    flat_checkpoint = flax.traverse_util.flatten_dict(checkpoint_params, sep="/")
    
    # Start with original params (randomly initialized)
    result = flat_original.copy()
    
    # Overwrite with checkpoint params where they exist
    for key, value in flat_checkpoint.items():
        if key in result:
            result[key] = value
        else:
            # This shouldn't happen if checkpoint is a subset, but log it just in case
            logging.debug(f"Checkpoint has parameter {key} not in original model, skipping.")
    
    # Log which parameters are being kept from random initialization
    missing_keys = set(flat_original.keys()) - set(flat_checkpoint.keys())
    if missing_keys:
        logging.info(f"Keeping randomly initialized parameters for: {sorted(missing_keys)}")
    
    return flax.traverse_util.unflatten_dict(result, sep="/")

"""Execute the distributed denoiser and publish a latent after all ranks retire."""
import hashlib
import os
from pathlib import Path

import numpy as np
from c206_sampler import Schedule, initial_latent, unpatch
from sp3_control import communication_binding, startup, step_gate
from sp3_layout import CONDITIONS
from sp3_model import SP3Model
from sp3_transport import SP3Transport
from tp4_profile import PROFILE


def save_checkpoint(path, latent, binding, step):
    temporary = path.with_suffix(".incoming.npy")
    with temporary.open("wb") as stream:
        np.save(stream, latent.array)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    return dict(binding, status="SP3_STEP_CHECKPOINT_SAVED", group=0, rank=0, path=str(path),
                file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                latent_sha256=hashlib.sha256(memoryview(latent.array).cast("B")).hexdigest(),
                completed_step=step, next_step=step+1, shape=list(latent.shape),
                dtype=str(latent.dtype), latent_bytes=latent.nbytes)


def run(board, assets, *, seed, generation, cohort, output, control, timeout=43200,
        cancel_check=None, step_limit=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    role = assets.role
    is_root = role.group == 0 and role.rank == 0
    manifest = assets.manifest
    with np.load(assets.path(manifest["schedule"]), allow_pickle=False) as arrays:
        schedule = Schedule.from_arrays(arrays, hidden=PROFILE["hidden"])
    if schedule.updates != assets.updates:
        raise ValueError("Sampler and assets disagree on the number of updates")
    steps = assets.updates if step_limit is None else step_limit
    if type(steps) is not int or not 1 <= steps <= assets.updates:
        raise ValueError("Invalid update count")
    binding = dict(cohort=cohort, generation=generation, profile_sha256=PROFILE["profile_sha256"],
                   source_binding_sha256=manifest["source_binding_sha256"],
                   source_manifest_sha256=manifest["source_manifest_sha256"],
                   schedule_sha256=manifest["files"][manifest["schedule"]],
                   updates=assets.updates, seed=seed, execution_steps=steps)
    binding.update(communication_binding(board))
    transport = SP3Transport(board, role, generation)
    pending = output / "current_latent.npy"
    model = None
    try:
        model = SP3Model(board, assets, transport, generation=generation, cancel_check=cancel_check)
        model.execution_steps = steps
        ready = startup(board, role, binding, control, timeout=timeout, cancel_check=cancel_check)
        tensors = model.tensors
        latent = tensors["latent"]
        if is_root:
            noise = np.load(assets.path(manifest["noise_reference"]), allow_pickle=False)
            latent.array[:] = initial_latent(latent.shape, seed, schedule.sigmas[0], noise,
                                              backend="portable")[0]
            latent.root.flush()
        transport.share_latent(latent, step=0)
        for step in range(steps):
            if cancel_check is not None:
                cancel_check()
            latent.root.invalidate()
            patch, _ = schedule.input_for_step(latent.array, step)
            model.initialize(patch[role.token_offset:role.token_offset+role.rows], step=step)
            model.run_step(step)
            for condition in CONDITIONS:
                prediction = model.finish(condition, step)
                transport.gather_prediction(prediction, tensors[condition+"_full"],
                                            step=step, condition=condition)
            checkpoint = None
            if is_root:
                for condition in CONDITIONS:
                    tensors[condition+"_full"].root.invalidate()
                latent.array[:] = schedule.advance(latent.array, unpatch(tensors["positive_full"].array),
                                                   unpatch(tensors["negative_full"].array), step)
                latent.root.flush()
                checkpoint = save_checkpoint(pending, latent, binding, step)
            step_gate(ready, step, control, board=board, model=model, checkpoint=checkpoint,
                      timeout=timeout, cancel_check=cancel_check)
            if step+1 < steps:
                transport.share_latent(latent, step=step+1)
        model.close()
        transport.close()
        if is_root:
            pending.replace(output / "final_latent.npy")
        return dict(completed_steps=steps, schedule_updates=assets.updates)
    except BaseException:
        board.cancel_native()
        if model is not None:
            model.join_workers()
        transport.close()
        raise

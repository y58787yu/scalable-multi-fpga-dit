"""Run one logical role of the Cosmos inference cluster on a ZCU102."""
import argparse
import os
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parent


def configure_groups(groups):
    if (ROOT / "tp2_profile_sp1.json").is_file():
        if groups not in (1, 2):
            raise ValueError("The 2B image supports one or two groups")
        os.environ["C209_TP2_PROFILE"] = str(ROOT / f"tp2_profile_sp{groups}.json")
    elif groups != 3:
        raise ValueError("The 14B image requires three groups")


def polarities(text):
    values = text.split(",")
    if len(values) != 4 or any(v not in ("0", "1") for v in values):
        raise argparse.ArgumentTypeError("Provide four comma-separated bits in physical SFP order")
    return [int(v) for v in values]


def stop_board(board):
    # The native call can return early on cancellation. Only release CMA after
    # the batch master and all DMA channels have retired their transactions.
    board.cancel_native()
    while board.batch_status()["busy"]:
        time.sleep(.05)
    while not board.stop_owned_dma():
        time.sleep(.05)
    board.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", type=int, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--groups", type=int, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--nonce", required=True)
    parser.add_argument("--generation", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--rx-polarities", type=polarities, required=True)
    args = parser.parse_args()
    configure_groups(args.groups)
    os.environ.setdefault("BOARD", "ZCU102")
    os.environ.setdefault("XILINX_XRT", "/usr")
    from tp4_startup import atomic_json
    from tp4_profile import PROFILE, Role, validate_generation

    validate_generation(args.generation)
    role = Role(args.group, args.rank)
    if not (0 <= args.seed < 2**32 and args.steps >= 1 and args.cohort and args.nonce):
        parser.error("Supply a uint32 seed, a positive step count and nonempty run identities")
    job = args.job.resolve()
    job.mkdir(parents=True, exist_ok=True)
    if any((job / f).exists() for f in ("initialize.json", "release.json", "prepared.json",
                                         "ready.json", "result.json")) or any(job.glob("step_*.json")):
        raise FileExistsError(f"Job already contains cluster state: {job}")
    output = job / "output"
    output.mkdir(exist_ok=False)
    identity = dict(group=args.group, rank=args.rank, nonce=args.nonce, cohort=args.cohort,
                    generation=args.generation, seed=args.seed, execution_steps=args.steps)

    def publish(status, **fields):
        atomic_json(job / "result.json", dict(identity, status=status, **fields))

    board = cancellation = None
    try:
        publish("STARTING")
        from sp3_assets import SP3Assets
        from board_driver_tp4 import TP4Board
        from sp3_cancel import SP3Cancellation
        from sp3_server import run
        from release_assets import verify_hardware

        verify_hardware(ROOT)
        assets = SP3Assets(args.assets, role, PROFILE["profile_sha256"])
        if args.steps not in (1, assets.updates):
            raise ValueError(f"Use one update or the complete {assets.updates}-step schedule")
        cancellation = SP3Cancellation(ROOT, job, args.nonce)
        cancellation.start()
        board = TP4Board(ROOT, role, load=True, rx_polarities=args.rx_polarities)
        result = run(board, assets, seed=args.seed, generation=args.generation, cohort=args.cohort,
                     output=output, control=job, cancel_check=cancellation.check,
                     step_limit=args.steps)
        if result["completed_steps"] != args.steps:
            raise RuntimeError("The final step did not retire")
        board.close()
        board = None
        publish("COMPLETED", actual_completed_steps=args.steps)
    except BaseException as error:
        publish("FAILED", error=str(error))
        if board is not None:
            try:
                stop_board(board)
            except BaseException as cleanup_error:
                # Exiting while the PL can still reach these CMA pages risks
                # corrupting the next allocation. Keep this process and its lease
                # alive until the operator resets or powers down the board.
                from threading import Event
                publish("FAILED", error=str(error), cleanup_error=str(cleanup_error),
                        hardware_lease_held=True)
                Event().wait()
        raise
    finally:
        if cancellation is not None:
            cancellation.close()


if __name__ == "__main__":
    main()

"""CPU tests for the static Stokes batch generator."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from solver.gen_data.pipeline.batch_storage import (  # noqa: E402
    batch_path,
    load_completed_batch,
)
from solver.gen_data.pipeline.simulation_allocation import (  # noqa: E402
    PhysicalFamilyId,
    DatasetSplit,
)
from solver.gen_data.pipeline.dataset_generation import (  # noqa: E402
    DatasetChunkConfig,
    generate_simulations,
    scan_dataset_generation,
)
from solver.gen_data.stokes_sampling import (  # noqa: E402
    STOKES_PARAMETER_GROUP_IDS,
    StokesSample,
    sample_stokes_simulation,
)
from solver.gen_data.stokes_batch_generator import (  # noqa: E402
    make_static_stokes_batch_generator,
)
from solver.gen_data.stokes_static_pipeline import (  # noqa: E402
    PAPER_STATIC_STOKES_CONTRACT,
    StaticStokesContract,
)

jax.config.update("jax_enable_x64", True)


class InjectedInterruption(OSError):
    """Controlled exception not converted to a numerical simulation rejection."""


def _contract() -> StaticStokesContract:
    return StaticStokesContract(
        nx=64,
        length=2.0 * math.pi,
        dno_order=2,
        pad_factor=2,
        maximum_wavenumber=24.0,
        role="reduced_wiring_evidence_only",
    )


def _run_spec(
    root: Path,
    *,
    parameter_group_ids: tuple[str, ...],
    targets: tuple[int, ...],
    batch_size: int = 2,
) -> DatasetChunkConfig:
    return DatasetChunkConfig(
        root=root,
        family_name="stokes",
        family_id=PhysicalFamilyId.STOKES,
        dataset_split=DatasetSplit.TEST,
        simulation_targets=dict(zip(parameter_group_ids, targets, strict=True)),
        batch_size=batch_size,
    )


def _zero_state(
    sample: StokesSample,
    contract: StaticStokesContract,
) -> tuple[jax.Array, jax.Array]:
    del sample
    zeros = jnp.zeros(contract.nx, dtype=jnp.float64)
    return zeros, zeros


def _identity_target(
    eta: jax.Array,
    xi: jax.Array,
    depth: float | jax.Array,
    *,
    nx: int,
    length: float,
    dno_order: int,
    pad_factor: int,
    maximum_wavenumber: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    del depth, nx, length, dno_order, pad_factor, maximum_wavenumber
    return eta, xi, jnp.zeros_like(eta)


class StaticStokesBatchExecutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_exhaustion_is_zero_row_rejection_and_sibling_still_commits(
        self,
    ) -> None:
        contract = _contract()
        finite_cell = STOKES_PARAMETER_GROUP_IDS[0]
        deep_cell = STOKES_PARAMETER_GROUP_IDS[2]
        spec = _run_spec(
            self.root,
            parameter_group_ids=(finite_cell, deep_cell),
            targets=(1, 1),
        )
        sampled_attempts: list[int] = []
        constructed_attempts: list[int] = []
        attempt_by_sample_id: dict[int, int] = {}
        target_calls = 0

        def controlled_sampler(
            parameter_group_id: str,
            *,
            dataset_split: DatasetSplit,
            attempt_number: int,
            domain_length: float,
            gravity: float,
            maximum_ursell_redraws: int,
        ) -> StokesSample:
            sampled_attempts.append(attempt_number)
            forced_ursell = 27.0 if attempt_number == 0 else 1.0
            with patch(
                "solver.gen_data.stokes_sampling._evaluate_finite_depth_ursell",
                return_value=forced_ursell,
            ):
                sample = sample_stokes_simulation(
                    parameter_group_id,
                    dataset_split=dataset_split,
                    attempt_number=attempt_number,
                    domain_length=domain_length,
                    gravity=gravity,
                    maximum_ursell_redraws=maximum_ursell_redraws,
                )
                attempt_by_sample_id[id(sample)] = attempt_number
                return sample

        def checked_constructor(
            sample: StokesSample,
            contract: StaticStokesContract,
        ) -> tuple[jax.Array, jax.Array]:
            self.assertFalse(
                batch_path(
                    self.root,
                    family="stokes",
                    split="test",
                    batch_id=len(constructed_attempts),
                ).exists()
            )
            constructed_attempts.append(attempt_by_sample_id[id(sample)])
            return _zero_state(sample, contract)

        def checked_target(
            eta: jax.Array,
            xi: jax.Array,
            depth: float | jax.Array,
            *,
            nx: int,
            length: float,
            dno_order: int,
            pad_factor: int,
            maximum_wavenumber: float,
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            nonlocal target_calls
            target_calls += 1
            return _identity_target(
                eta,
                xi,
                depth,
                nx=nx,
                length=length,
                dno_order=dno_order,
                pad_factor=pad_factor,
                maximum_wavenumber=maximum_wavenumber,
            )

        generator = make_static_stokes_batch_generator(
            chunk_config=spec,
            contract=contract,
            maximum_ursell_redraws=0,
            sampler=controlled_sampler,
            state_constructor=checked_constructor,
            target_evaluator=checked_target,
        )
        state = generate_simulations(spec, generator)

        self.assertTrue(state.complete)
        self.assertEqual(
            dict(state.accepted_simulation_counts),
            {finite_cell: 1, deep_cell: 1},
        )
        self.assertEqual(sampled_attempts, [0, 1, 2])
        self.assertEqual(constructed_attempts, [1, 2])
        self.assertEqual(target_calls, 2)
        self.assertEqual(len(state.completed_batches), 2)

        first = load_completed_batch(state.completed_batches[0])
        specifications = [
            json.loads(str(value)) for value in first.plan["simulation_spec_json"]
        ]
        self.assertEqual(
            [record["status"] for record in specifications],
            ["failed_ursell_redraw_limit", "accepted"],
        )
        self.assertEqual(
            specifications[0]["amplitude_attempts"][0]["ursell_upper_bound"],
            27.0,
        )
        self.assertEqual(
            [simulation.accepted for simulation in first.simulations],
            [False, True],
        )
        self.assertEqual(first.simulations[0].failed_checks, ("outside_support",))
        assert first.shard is not None
        np.testing.assert_array_equal(
            first.shard["simulation_local_index"],
            np.asarray([1], dtype=np.int32),
        )

        second = load_completed_batch(state.completed_batches[1])
        self.assertEqual(str(second.plan["parameter_group_id"][0]), finite_cell)

        prior_sample_count = len(sampled_attempts)
        replay = generate_simulations(spec, generator)
        self.assertTrue(replay.complete)
        self.assertEqual(len(sampled_attempts), prior_sample_count)

    def test_interrupted_batch_is_resampled_exactly_on_restart(self) -> None:
        contract = _contract()
        deep_cell = STOKES_PARAMETER_GROUP_IDS[2]
        spec = _run_spec(
            self.root,
            parameter_group_ids=(deep_cell,),
            targets=(2,),
        )
        first_records: list[dict[str, object]] = []
        replay_records: list[dict[str, object]] = []

        def recording_sampler(records: list[dict[str, object]]):
            def sample(
                parameter_group_id: str,
                *,
                dataset_split: DatasetSplit,
                attempt_number: int,
                domain_length: float,
                gravity: float,
                maximum_ursell_redraws: int,
            ) -> StokesSample:
                value = sample_stokes_simulation(
                    parameter_group_id,
                    dataset_split=dataset_split,
                    attempt_number=attempt_number,
                    domain_length=domain_length,
                    gravity=gravity,
                    maximum_ursell_redraws=maximum_ursell_redraws,
                )
                records.append(value.to_json_record())
                return value

            return sample

        def interrupt_after_proposal(
            sample: StokesSample,
            contract: StaticStokesContract,
        ) -> tuple[jax.Array, jax.Array]:
            del contract
            raise InjectedInterruption("during batch evaluation")

        interrupted = make_static_stokes_batch_generator(
            chunk_config=spec,
            contract=contract,
            maximum_ursell_redraws=0,
            sampler=recording_sampler(first_records),
            state_constructor=interrupt_after_proposal,
            target_evaluator=_identity_target,
        )
        with self.assertRaisesRegex(
            InjectedInterruption,
            "during batch evaluation",
        ):
            generate_simulations(spec, interrupted)

        expected_path = batch_path(
            self.root,
            family="stokes",
            split="test",
            batch_id=0,
        )
        self.assertFalse(expected_path.exists())
        self.assertEqual(scan_dataset_generation(spec).completed_batches, ())

        resumed = make_static_stokes_batch_generator(
            chunk_config=spec,
            contract=contract,
            maximum_ursell_redraws=0,
            sampler=recording_sampler(replay_records),
            state_constructor=_zero_state,
            target_evaluator=_identity_target,
        )
        state = generate_simulations(spec, resumed)
        self.assertTrue(state.complete)
        self.assertEqual(first_records, replay_records)
        self.assertEqual(state.completed_batches, (expected_path,))

    def test_paper_role_rejects_injected_implementation_hooks(self) -> None:
        contract = PAPER_STATIC_STOKES_CONTRACT
        deep_cell = STOKES_PARAMETER_GROUP_IDS[2]
        spec = _run_spec(
            self.root,
            parameter_group_ids=(deep_cell,),
            targets=(1,),
            batch_size=1,
        )
        make_static_stokes_batch_generator(
            chunk_config=spec,
            contract=contract,
        )
        with self.assertRaisesRegex(ValueError, "paper-dataset execution"):
            make_static_stokes_batch_generator(
                chunk_config=spec,
                contract=contract,
                state_constructor=_zero_state,
            )
        with self.assertRaisesRegex(ValueError, "paper-dataset execution"):
            make_static_stokes_batch_generator(
                chunk_config=spec,
                contract=contract,
                target_evaluator=_identity_target,
            )
        zero_redraw_spec = _run_spec(
            self.root / "zero_redraw",
            parameter_group_ids=(deep_cell,),
            targets=(1,),
            batch_size=1,
        )
        with self.assertRaisesRegex(ValueError, "paper-dataset execution"):
            make_static_stokes_batch_generator(
                chunk_config=zero_redraw_spec,
                contract=contract,
                maximum_ursell_redraws=0,
            )


if __name__ == "__main__":
    unittest.main()

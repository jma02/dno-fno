"""CPU tests for the static Stokes batch executor."""

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
    ParameterGroupTarget,
    PhysicalFamilyId,
    DatasetSplit,
)
from solver.gen_data.pipeline.simulation_checks import SimulationCheck  # noqa: E402
from solver.gen_data.pipeline.dataset_generation import (  # noqa: E402
    DatasetChunkConfig,
    generate_simulations,
    scan_dataset_generation,
)
from solver.gen_data.stokes_sampling import (  # noqa: E402
    DEFAULT_MAXIMUM_URSELL_REDRAWS,
    STOKES_PARAMETER_GROUP_IDS,
    StokesSample,
    sample_stokes_simulation,
)
from solver.gen_data.stokes_batch_executor import (  # noqa: E402
    make_static_stokes_batch_executor,
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
    contract: StaticStokesContract,
    *,
    parameter_group_ids: tuple[str, ...],
    targets: tuple[int, ...],
    maximum_ursell_redraws: int = 0,
    batch_size: int = 2,
) -> DatasetChunkConfig:
    return DatasetChunkConfig(
        root=root,
        family_name="stokes",
        family_id=PhysicalFamilyId.STOKES,
        dataset_split=DatasetSplit.TEST,
        worker_stream_id=11,
        simulation_targets=tuple(
            ParameterGroupTarget(parameter_group_id, target)
            for parameter_group_id, target in zip(parameter_group_ids, targets)
        ),
        batch_size=batch_size,
        first_attempt_index=0,
        configuration={
            "contract": contract.to_json_record(),
            "sampler": {
                "maximum_ursell_redraws": maximum_ursell_redraws,
            },
        },
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
            contract,
            parameter_group_ids=(finite_cell, deep_cell),
            targets=(1, 1),
        )
        sampled_attempts: list[int] = []
        constructed_attempts: list[int] = []
        target_calls = 0

        def controlled_sampler(
            assignment,
            *,
            domain_length: float,
            gravity: float,
            maximum_ursell_redraws: int,
        ) -> StokesSample:
            sampled_attempts.append(assignment.simulation_key.attempt_index)
            forced_ursell = (
                27.0 if assignment.simulation_key.attempt_index == 0 else 1.0
            )
            with patch(
                "solver.gen_data.stokes_sampling._evaluate_finite_depth_ursell",
                return_value=forced_ursell,
            ):
                return sample_stokes_simulation(
                    assignment,
                    domain_length=domain_length,
                    gravity=gravity,
                    maximum_ursell_redraws=maximum_ursell_redraws,
                )

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
            constructed_attempts.append(sample.assignment.simulation_key.attempt_index)
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

        executor = make_static_stokes_batch_executor(
            chunk_config=spec,
            contract=contract,
            maximum_ursell_redraws=0,
            metadata={"test_scope": "forced_exhaustion_and_sibling"},
            sampler=controlled_sampler,
            state_constructor=checked_constructor,
            target_evaluator=checked_target,
        )
        state = generate_simulations(spec, executor)

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
        self.assertTrue(
            first.simulations[0].failed_bits & int(SimulationCheck.OUTSIDE_SUPPORT)
        )
        self.assertEqual(first.metadata["sampling_exhaustions"], 1)
        assert first.shard is not None
        np.testing.assert_array_equal(
            first.shard["simulation_local_index"],
            np.asarray([1], dtype=np.int32),
        )

        second = load_completed_batch(state.completed_batches[1])
        replacement = json.loads(str(second.plan["simulation_spec_json"][0]))
        self.assertEqual(replacement["parameter_group_id"], finite_cell)
        self.assertEqual(replacement["attempt_index"], 2)

        prior_sample_count = len(sampled_attempts)
        replay = generate_simulations(spec, executor)
        self.assertTrue(replay.complete)
        self.assertEqual(len(sampled_attempts), prior_sample_count)

    def test_interrupted_batch_is_resampled_exactly_on_restart(self) -> None:
        contract = _contract()
        deep_cell = STOKES_PARAMETER_GROUP_IDS[2]
        spec = _run_spec(
            self.root,
            contract,
            parameter_group_ids=(deep_cell,),
            targets=(2,),
        )
        first_records: list[dict[str, object]] = []
        replay_records: list[dict[str, object]] = []

        def recording_sampler(records: list[dict[str, object]]):
            def sample(
                assignment,
                *,
                domain_length: float,
                gravity: float,
                maximum_ursell_redraws: int,
            ) -> StokesSample:
                value = sample_stokes_simulation(
                    assignment,
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

        interrupted = make_static_stokes_batch_executor(
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

        resumed = make_static_stokes_batch_executor(
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

    def test_executor_requires_contract_and_sampler_to_match_run_record(
        self,
    ) -> None:
        contract = _contract()
        deep_cell = STOKES_PARAMETER_GROUP_IDS[2]
        wrong_contract_spec = DatasetChunkConfig(
            root=self.root / "contract",
            family_name="stokes",
            family_id=PhysicalFamilyId.STOKES,
            dataset_split=DatasetSplit.TEST,
            worker_stream_id=11,
            simulation_targets=(ParameterGroupTarget(deep_cell, 1),),
            batch_size=1,
            configuration={
                "contract": {
                    **contract.to_json_record(),
                    "maximum_wavenumber": 16.0,
                },
                "sampler": {"maximum_ursell_redraws": 0},
            },
        )
        with self.assertRaisesRegex(ValueError, "contract differs"):
            make_static_stokes_batch_executor(
                chunk_config=wrong_contract_spec,
                contract=contract,
                maximum_ursell_redraws=0,
            )

        wrong_sampler_spec = _run_spec(
            self.root / "sampler",
            contract,
            parameter_group_ids=(deep_cell,),
            targets=(1,),
            maximum_ursell_redraws=1,
            batch_size=1,
        )
        with self.assertRaisesRegex(ValueError, "redraw limit differs"):
            make_static_stokes_batch_executor(
                chunk_config=wrong_sampler_spec,
                contract=contract,
                maximum_ursell_redraws=0,
            )

    def test_paper_role_rejects_injected_implementation_hooks(self) -> None:
        contract = PAPER_STATIC_STOKES_CONTRACT
        deep_cell = STOKES_PARAMETER_GROUP_IDS[2]
        spec = _run_spec(
            self.root,
            contract,
            parameter_group_ids=(deep_cell,),
            targets=(1,),
            maximum_ursell_redraws=DEFAULT_MAXIMUM_URSELL_REDRAWS,
            batch_size=1,
        )
        make_static_stokes_batch_executor(
            chunk_config=spec,
            contract=contract,
        )
        with self.assertRaisesRegex(ValueError, "paper-dataset execution"):
            make_static_stokes_batch_executor(
                chunk_config=spec,
                contract=contract,
                state_constructor=_zero_state,
            )
        with self.assertRaisesRegex(ValueError, "paper-dataset execution"):
            make_static_stokes_batch_executor(
                chunk_config=spec,
                contract=contract,
                target_evaluator=_identity_target,
            )
        zero_redraw_spec = _run_spec(
            self.root / "zero_redraw",
            contract,
            parameter_group_ids=(deep_cell,),
            targets=(1,),
            maximum_ursell_redraws=0,
            batch_size=1,
        )
        with self.assertRaisesRegex(ValueError, "paper-dataset execution"):
            make_static_stokes_batch_executor(
                chunk_config=zero_redraw_spec,
                contract=contract,
                maximum_ursell_redraws=0,
            )


if __name__ == "__main__":
    unittest.main()

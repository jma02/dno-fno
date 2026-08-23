"""CPU tests for the static Stokes accepted-quota executor."""

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

from solver.gen_data.pipeline.archive import (  # noqa: E402
    BatchStatus,
    file_sha256,
    inspect_batch,
)
from solver.gen_data.pipeline.production import (  # noqa: E402
    CellQuota,
    PhysicalFamilyId,
    SplitId,
)
from solver.gen_data.pipeline.quality import QualityReason  # noqa: E402
from solver.gen_data.pipeline.quota_driver import (  # noqa: E402
    AcceptedQuotaRunSpec,
    run_accepted_quotas,
    scan_quota_run,
)
from solver.gen_data.pipeline.reference import DiscreteDnoTarget  # noqa: E402
from solver.gen_data.stokes_sampling import (  # noqa: E402
    DEFAULT_MAXIMUM_URSELL_REDRAWS,
    STOKES_SAMPLE_CELLS,
    StokesSample,
    sample_stokes_case,
)
from solver.gen_data.stokes_quota_executor import (  # noqa: E402
    StaticStokesQuotaExecutor,
)
from solver.gen_data.stokes_static_pipeline import (  # noqa: E402
    PAPER_STATIC_STOKES_CONTRACT,
    StaticStokesContract,
)

jax.config.update("jax_enable_x64", True)


class InjectedInterruption(OSError):
    """Controlled exception not converted to a numerical case rejection."""


def _contract() -> StaticStokesContract:
    return StaticStokesContract.reduced_wiring_evidence(
        DiscreteDnoTarget(
            nx=64,
            length=2.0 * math.pi,
            dno_order=2,
            pad_factor=2,
            maximum_wavenumber=24.0,
        )
    )


def _run_spec(
    root: Path,
    contract: StaticStokesContract,
    *,
    cell_ids: tuple[str, ...],
    targets: tuple[int, ...],
    maximum_ursell_redraws: int = 0,
    batch_size: int = 2,
) -> AcceptedQuotaRunSpec:
    return AcceptedQuotaRunSpec(
        root=root,
        family_name="stokes",
        family_id=PhysicalFamilyId.STOKES,
        revision_id=1,
        split_id=SplitId.TEST,
        stream_id=11,
        quotas=tuple(
            CellQuota(cell_id, target) for cell_id, target in zip(cell_ids, targets)
        ),
        cell_codes={cell_id: index for index, cell_id in enumerate(cell_ids)},
        batch_size=batch_size,
        first_attempt_index=0,
        configuration={
            "contract": contract.to_json_record(),
            "sampler": {
                "maximum_ursell_redraws": maximum_ursell_redraws,
            },
        },
    )


def _proposal_contains(root: Path, case_id: int) -> bool:
    for path in (root / "proposals/stokes/test").glob("batch_*.npz"):
        with np.load(path, allow_pickle=False) as proposal:
            if case_id in set(map(int, proposal["case_id"])):
                return True
    return False


def _zero_state(
    sample: StokesSample,
    contract: StaticStokesContract,
) -> tuple[jax.Array, jax.Array]:
    del sample
    zeros = jnp.zeros(contract.target.nx, dtype=jnp.float64)
    return zeros, zeros


def _identity_target(
    eta: jax.Array,
    xi: jax.Array,
    depth: float | jax.Array,
    *,
    definition: DiscreteDnoTarget,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    del depth, definition
    return eta, xi, jnp.zeros_like(eta)


class StaticStokesQuotaExecutorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_exhaustion_is_zero_row_rejection_and_sibling_still_commits(
        self,
    ) -> None:
        contract = _contract()
        finite_cell = STOKES_SAMPLE_CELLS[0].cell_id
        deep_cell = STOKES_SAMPLE_CELLS[2].cell_id
        spec = _run_spec(
            self.root,
            contract,
            cell_ids=(finite_cell, deep_cell),
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
            sampled_attempts.append(assignment.case_key.attempt_index)
            forced_ursell = 27.0 if assignment.case_key.attempt_index == 0 else 1.0
            with patch(
                "solver.gen_data.stokes_sampling._evaluate_finite_depth_ursell",
                return_value=forced_ursell,
            ):
                return sample_stokes_case(
                    assignment,
                    domain_length=domain_length,
                    gravity=gravity,
                    maximum_ursell_redraws=maximum_ursell_redraws,
                )

        def checked_constructor(
            sample: StokesSample,
            active_contract: StaticStokesContract,
        ) -> tuple[jax.Array, jax.Array]:
            self.assertTrue(
                _proposal_contains(
                    self.root,
                    sample.assignment.case_key.case_id,
                )
            )
            constructed_attempts.append(sample.assignment.case_key.attempt_index)
            return _zero_state(sample, active_contract)

        def checked_target(
            eta: jax.Array,
            xi: jax.Array,
            depth: float | jax.Array,
            *,
            definition: DiscreteDnoTarget,
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            nonlocal target_calls
            target_calls += 1
            return _identity_target(
                eta,
                xi,
                depth,
                definition=definition,
            )

        executor = StaticStokesQuotaExecutor(
            run_spec=spec,
            contract=contract,
            maximum_ursell_redraws=0,
            metadata={"test_scope": "forced_exhaustion_and_sibling"},
            sampler=controlled_sampler,
            state_constructor=checked_constructor,
            target_evaluator=checked_target,
        )
        state = run_accepted_quotas(spec, executor)

        self.assertTrue(state.complete)
        self.assertEqual(
            dict(state.accepted_by_cell),
            {finite_cell: 1, deep_cell: 1},
        )
        self.assertEqual(sampled_attempts, [0, 1, 2])
        self.assertEqual(constructed_attempts, [1, 2])
        self.assertEqual(target_calls, 2)
        self.assertEqual(state.next_attempt_index, 3)
        self.assertEqual(len(state.committed), 2)

        first = state.committed[0]
        self.assertFalse(first.failure.exists())
        with np.load(first.proposal, allow_pickle=False) as proposal:
            specifications = [
                json.loads(str(value)) for value in proposal["case_spec_json"]
            ]
        self.assertEqual(
            [record["status"] for record in specifications],
            ["failed_ursell_redraw_limit", "accepted"],
        )
        self.assertEqual(
            specifications[0]["amplitude_attempts"][0]["ursell_upper_bound"],
            27.0,
        )
        first_result = json.loads(first.result.read_text(encoding="utf-8"))
        self.assertEqual(
            [case["accepted"] for case in first_result["cases"]],
            [False, True],
        )
        self.assertTrue(
            first_result["cases"][0]["failed_bits"] & int(QualityReason.OUTSIDE_SUPPORT)
        )
        self.assertEqual(
            first_result["metadata"]["sampling_exhaustions"],
            1,
        )
        with np.load(first.shard, allow_pickle=False) as shard:
            np.testing.assert_array_equal(
                shard["case_local_index"],
                np.asarray([1], dtype=np.int32),
            )

        second = state.committed[1]
        with np.load(second.proposal, allow_pickle=False) as proposal:
            replacement = json.loads(str(proposal["case_spec_json"][0]))
        self.assertEqual(replacement["cell_id"], finite_cell)
        self.assertEqual(replacement["attempt_index"], 2)

        prior_sample_count = len(sampled_attempts)
        replay = run_accepted_quotas(spec, executor)
        self.assertTrue(replay.complete)
        self.assertEqual(len(sampled_attempts), prior_sample_count)

    def test_proposal_only_interruption_resamples_exactly_on_replay(self) -> None:
        contract = _contract()
        deep_cell = STOKES_SAMPLE_CELLS[2].cell_id
        spec = _run_spec(
            self.root,
            contract,
            cell_ids=(deep_cell,),
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
                value = sample_stokes_case(
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
            active_contract: StaticStokesContract,
        ) -> tuple[jax.Array, jax.Array]:
            del active_contract
            self.assertTrue(
                _proposal_contains(
                    self.root,
                    sample.assignment.case_key.case_id,
                )
            )
            raise InjectedInterruption("after durable proposal")

        interrupted = StaticStokesQuotaExecutor(
            run_spec=spec,
            contract=contract,
            maximum_ursell_redraws=0,
            sampler=recording_sampler(first_records),
            state_constructor=interrupt_after_proposal,
            target_evaluator=_identity_target,
        )
        with self.assertRaisesRegex(
            InjectedInterruption,
            "after durable proposal",
        ):
            run_accepted_quotas(spec, interrupted)

        pending = scan_quota_run(spec).pending
        self.assertIsNotNone(pending)
        assert pending is not None
        self.assertEqual(pending.status, BatchStatus.PROPOSED)
        proposal_hash = file_sha256(pending.paths.proposal)

        resumed = StaticStokesQuotaExecutor(
            run_spec=spec,
            contract=contract,
            maximum_ursell_redraws=0,
            sampler=recording_sampler(replay_records),
            state_constructor=_zero_state,
            target_evaluator=_identity_target,
        )
        state = run_accepted_quotas(spec, resumed)
        self.assertTrue(state.complete)
        self.assertEqual(first_records, replay_records)
        self.assertEqual(file_sha256(pending.paths.proposal), proposal_hash)
        self.assertEqual(
            inspect_batch(
                pending.paths,
                expected_fingerprint=spec.config_fingerprint,
            ).status,
            BatchStatus.COMMITTED,
        )

    def test_executor_requires_contract_and_sampler_to_match_run_record(
        self,
    ) -> None:
        contract = _contract()
        deep_cell = STOKES_SAMPLE_CELLS[2].cell_id
        wrong_contract_spec = AcceptedQuotaRunSpec(
            root=self.root / "contract",
            family_name="stokes",
            family_id=PhysicalFamilyId.STOKES,
            revision_id=1,
            split_id=SplitId.TEST,
            stream_id=11,
            quotas=(CellQuota(deep_cell, 1),),
            cell_codes={deep_cell: 0},
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
            StaticStokesQuotaExecutor(
                run_spec=wrong_contract_spec,
                contract=contract,
                maximum_ursell_redraws=0,
            )

        wrong_sampler_spec = _run_spec(
            self.root / "sampler",
            contract,
            cell_ids=(deep_cell,),
            targets=(1,),
            maximum_ursell_redraws=1,
            batch_size=1,
        )
        with self.assertRaisesRegex(ValueError, "redraw limit differs"):
            StaticStokesQuotaExecutor(
                run_spec=wrong_sampler_spec,
                contract=contract,
                maximum_ursell_redraws=0,
            )

    def test_paper_role_rejects_injected_implementation_hooks(self) -> None:
        contract = PAPER_STATIC_STOKES_CONTRACT
        deep_cell = STOKES_SAMPLE_CELLS[2].cell_id
        spec = _run_spec(
            self.root,
            contract,
            cell_ids=(deep_cell,),
            targets=(1,),
            maximum_ursell_redraws=DEFAULT_MAXIMUM_URSELL_REDRAWS,
            batch_size=1,
        )
        StaticStokesQuotaExecutor(
            run_spec=spec,
            contract=contract,
        )
        with self.assertRaisesRegex(ValueError, "paper-dataset execution"):
            StaticStokesQuotaExecutor(
                run_spec=spec,
                contract=contract,
                state_constructor=_zero_state,
            )
        with self.assertRaisesRegex(ValueError, "paper-dataset execution"):
            StaticStokesQuotaExecutor(
                run_spec=spec,
                contract=contract,
                target_evaluator=_identity_target,
            )
        zero_redraw_spec = _run_spec(
            self.root / "zero_redraw",
            contract,
            cell_ids=(deep_cell,),
            targets=(1,),
            maximum_ursell_redraws=0,
            batch_size=1,
        )
        with self.assertRaisesRegex(ValueError, "paper-dataset execution"):
            StaticStokesQuotaExecutor(
                run_spec=zero_redraw_spec,
                contract=contract,
                maximum_ursell_redraws=0,
            )


if __name__ == "__main__":
    unittest.main()

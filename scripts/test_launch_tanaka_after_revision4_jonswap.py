"""Isolated integration tests for the sequential corpus supervisor."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[1]
SUPERVISOR = ROOT / "scripts/launch_tanaka_after_revision4_jonswap.sh"
TANAKA_LAUNCHER = ROOT / "scripts/launch_revision3_tanaka_bulk.sh"
GPU_ADMISSION = ROOT / "scripts/check_tanaka_gpu_admission.sh"
EXPECTED_CASES = {"train": 16_384, "validation": 1_024, "test": 1_024}


class SequentialCorpusSupervisorTest(unittest.TestCase):
    temporary: tempfile.TemporaryDirectory[str]
    mock_root: Path
    event_log: Path

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.mock_root = Path(self.temporary.name)
        self.event_log = self.mock_root / "events.log"
        (self.mock_root / "jon").mkdir()
        (self.mock_root / "tan").mkdir()
        self._write_json(
            self.mock_root / "jon_gate.json",
            self._gate_record("jonswap_tma"),
        )
        self._write_json(
            self.mock_root / "tan_gate.json",
            self._gate_record("tanaka"),
        )
        self._write_fake_commands()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _gate_record(family: str) -> dict[str, object]:
        return {
            "status": "passed",
            "family": family,
            "accepted_cases": EXPECTED_CASES,
        }

    @staticmethod
    def _write_json(path: Path, record: dict[str, object]) -> None:
        path.write_text(json.dumps(record), encoding="utf-8")

    @staticmethod
    def _write_executable(path: Path, source: str) -> None:
        path.write_text(textwrap.dedent(source).lstrip(), encoding="utf-8")
        path.chmod(0o755)

    def _write_fake_commands(self) -> None:
        self._write_executable(
            self.mock_root / "tmux",
            r"""
            #!/usr/bin/env bash
            set -euo pipefail
            if [[ "${TMUX_MODE:-absent}" == "present_once" \
                    && ! -f "$MOCK_ROOT/tmux_seen" ]]; then
                touch "$MOCK_ROOT/tmux_seen"
                exit 0
            fi
            exit 1
            """,
        )
        self._write_executable(
            self.mock_root / "sleep",
            r"""
            #!/usr/bin/env bash
            set -euo pipefail
            if [[ "${GPU_SLEEP_MODE:-normal}" == "fail" ]]; then
                exit 73
            fi
            if [[ "${GPU_SMI_MODE:-clean}" == "eventual_clear" ]]; then
                touch "$MOCK_ROOT/gpu_clear"
            fi
            exit 0
            """,
        )
        self._write_executable(
            self.mock_root / "nvidia-smi",
            r"""
            #!/usr/bin/env bash
            set -euo pipefail
            gpu=""
            query=""
            while (($#)); do
                case "$1" in
                    -i)
                        gpu=$2
                        shift 2
                        ;;
                    --query-gpu=*)
                        query=gpu
                        shift
                        ;;
                    --query-compute-apps=*)
                        query=apps
                        shift
                        ;;
                    *)
                        shift
                        ;;
                esac
            done
            printf 'query:%s:%s\n' "$query" "$gpu" \
                >>"$MOCK_ROOT/nvidia_queries.log"
            mode=${GPU_SMI_MODE:-clean}
            if [[ "$mode" == "error" ]]; then
                exit 9
            fi
            if [[ "$mode" == "eventual_clear" \
                    && -f "$MOCK_ROOT/gpu_clear" ]]; then
                mode=clean
            fi
            if [[ "$mode" == "state_change" ]]; then
                if [[ -f "$MOCK_ROOT/direct_preflight_seen" ]]; then
                    mode=occupied
                else
                    mode=clean
                fi
            fi
            if [[ "$mode" == "state_change_malformed" ]]; then
                if [[ -f "$MOCK_ROOT/direct_preflight_seen" ]]; then
                    mode=malformed_process
                else
                    mode=clean
                fi
            fi
            if [[ "$query" == gpu ]]; then
                case "$mode" in
                    malformed_index)
                        printf '9, 49140, 49140\n'
                        ;;
                    malformed_memory)
                        printf '%s, N/A, 49140\n' "$gpu"
                        ;;
                    duplicate_rows)
                        printf '%s, 49140, 49140\n' "$gpu"
                        printf '%s, 49140, 49140\n' "$gpu"
                        ;;
                    trailing_field)
                        printf '%s, 49140, 49140,\n' "$gpu"
                        ;;
                    low_memory)
                        printf '%s, 40000, 49140\n' "$gpu"
                        ;;
                    *)
                        printf '%s, 49140, 49140\n' "$gpu"
                        ;;
                esac
            elif [[ "$query" == apps ]]; then
                case "$mode" in
                    occupied|eventual_clear)
                        [[ "$gpu" != 1 ]] || printf '424242\n'
                        ;;
                    malformed_process)
                        printf 'N/A\n'
                        ;;
                esac
            else
                exit 10
            fi
            """,
        )
        self._write_executable(
            self.mock_root / "direct_python",
            r"""
            #!/usr/bin/env python3
            import json
            import os
            from pathlib import Path
            import sys

            root = Path(os.environ["MOCK_ROOT"])
            with (root / "direct_python_calls.log").open("a") as stream:
                stream.write(" ".join(sys.argv[1:]) + "\n")
            reserved_exit_at = os.environ.get("DIRECT_PYTHON_RESERVED_EXIT_AT", "")
            reserved_exit_status = int(
                os.environ.get("DIRECT_PYTHON_RESERVED_EXIT_STATUS", "75")
            )
            if "-c" in sys.argv:
                inline_source = sys.argv[sys.argv.index("-c") + 1]
                stage_markers = {
                    "preflight_identity": (
                        "Tanaka preflights do not share one frozen identity"
                    ),
                    "summary": "completion differs from its frozen preflight",
                    "final_identity": (
                        "completed Tanaka chunks do not share one identity"
                    ),
                }
                marker = stage_markers.get(reserved_exit_at)
                if marker is not None and marker in inline_source:
                    raise SystemExit(reserved_exit_status)
            if reserved_exit_at == "dry_run" and "--dry-run" in sys.argv:
                raise SystemExit(reserved_exit_status)
            if reserved_exit_at == "execute" and "--execute" in sys.argv:
                raise SystemExit(reserved_exit_status)
            if "--dry-run" in sys.argv:
                (root / "direct_preflight_seen").touch()
                execution = {"mock": "execution"}
                print(json.dumps({
                    "schema": "paper_corpus_quota_preflight_v1",
                    "no_numerical_generation_performed": True,
                    "run_spec": {
                        "family_name": "tanaka",
                        "revision_id": 3,
                        "batch_size": 256,
                        "maximum_attempts_per_accepted_case": 4,
                        "configuration": {
                            "execution_platform": "gpu",
                            "trajectory_execution": execution,
                            "dependency_environment": {},
                            "source_sha256": {},
                            "ordered_cell_ids": [],
                        },
                    },
                    "execution": execution,
                }))
            """,
        )
        self._write_executable(
            self.mock_root / "launch_real_tanaka_reserved_dry_run.sh",
            r"""
            #!/usr/bin/env bash
            set -euo pipefail
            exec env \
                PYTHON="$MOCK_ROOT/direct_python" \
                DIRECT_PYTHON_RESERVED_EXIT_AT=dry_run \
                /bin/bash "$REAL_TANAKA_LAUNCHER"
            """,
        )
        self._write_launcher("jon", "JON_LAUNCH_MODE")
        self._write_launcher("tan", "TAN_LAUNCH_MODE")
        self._write_executable(
            self.mock_root / "audit.py",
            r"""
            #!/usr/bin/env python3
            import json
            import os
            from pathlib import Path
            import signal
            import sys
            import time

            if (os.environ.get("CUDA_VISIBLE_DEVICES") != ""
                    or os.environ.get("JAX_PLATFORMS") != "cpu"
                    or os.environ.get("JAX_ENABLE_X64") != "true"):
                raise SystemExit("audit invocation was not CPU-only fp64")
            root = Path(sys.argv[sys.argv.index("--root") + 1]).resolve()
            output = Path(sys.argv[sys.argv.index("--output") + 1])
            family = root.name
            with (Path(os.environ["MOCK_ROOT"]) / "events.log").open("a") as stream:
                stream.write(f"audit:{family}\n")
            if os.environ.get("HANG_AUDIT_FAMILY") == family:
                signal.signal(signal.SIGTERM, signal.SIG_IGN)
                while True:
                    time.sleep(60)
            is_jonswap = family == "jon"
            rows = 294_912 if is_jonswap else 3_686_400
            if os.environ.get("AUDIT_MISMATCH_FAMILY") == family:
                rows += 1
            record = {
                "schema": (
                    "paper_corpus_jonswap_tma_revision4_completion_audit_v1"
                    if is_jonswap
                    else "paper_corpus_tanaka_revision3_completion_audit_v1"
                ),
                "status": "pass",
                "corpus_root": str(root),
                "accepted": 18_432,
                "attempted": 18_435,
                "rejected": 3,
                "retained_rows": rows,
                "accepted_by_split": {
                    "train": 16_384,
                    "validation": 1_024,
                    "test": 1_024,
                },
            }
            output.write_text(json.dumps(record), encoding="utf-8")
            """,
        )
        self._write_executable(
            self.mock_root / "build.sh",
            r"""
            #!/usr/bin/env bash
            set -euo pipefail
            [[ -f "$MOCK_ROOT/jon/jonswap_tma_completion_audit.json" ]]
            [[ -f "$MOCK_ROOT/tan/tanaka_completion_audit.json" ]]
            printf 'build\n' >>"$MOCK_ROOT/events.log"
            if [[ "${HANG_BUILD:-false}" == "true" ]]; then
                trap '' TERM
                while :; do
                    /bin/sleep 60
                done
            fi
            """,
        )

    def _write_launcher(self, family: str, mode_variable: str) -> None:
        gate_name = (
            "jonswap_view_builder_input_check.json"
            if family == "jon"
            else "tanaka_view_builder_input_check.json"
        )
        template = r"""
            #!/usr/bin/env bash
            set -euo pipefail
            count_file="$MOCK_ROOT/@FAMILY@_launch_count"
            count=0
            if [[ -f "$count_file" ]]; then
                read -r count <"$count_file"
            fi
            count=$((count + 1))
            printf '%s\n' "$count" >"$count_file"
            printf 'launch:@FAMILY@:%s\n' "$count" >>"$MOCK_ROOT/events.log"
            mode="${@MODE_VARIABLE@:-success}"
            case "$mode" in
                always_fail)
                    exit 7
                    ;;
                fail_once)
                    ((count > 1)) || exit 7
                    ;;
                must_not_run)
                    exit 91
                    ;;
                temporary_once)
                    ((count > 1)) || exit 75
                    ;;
                telemetry_error)
                    exit 70
                    ;;
                success)
                    ;;
                *)
                    exit 92
                    ;;
            esac
            cp "$MOCK_ROOT/@FAMILY@_gate.json" "$OUTPUT_BASE/@GATE_NAME@"
        """
        source = (
            template.replace("@FAMILY@", family)
            .replace("@MODE_VARIABLE@", mode_variable)
            .replace("@GATE_NAME@", gate_name)
        )
        self._write_executable(self.mock_root / f"launch_{family}.sh", source)

    def _environment(
        self,
        *,
        maximum_retries: int = 3,
        tmux_mode: str = "absent",
        jonswap_launch_mode: str = "success",
        tanaka_launch_mode: str = "success",
        audit_mismatch_family: str = "",
        hang_audit_family: str = "",
        hang_build: bool = False,
        audit_timeout_seconds: int = 3_600,
        build_timeout_seconds: int = 1_800,
        timeout_kill_after_seconds: int = 120,
        gpu_smi_mode: str = "clean",
        gpu_sleep_mode: str = "normal",
        nvidia_smi_bin: Path | None = None,
        minimum_free_mib: int = 40_960,
        tanaka_launch_script: Path | None = None,
    ) -> dict[str, str]:
        environment = os.environ.copy()
        environment.update(
            {
                "MOCK_ROOT": str(self.mock_root),
                "PYTHON": sys.executable,
                "BASH_BIN": "/bin/bash",
                "TMUX_BIN": str(self.mock_root / "tmux"),
                "SLEEP_BIN": str(self.mock_root / "sleep"),
                "TIMEOUT_BIN": "/usr/bin/timeout",
                "WAIT_SESSION": "mock_jonswap",
                "WAIT_INTERVAL_SECONDS": "0",
                "RETRY_DELAY_SECONDS": "0",
                "MAX_RETRIES": str(maximum_retries),
                "AUDIT_TIMEOUT_SECONDS": str(audit_timeout_seconds),
                "BUILD_TIMEOUT_SECONDS": str(build_timeout_seconds),
                "TIMEOUT_KILL_AFTER_SECONDS": str(timeout_kill_after_seconds),
                "TANAKA_GPU_WAIT_INTERVAL_SECONDS": "0",
                "GPU_ZERO": "0",
                "GPU_ONE": "1",
                "GPU_ADMISSION_SCRIPT": str(GPU_ADMISSION),
                "NVIDIA_SMI_BIN": str(nvidia_smi_bin or self.mock_root / "nvidia-smi"),
                "TANAKA_GPU_MINIMUM_FREE_MIB": str(minimum_free_mib),
                "JONSWAP_ROOT": str(self.mock_root / "jon"),
                "TANAKA_ROOT": str(self.mock_root / "tan"),
                "JONSWAP_COMPLETION_GATE": str(
                    self.mock_root / "jon/jonswap_view_builder_input_check.json"
                ),
                "TANAKA_COMPLETION_GATE": str(
                    self.mock_root / "tan/tanaka_view_builder_input_check.json"
                ),
                "JONSWAP_AUDIT_ARTIFACT": str(
                    self.mock_root / "jon/jonswap_tma_completion_audit.json"
                ),
                "TANAKA_AUDIT_ARTIFACT": str(
                    self.mock_root / "tan/tanaka_completion_audit.json"
                ),
                "JONSWAP_LAUNCH_SCRIPT": str(self.mock_root / "launch_jon.sh"),
                "TANAKA_LAUNCH_SCRIPT": str(self.mock_root / "launch_tan.sh"),
                "REAL_TANAKA_LAUNCHER": str(TANAKA_LAUNCHER),
                "JONSWAP_AUDIT_SCRIPT": str(self.mock_root / "audit.py"),
                "TANAKA_AUDIT_SCRIPT": str(self.mock_root / "audit.py"),
                "BUILD_SCRIPT": str(self.mock_root / "build.sh"),
                "SUPERVISOR_LOG": str(self.mock_root / "supervisor.log"),
                "TMUX_MODE": tmux_mode,
                "JON_LAUNCH_MODE": jonswap_launch_mode,
                "TAN_LAUNCH_MODE": tanaka_launch_mode,
                "AUDIT_MISMATCH_FAMILY": audit_mismatch_family,
                "HANG_AUDIT_FAMILY": hang_audit_family,
                "HANG_BUILD": "true" if hang_build else "false",
                "GPU_SMI_MODE": gpu_smi_mode,
                "GPU_SLEEP_MODE": gpu_sleep_mode,
            }
        )
        if tanaka_launch_script is not None:
            environment["TANAKA_LAUNCH_SCRIPT"] = str(tanaka_launch_script)
        return environment

    def _run(
        self,
        *,
        maximum_retries: int = 3,
        tmux_mode: str = "absent",
        jonswap_launch_mode: str = "success",
        tanaka_launch_mode: str = "success",
        audit_mismatch_family: str = "",
        hang_audit_family: str = "",
        hang_build: bool = False,
        audit_timeout_seconds: int = 3_600,
        build_timeout_seconds: int = 1_800,
        timeout_kill_after_seconds: int = 120,
        gpu_smi_mode: str = "clean",
        gpu_sleep_mode: str = "normal",
        nvidia_smi_bin: Path | None = None,
        minimum_free_mib: int = 40_960,
        tanaka_launch_script: Path | None = None,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["/bin/bash", str(SUPERVISOR)],
            cwd=ROOT,
            env=self._environment(
                maximum_retries=maximum_retries,
                tmux_mode=tmux_mode,
                jonswap_launch_mode=jonswap_launch_mode,
                tanaka_launch_mode=tanaka_launch_mode,
                audit_mismatch_family=audit_mismatch_family,
                hang_audit_family=hang_audit_family,
                hang_build=hang_build,
                audit_timeout_seconds=audit_timeout_seconds,
                build_timeout_seconds=build_timeout_seconds,
                timeout_kill_after_seconds=timeout_kill_after_seconds,
                gpu_smi_mode=gpu_smi_mode,
                gpu_sleep_mode=gpu_sleep_mode,
                nvidia_smi_bin=nvidia_smi_bin,
                minimum_free_mib=minimum_free_mib,
                tanaka_launch_script=tanaka_launch_script,
            ),
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )

    def _run_direct_launcher(
        self,
        *,
        gpu_smi_mode: str,
        python: Path | None = None,
        nvidia_smi_bin: Path | None = None,
        gpu_zero: str = "0",
        gpu_one: str = "1",
        reserved_exit_at: str = "",
        reserved_exit_status: int = 75,
    ) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment.update(
            {
                "MOCK_ROOT": str(self.mock_root),
                "PYTHON": str(python or sys.executable),
                "OUTPUT_BASE": str(self.mock_root / "direct_output"),
                "GPU_ZERO": gpu_zero,
                "GPU_ONE": gpu_one,
                "GPU_ADMISSION_SCRIPT": str(GPU_ADMISSION),
                "NVIDIA_SMI_BIN": str(nvidia_smi_bin or self.mock_root / "nvidia-smi"),
                "GPU_SMI_MODE": gpu_smi_mode,
                "DIRECT_PYTHON_RESERVED_EXIT_AT": reserved_exit_at,
                "DIRECT_PYTHON_RESERVED_EXIT_STATUS": str(reserved_exit_status),
            }
        )
        return subprocess.run(
            ["/bin/bash", str(TANAKA_LAUNCHER)],
            cwd=ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )

    def _install_gate(self, family: str) -> None:
        name = (
            "jonswap_view_builder_input_check.json"
            if family == "jon"
            else "tanaka_view_builder_input_check.json"
        )
        source = self.mock_root / f"{family}_gate.json"
        (self.mock_root / family / name).write_bytes(source.read_bytes())

    def _events(self) -> list[str]:
        if not self.event_log.exists():
            return []
        return self.event_log.read_text(encoding="utf-8").splitlines()

    def _nvidia_queries(self) -> list[str]:
        path = self.mock_root / "nvidia_queries.log"
        if not path.exists():
            return []
        return path.read_text(encoding="utf-8").splitlines()

    def test_existing_gates_skip_relaunch_but_still_run_fresh_audits(self) -> None:
        self._install_gate("jon")
        self._install_gate("tan")

        result = self._run(
            tmux_mode="present_once",
            jonswap_launch_mode="must_not_run",
            tanaka_launch_mode="must_not_run",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self._events(), ["audit:jon", "audit:tan", "build"])
        self.assertIn("lightweight completion gate already passes", result.stderr)

    def test_existing_tanaka_gate_does_not_wait_for_occupied_gpus(self) -> None:
        self._install_gate("jon")
        self._install_gate("tan")

        result = self._run(
            gpu_smi_mode="occupied",
            jonswap_launch_mode="must_not_run",
            tanaka_launch_mode="must_not_run",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self._events(), ["audit:jon", "audit:tan", "build"])
        self.assertEqual(self._nvidia_queries(), [])

    def test_occupied_gpu_waits_without_launching_or_signaling(self) -> None:
        self._install_gate("jon")

        result = self._run(
            gpu_smi_mode="occupied",
            gpu_sleep_mode="fail",
            jonswap_launch_mode="must_not_run",
            tanaka_launch_mode="must_not_run",
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self._events(), ["audit:jon"])
        self.assertIn("compute_pids=424242", result.stderr)
        self.assertIn("foreign processes are never signaled", result.stderr)

    def test_low_free_memory_waits_without_launching(self) -> None:
        self._install_gate("jon")

        result = self._run(
            gpu_smi_mode="low_memory",
            gpu_sleep_mode="fail",
            jonswap_launch_mode="must_not_run",
            tanaka_launch_mode="must_not_run",
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self._events(), ["audit:jon"])
        self.assertIn("free=40000/49140 MiB", result.stderr)
        self.assertIn("required_free=40960 MiB", result.stderr)

    def test_gpu_wait_admits_after_foreign_process_clears(self) -> None:
        self._install_gate("jon")

        result = self._run(
            gpu_smi_mode="eventual_clear",
            jonswap_launch_mode="must_not_run",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            self._events(),
            ["audit:jon", "launch:tan:1", "audit:tan", "build"],
        )
        self.assertIn("GPU admission WAIT", result.stderr)
        self.assertIn("GPU admission PASS", result.stderr)

    def test_higher_memory_floor_reports_its_actual_margin(self) -> None:
        self._install_gate("jon")

        result = self._run(
            minimum_free_mib=45_000,
            jonswap_launch_mode="must_not_run",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("required_free=45000 MiB", result.stderr)
        self.assertIn("margin=9975 MiB", result.stderr)

    def test_malformed_gpu_memory_fails_closed_before_tanaka(self) -> None:
        self._install_gate("jon")

        result = self._run(
            gpu_smi_mode="malformed_memory",
            jonswap_launch_mode="must_not_run",
            tanaka_launch_mode="must_not_run",
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self._events(), ["audit:jon"])
        self.assertIn("inventory contains malformed fields", result.stderr)

    def test_nvidia_smi_error_fails_closed_before_tanaka(self) -> None:
        self._install_gate("jon")

        result = self._run(
            gpu_smi_mode="error",
            jonswap_launch_mode="must_not_run",
            tanaka_launch_mode="must_not_run",
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self._events(), ["audit:jon"])
        self.assertIn("inventory query exited 9", result.stderr)

    def test_missing_nvidia_smi_fails_closed_before_tanaka(self) -> None:
        self._install_gate("jon")

        result = self._run(
            nvidia_smi_bin=self.mock_root / "missing-nvidia-smi",
            jonswap_launch_mode="must_not_run",
            tanaka_launch_mode="must_not_run",
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self._events(), ["audit:jon"])
        self.assertIn("nvidia-smi command is unavailable", result.stderr)

    def test_temporary_launcher_unavailability_does_not_consume_retry(self) -> None:
        self._install_gate("jon")

        result = self._run(
            maximum_retries=0,
            jonswap_launch_mode="must_not_run",
            tanaka_launch_mode="temporary_once",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            self._events(),
            [
                "audit:jon",
                "launch:tan:1",
                "launch:tan:2",
                "audit:tan",
                "build",
            ],
        )
        self.assertIn("without consuming retry budget", result.stderr)

    def test_launcher_telemetry_error_is_not_retried(self) -> None:
        self._install_gate("jon")

        result = self._run(
            maximum_retries=3,
            jonswap_launch_mode="must_not_run",
            tanaka_launch_mode="telemetry_error",
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self._events(), ["audit:jon", "launch:tan:1"])
        self.assertIn("rejected invalid GPU telemetry", result.stderr)

    def test_direct_launcher_rejects_occupied_gpu_before_output(self) -> None:
        result = self._run_direct_launcher(gpu_smi_mode="occupied")

        self.assertEqual(result.returncode, 75, result.stderr)
        self.assertFalse((self.mock_root / "direct_output").exists())
        self.assertIn("compute_pids=424242", result.stderr)

    def test_direct_launcher_rejects_low_memory_before_output(self) -> None:
        result = self._run_direct_launcher(gpu_smi_mode="low_memory")

        self.assertEqual(result.returncode, 75, result.stderr)
        self.assertFalse((self.mock_root / "direct_output").exists())
        self.assertIn("free=40000/49140 MiB", result.stderr)

    def test_direct_launcher_rechecks_after_preflights_before_lane_fork(self) -> None:
        result = self._run_direct_launcher(
            gpu_smi_mode="state_change",
            python=self.mock_root / "direct_python",
        )

        self.assertEqual(result.returncode, 75, result.stderr)
        calls = (self.mock_root / "direct_python_calls.log").read_text(encoding="utf-8")
        self.assertEqual(calls.count("--dry-run"), 6)
        self.assertNotIn("--execute", calls)
        self.assertIn("GPU admission PASS (pre-preflight)", result.stderr)
        self.assertIn("GPU admission WAIT (pre-lane-fork)", result.stderr)

    def test_direct_launcher_malformed_telemetry_uses_fatal_status(self) -> None:
        result = self._run_direct_launcher(gpu_smi_mode="malformed_process")

        self.assertEqual(result.returncode, 70, result.stderr)
        self.assertFalse((self.mock_root / "direct_output").exists())
        self.assertIn("compute-process row is malformed", result.stderr)

    def test_direct_launcher_remaps_reserved_dry_run_failure(self) -> None:
        result = self._run_direct_launcher(
            gpu_smi_mode="clean",
            python=self.mock_root / "direct_python",
            reserved_exit_at="dry_run",
        )

        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("reserved GPU-admission exit 75", result.stderr)
        self.assertIn("remapping to ordinary failure", result.stderr)
        calls = (self.mock_root / "direct_python_calls.log").read_text(encoding="utf-8")
        self.assertEqual(calls.count("--dry-run"), 1)
        self.assertNotIn("--execute", calls)

    def test_direct_launcher_remaps_reserved_telemetry_code_from_dry_run(
        self,
    ) -> None:
        result = self._run_direct_launcher(
            gpu_smi_mode="clean",
            python=self.mock_root / "direct_python",
            reserved_exit_at="dry_run",
            reserved_exit_status=70,
        )

        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("reserved GPU-admission exit 70", result.stderr)
        self.assertIn("remapping to ordinary failure", result.stderr)

    def test_direct_launcher_remaps_reserved_execute_failure(self) -> None:
        result = self._run_direct_launcher(
            gpu_smi_mode="clean",
            python=self.mock_root / "direct_python",
            reserved_exit_at="execute",
        )

        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("reserved GPU-admission exit 75", result.stderr)
        self.assertIn("remapping to ordinary failure", result.stderr)
        calls = (self.mock_root / "direct_python_calls.log").read_text(encoding="utf-8")
        self.assertEqual(calls.count("--dry-run"), 6)
        self.assertIn("--execute", calls)

    def test_direct_launcher_remaps_reserved_codes_from_every_late_stage(
        self,
    ) -> None:
        for stage in ("execute", "preflight_identity", "summary", "final_identity"):
            for status in (70, 75):
                with self.subTest(stage=stage, status=status):
                    result = self._run_direct_launcher(
                        gpu_smi_mode="clean",
                        python=self.mock_root / "direct_python",
                        reserved_exit_at=stage,
                        reserved_exit_status=status,
                    )

                    self.assertEqual(result.returncode, 1, result.stderr)
                    self.assertIn(
                        f"reserved GPU-admission exit {status}", result.stderr
                    )
                    self.assertIn("remapping to ordinary failure", result.stderr)

    def test_direct_launcher_preserves_terminal_telemetry_failure(self) -> None:
        result = self._run_direct_launcher(
            gpu_smi_mode="state_change_malformed",
            python=self.mock_root / "direct_python",
        )

        self.assertEqual(result.returncode, 70, result.stderr)
        self.assertIn("GPU admission PASS (pre-preflight)", result.stderr)
        self.assertIn("compute-process row is malformed", result.stderr)
        self.assertNotIn("remapping to ordinary failure", result.stderr)
        calls = (self.mock_root / "direct_python_calls.log").read_text(encoding="utf-8")
        self.assertEqual(calls.count("--dry-run"), 6)
        self.assertNotIn("--execute", calls)

    def test_non_admission_reserved_failure_consumes_supervisor_retry(self) -> None:
        self._install_gate("jon")

        result = self._run(
            maximum_retries=1,
            jonswap_launch_mode="must_not_run",
            tanaka_launch_script=(
                self.mock_root / "launch_real_tanaka_reserved_dry_run.sh"
            ),
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self._events(), ["audit:jon"])
        self.assertIn("Tanaka launch 1 exited 1", result.stderr)
        self.assertIn("Tanaka exhausted 1 retries", result.stderr)
        self.assertNotIn("without consuming retry budget", result.stderr)
        calls = (self.mock_root / "direct_python_calls.log").read_text(encoding="utf-8")
        self.assertEqual(calls.count("--dry-run"), 2)

    def test_inventory_with_trailing_empty_field_is_malformed(self) -> None:
        result = self._run_direct_launcher(gpu_smi_mode="trailing_field")

        self.assertEqual(result.returncode, 70, result.stderr)
        self.assertIn("must contain exactly three fields", result.stderr)

    def test_duplicate_physical_gpu_indices_are_rejected(self) -> None:
        result = self._run_direct_launcher(
            gpu_smi_mode="clean", gpu_zero="0", gpu_one="0"
        )

        self.assertEqual(result.returncode, 70, result.stderr)
        self.assertEqual(self._nvidia_queries(), [])
        self.assertIn("must be distinct canonical decimals", result.stderr)

    def test_gpu_admission_helper_is_source_only(self) -> None:
        result = subprocess.run(
            ["/bin/bash", str(GPU_ADMISSION)],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )

        self.assertEqual(result.returncode, 70, result.stderr)
        self.assertIn("source-only helper", result.stderr)

    def test_missing_gate_retries_exact_resume_then_succeeds(self) -> None:
        result = self._run(jonswap_launch_mode="fail_once")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            self._events(),
            [
                "launch:jon:1",
                "launch:jon:2",
                "audit:jon",
                "launch:tan:1",
                "audit:tan",
                "build",
            ],
        )
        self.assertIn("launch 1 exited 7 without a valid gate", result.stderr)

    def test_exhausted_retries_fail_before_audit_or_build(self) -> None:
        result = self._run(
            maximum_retries=2,
            jonswap_launch_mode="always_fail",
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(
            self._events(),
            ["launch:jon:1", "launch:jon:2", "launch:jon:3"],
        )
        self.assertIn("exhausted 2 retries", result.stderr)

    def test_mismatched_audit_artifact_fails_closed(self) -> None:
        self._install_gate("jon")
        self._install_gate("tan")

        result = self._run(audit_mismatch_family="jon")

        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self._events(), ["audit:jon"])
        self.assertIn("audit artifact did not pass validation", result.stderr)

    def test_successful_chain_builds_only_after_both_audits(self) -> None:
        result = self._run()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            self._events(),
            [
                "launch:jon:1",
                "audit:jon",
                "launch:tan:1",
                "audit:tan",
                "build",
            ],
        )

    def test_hanging_audit_times_out_and_fails_before_build(self) -> None:
        self._install_gate("jon")
        self._install_gate("tan")

        result = self._run(
            hang_audit_family="jon",
            audit_timeout_seconds=1,
            timeout_kill_after_seconds=1,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self._events(), ["audit:jon"])
        self.assertIn("CPU completion audit exceeded 1s", result.stderr)

    def test_hanging_builder_times_out_after_both_audits(self) -> None:
        self._install_gate("jon")
        self._install_gate("tan")

        result = self._run(
            hang_build=True,
            build_timeout_seconds=1,
            timeout_kill_after_seconds=1,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self._events(), ["audit:jon", "audit:tan", "build"])
        self.assertIn("final view build exceeded 1s", result.stderr)


if __name__ == "__main__":
    unittest.main()

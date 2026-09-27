"""Check ordered CPU prefetch against bulk DataLoader, including RNG and tails."""

import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "torch-attention"))
from train import Waves, cpu_batches, make_loader  # noqa: E402


def main() -> None:
    torch.set_num_threads(1)
    results = []
    for rows, workers, explicit_generator in ((513, 4, True), (1025, 16, True), (513, 4, False)):
        data = Waves(ROOT.parent / "local-data/paper_equal_subset_20260924", "train")
        data.rows = data.rows[:rows]
        runs = []
        for parallel in (False, True):
            torch.manual_seed(123)
            generator = torch.Generator().manual_seed(7) if explicit_generator else None
            loader = make_loader(data, 256, shuffle=True, generator=generator, bulk=True)
            epochs = [list(cpu_batches(loader, workers if parallel else 0)) for _ in range(2)]
            rng = generator.get_state() if generator is not None else torch.get_rng_state()
            runs.append((epochs, rng))
        for serial_epoch, parallel_epoch in zip(runs[0][0], runs[1][0], strict=True):
            assert sum(len(batch[0]) for batch in parallel_epoch) == rows
            assert len(parallel_epoch[-1][0]) == rows % 256
            for serial, parallel in zip(serial_epoch, parallel_epoch, strict=True):
                for a, b in zip(serial, parallel, strict=True):
                    torch.testing.assert_close(a, b, atol=0, rtol=0)
        assert torch.equal(runs[0][1], runs[1][1])
        results.append({"rows": rows, "workers": workers, "explicit_generator": explicit_generator,
                        "epochs": 2, "all_tensors_exact": True, "rng_state_exact": True,
                        "partial_batch_preserved": True})
    Path(__file__).with_suffix(".json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results))


if __name__ == "__main__":
    main()

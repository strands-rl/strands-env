import json

from strands_env.eval.cli import eval_cmd

STUB = "tests.unit.eval.stub_hooks"


class TestRunOutputs:
    def test_writes_the_three_run_files(self, runner, tmp_path):
        """The CLI owns `metadata.json` and `metrics.json`; the evaluator streams `results.jsonl`."""
        run_dir = tmp_path / "run"

        result = runner.invoke(eval_cmd, ["--evaluator", STUB, "--env", STUB, "-o", str(run_dir)])

        assert result.exit_code == 0, result.output
        assert json.loads((run_dir / "metadata.json").read_text())["benchmark"] == "stub"
        assert json.loads((run_dir / "metrics.json").read_text()) == {"pass@1": 1.0}
        rows = [json.loads(line) for line in (run_dir / "results.jsonl").read_text().splitlines()]
        assert [(r["task"]["id"], r["result"]["reward_result"]["reward"]) for r in rows] == [("p1_0", 1.0)]

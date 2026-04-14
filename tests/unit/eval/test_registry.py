# Copyright 2025-2026 Strands RL Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for benchmark registry."""

import pytest

from strands_env.cli import cli
from strands_env.eval import Evaluator, get_benchmark, list_benchmarks
from strands_env.eval.registry import _BENCHMARKS
from strands_env.eval.registry import register_eval as register_benchmark


class TestBenchmarkRegistry:
    def test_aime_registered(self):
        """AIME benchmarks are registered by default."""
        benchmarks = list_benchmarks()
        assert "aime-2024" in benchmarks
        assert "aime-2025" in benchmarks

    def test_get_benchmark_returns_class(self):
        """get_benchmark returns the evaluator class."""
        cls = get_benchmark("aime-2024")
        assert issubclass(cls, Evaluator)

    def test_get_benchmark_unknown_raises(self):
        """get_benchmark raises KeyError for unknown benchmark."""
        with pytest.raises(KeyError, match="Unknown benchmark 'nonexistent'"):
            get_benchmark("nonexistent")

    def test_register_decorator(self):
        """register decorator adds benchmark to registry."""
        name = "_test_benchmark_decorator"

        @register_benchmark(name)
        class TestEvaluator(Evaluator):
            pass

        try:
            assert name in list_benchmarks()
            assert get_benchmark(name) is TestEvaluator
        finally:
            _BENCHMARKS.pop(name, None)

    def test_register_duplicate_raises(self):
        """Registering same name twice raises ValueError."""
        name = "_test_duplicate_benchmark"

        @register_benchmark(name)
        class First(Evaluator):
            pass

        try:
            with pytest.raises(ValueError, match=f"Benchmark '{name}' is already registered"):

                @register_benchmark(name)
                class Second(Evaluator):
                    pass
        finally:
            _BENCHMARKS.pop(name, None)


class TestListCommand:
    def test_list_benchmarks(self, runner):
        """List command shows registered benchmarks."""
        result = runner.invoke(cli, ["eval", "list"])
        assert result.exit_code == 0
        assert "Benchmarks:" in result.output
        assert "aime-2024" in result.output
        assert "aime-2025" in result.output


class TestEvalCommand:
    def test_eval_requires_env(self, runner):
        """Eval run command requires --env option."""
        result = runner.invoke(cli, ["eval", "run", "aime-2024"])
        assert result.exit_code != 0
        assert "Missing option" in result.output or "--env" in result.output

    def test_eval_unknown_benchmark(self, runner):
        """Eval run command fails for unknown benchmark."""
        result = runner.invoke(cli, ["eval", "run", "nonexistent", "--env", "some.module"])
        assert result.exit_code != 0
        assert isinstance(result.exception, KeyError)
        assert "Unknown benchmark" in str(result.exception)

    def test_eval_invalid_hook_module(self, runner):
        """Eval run command fails if hook module doesn't exist."""
        result = runner.invoke(cli, ["eval", "run", "aime-2024", "--env", "nonexistent.module"])
        assert result.exit_code != 0
        assert isinstance(result.exception, ValueError)

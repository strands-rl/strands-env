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

"""Module entry point for the evaluation CLI: `python -m strands_env.eval`."""

from __future__ import annotations

import os
import sys

from .cli import eval_cmd

# Ensure the current working directory is importable so user-provided hooks resolve.
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

if __name__ == "__main__":
    eval_cmd()

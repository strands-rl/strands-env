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

"""CLI entry point for Strands Agents Environments."""

from __future__ import annotations

import os
import sys

import click

# Add cwd to sys.path so local modules (e.g., examples/) are importable as dotted paths.
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

from .eval import eval_group


@click.group()
def cli() -> None:
    """Strands Agents Environments: CLI main entrypoint."""
    pass


# Register command groups
cli.add_command(eval_group)


def main() -> None:
    """Run the Strands Agents Environments CLI."""
    cli()


if __name__ == "__main__":
    main()

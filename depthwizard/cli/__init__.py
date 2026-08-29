"""DepthWizard CLI — command modules.

Each command module exposes exactly:
    NAME : str                 subcommand name (verb-style, no numbers)
    HELP : str                 one-line help
    add_parser(subparsers)     registers its argparse subparser
    run(args) -> int           executes; 0 = success

Everything fires from the repo-root ``main.py``:

    python main.py inspect --rgb-dir ... --truth-dir ...
    python main.py splits  --rgb-dir ... --truth-dir ...
    python main.py depth   --rgb-dir ... --device auto
    python main.py fit-baseline / eval-baseline / dummies / reference
    python main.py train / evaluate / infer / eval-scene / gt-check / diag
    python main.py serve  --port 8000          # FastAPI bridge for the webapp

Command modules are imported LAZILY (only when invoked), so ``main.py --help``
stays instant and works even before heavy deps (torch/transformers) are
installed.
"""

from depthwizard.cli.registry import COMMANDS, main  # noqa: F401

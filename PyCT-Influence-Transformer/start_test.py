#!/usr/bin/env python3
from __future__ import annotations

from start_cli import configure_logging, parse_args
from start_launch import run_launcher


def main(argv=None) -> None:
    args = parse_args(argv)
    configure_logging(args)
    run_launcher(args)


if __name__ == "__main__":
    main()

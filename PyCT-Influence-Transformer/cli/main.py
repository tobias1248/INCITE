from __future__ import annotations

from cli.args import configure_logging, parse_args


def main(argv=None) -> None:
    args = parse_args(argv)
    configure_logging(args)

    # Delay heavy imports so --help and argument validation stay lightweight.
    from orchestration.launcher import run_launcher

    run_launcher(args)


if __name__ == "__main__":
    main()


__all__ = ["main"]

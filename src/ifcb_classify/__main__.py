"""Package entry point so ``python -m ifcb_classify ...`` runs the CLI."""

from ifcb_classify.cli import run_cli


def main():
    """Console-script / module entry point; delegates to the CLI dispatcher."""
    run_cli()


if __name__ == "__main__":
    main()

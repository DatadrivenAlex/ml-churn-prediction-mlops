"""Allow ``python -m churn``."""

from churn.cli import main

if __name__ == "__main__":
    raise SystemExit(main())

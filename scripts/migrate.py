"""Apply the SQL migrations, idempotently.

Compose mounts `migrations/` into `docker-entrypoint-initdb.d`, which only runs on an empty
data directory. This script is the path for every other case: an existing database, a
non-Docker Postgres, or CI. Applied filenames are recorded in a `_migrations` table, so
running it twice is a no-op and running it after adding a file applies only the new one.

No Alembic. There is one schema with one migration; Alembic would be ceremony without payoff,
and a plain ordered-files scheme is something a reader can verify at a glance.

Usage:
    uv run python scripts/migrate.py [--dsn postgresql://...] [--dry-run]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"

TRACKING_TABLE = """
CREATE TABLE IF NOT EXISTS _migrations (
    filename    TEXT PRIMARY KEY,
    applied_at  TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""


def discover(directory: Path) -> list[Path]:
    """Migration files in lexical order, which is why they are numbered."""
    if not directory.is_dir():
        return []
    return sorted(directory.glob("*.sql"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsn", default=None, help="Postgres DSN; defaults to settings.")
    parser.add_argument("--dir", type=Path, default=MIGRATIONS_DIR, help="Directory of .sql files.")
    parser.add_argument(
        "--dry-run", action="store_true", help="Report what would be applied, change nothing."
    )
    args = parser.parse_args(argv)

    files = discover(args.dir)
    if not files:
        print(f"No .sql files found in {args.dir}")
        return 1

    from retrieval_engine.config import get_settings

    dsn = args.dsn if args.dsn else get_settings().postgres_dsn

    if args.dry_run:
        print(f"Would apply {len(files)} migration file(s) to {dsn.rsplit('@', 1)[-1]}:")
        for path in files:
            print(f"  {path.name}")
        return 0

    import psycopg

    applied: list[str] = []
    with psycopg.connect(dsn, autocommit=False) as connection:
        with connection.cursor() as cursor:
            cursor.execute(TRACKING_TABLE)
            cursor.execute("SELECT filename FROM _migrations")
            done = {row[0] for row in cursor.fetchall()}
        connection.commit()

        for path in files:
            if path.name in done:
                print(f"  skip {path.name} (already applied)")
                continue
            print(f"  apply {path.name}")
            with connection.cursor() as cursor:
                cursor.execute(path.read_text(encoding="utf-8"))  # type: ignore[arg-type]
                cursor.execute("INSERT INTO _migrations (filename) VALUES (%s)", (path.name,))
            connection.commit()
            applied.append(path.name)

    print(
        f"Done. {len(applied)} applied, {len(files) - len(applied)} already present."
        if applied
        else "Done. Nothing to apply, the schema is current."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

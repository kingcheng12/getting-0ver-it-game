import argparse
from pathlib import Path
from typing import Optional, Sequence

from .demonstrations import load_demonstration_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect Unity ML-Agents Release 17 demonstration files."
    )
    parser.add_argument("files", type=Path, nargs="+")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    for path in args.files:
        demonstration = load_demonstration_file(path)
        print(
            f"file={path} name={demonstration.name} "
            f"records={demonstration.recorded_steps} "
            f"transitions={sum(e.length for e in demonstration.episodes)}"
        )
        for episode in demonstration.episodes:
            print(
                f"  episode={episode.index} length={episode.length} "
                f"return={episode.episode_return:.6f} "
                f"terminal={episode.outcome} transitions={episode.length}"
            )


if __name__ == "__main__":
    main()

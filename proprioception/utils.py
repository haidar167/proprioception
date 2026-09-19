"""Utilities for reproducibility, git metadata extraction, and result logging."""

import datetime
import json
import os
import random
import subprocess
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import torch


def set_seed(seed: int = 0, num_threads: int = 4) -> None:
    """Set random seeds across random, numpy, and torch for CPU reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    max_threads = max(1, min(num_threads, os.cpu_count() or 4))
    torch.set_num_threads(max_threads)


def get_git_commit_hash(cwd: Union[str, Path, None] = None) -> str:
    """Retrieve the current Git commit hash, checking system git and standard Windows paths."""
    git_paths = [
        "git",
        r"C:\Program Files\Git\cmd\git.exe",
        r"C:\Program Files\Git\bin\git.exe",
    ]
    repo_dir = Path(cwd) if cwd else Path(__file__).resolve().parent.parent

    for git_cmd in git_paths:
        try:
            result = subprocess.run(
                [git_cmd, "-C", str(repo_dir), "rev-parse", "HEAD"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except Exception:
            continue
    return "uncommitted"


def save_results(
    data: Dict[str, Any], filepath: Union[str, Path], seed: int = 0
) -> Dict[str, Any]:
    """Enrich experiment metrics with timestamp, seed, and git commit hash, then save to JSON."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    enriched = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "seed": seed,
        "git_commit": get_git_commit_hash(path.parent),
        **data,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(enriched, f, indent=2)

    return enriched

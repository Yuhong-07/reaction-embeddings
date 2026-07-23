import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rhea_embedding.training.phase2 import train_phase2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", type=Path, default=Path("configs/data/phase2_rhea.yaml"))
    parser.add_argument("--model-config", type=Path, default=Path("configs/model/phase2_mvp.yaml"))
    parser.add_argument("--train-config", type=Path, default=Path("configs/train/phase2_mvp_cpu.yaml"))
    args = parser.parse_args()
    resolve = lambda path: path if path.is_absolute() else PROJECT_ROOT / path
    train_phase2(PROJECT_ROOT, resolve(args.data_config), resolve(args.model_config), resolve(args.train_config))

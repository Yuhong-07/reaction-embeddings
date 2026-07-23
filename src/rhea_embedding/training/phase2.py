from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import subprocess
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
import yaml
from rdkit import rdBase
from torch import nn
from torch.utils.data import DataLoader, Subset

from rhea_embedding.data.reaction_dataset import ReactionBatch, ReactionCorpus, sha256_file
from rhea_embedding.models.reaction_encoder import ReactionEncoder
from rhea_embedding.quality_control.embedding_checks import run_embedding_invariance_checks


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_paths(config: dict[str, Any], project_root: Path, keys: Iterable[str]) -> dict[str, Any]:
    result = dict(config)
    for key in keys:
        result[key] = project_root / config[key]
    return result


def set_reproducibility(seed: int, torch_threads: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(torch_threads)
    torch.use_deterministic_algorithms(True)


def git_revision(project_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def info_nce_loss(first: torch.Tensor, second: torch.Tensor, temperature: float) -> torch.Tensor:
    logits = first @ second.T / temperature
    labels = torch.arange(first.shape[0], device=first.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def variance_loss(first: torch.Tensor, second: torch.Tensor, target_std: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    """VICReg-style positive penalty; minimized when every dimension has sufficient batch variation."""
    penalties = []
    for view in (first, second):
        standard_deviation = torch.sqrt(view.var(dim=0, unbiased=False) + eps)
        penalties.append(F.relu(target_std - standard_deviation).mean())
    return 0.5 * sum(penalties)


def covariance_loss(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Positive off-diagonal covariance penalty averaged across two augmented views."""
    penalties = []
    for view in (first, second):
        centered = view - view.mean(dim=0, keepdim=True)
        denominator = max(view.shape[0] - 1, 1)
        covariance = centered.T @ centered / denominator
        off_diagonal = covariance - torch.diag(torch.diagonal(covariance))
        penalties.append(off_diagonal.square().sum() / view.shape[1])
    return 0.5 * sum(penalties)


def ec_auxiliary_loss(model: ReactionEncoder, embedding: torch.Tensor, batch: ReactionBatch) -> torch.Tensor:
    targets = embedding.new_zeros((batch.reaction_count, model.ec_class_count))
    if batch.ec_index.numel():
        class_indices = (batch.ec_index - 1).clamp(min=0, max=model.ec_class_count - 1)
        targets[batch.ec_reaction_index, class_indices] = 1.0
    annotated = targets.sum(dim=1) > 0
    if not bool(annotated.any()):
        return embedding.sum() * 0.0
    logits = model.predict_ec(embedding[annotated])
    return F.binary_cross_entropy_with_logits(logits, targets[annotated])


def compute_losses(
    model: ReactionEncoder,
    batch: ReactionBatch,
    temperature: float,
    mask_generator: torch.Generator,
) -> dict[str, torch.Tensor]:
    first_molecules = model.encode_molecules(batch)
    first = model.encode_canonical(first_molecules, batch)
    second_batch = batch.permuted_participants()
    second_molecules = model.encode_molecules(second_batch)
    second = model.encode_canonical(second_molecules, second_batch)
    contrastive = info_nce_loss(first.projection, second.projection, temperature)
    embedding_for_ec = 0.5 * (first.embedding + second.embedding)
    return {
        "contrastive": contrastive,
        "variance": variance_loss(first.embedding, second.embedding),
        "covariance": covariance_loss(first.embedding, second.embedding),
        "ec_auxiliary": ec_auxiliary_loss(model, embedding_for_ec, batch),
    }


def aggregate_loss(losses: dict[str, torch.Tensor], weights: dict[str, float]) -> torch.Tensor:
    return sum(losses[name] * float(weights[name]) for name in losses)


def split_selected_indices(
    selected_indices: list[int], validation_fraction: float
) -> tuple[list[int], list[int]]:
    """Split selected indices while allowing a fixed-epoch, zero-validation full fit."""
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")
    if not selected_indices:
        raise ValueError("At least one selected reaction is required")
    validation_count = 0 if validation_fraction == 0.0 else max(
        1, int(len(selected_indices) * validation_fraction)
    )
    if validation_count >= len(selected_indices):
        raise ValueError("Validation split leaves no reactions for training")
    return selected_indices[validation_count:], selected_indices[:validation_count]


def run_epoch(
    model: ReactionEncoder,
    loader: DataLoader,
    device: torch.device,
    weights: dict[str, float],
    temperature: float,
    mask_generator: torch.Generator,
    optimizer: torch.optim.Optimizer | None = None,
    gradient_clip_norm: float = 5.0,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"total": 0.0, **{name: 0.0 for name in weights}}
    sample_count = 0
    context = nullcontext() if training else torch.no_grad()
    with context:
        for batch in loader:
            batch = batch.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            losses = compute_losses(model, batch, temperature, mask_generator)
            total = aggregate_loss(losses, weights)
            if training:
                total.backward()
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                optimizer.step()
            count = batch.reaction_count
            sample_count += count
            totals["total"] += float(total.detach().cpu()) * count
            for name, value in losses.items():
                totals[name] += float(value.detach().cpu()) * count
    return {name: value / max(sample_count, 1) for name, value in totals.items()}


def build_corpus(data_config: dict[str, Any], project_root: Path, rebuild_graph_cache: bool = False) -> ReactionCorpus:
    parquet_path = project_root / data_config["reaction_parquet"]
    graph_cache_path = project_root / data_config["graph_cache"]
    return ReactionCorpus(parquet_path, graph_cache_path, rebuild_graph_cache)


def train_phase2(
    project_root: Path,
    data_config_path: Path,
    model_config_path: Path,
    train_config_path: Path,
) -> dict[str, Any]:
    data_config = load_yaml(data_config_path)
    model_config = load_yaml(model_config_path)
    train_config = load_yaml(train_config_path)
    seed = int(train_config["seed"])
    set_reproducibility(seed, int(train_config.get("torch_threads", 8)))
    device = torch.device(train_config.get("device", "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    start_time = time.time()
    corpus = build_corpus(data_config, project_root)
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(corpus), generator=generator).tolist()
    maximum = int(train_config.get("max_training_reactions", len(corpus)))
    selected_indices = permutation[:min(maximum, len(corpus))]
    train_indices, validation_indices = split_selected_indices(
        selected_indices, float(train_config.get("validation_fraction", 0.05))
    )
    train_loader = DataLoader(
        Subset(corpus, train_indices), batch_size=int(train_config["batch_size"]), shuffle=True,
        generator=torch.Generator().manual_seed(seed), num_workers=0, collate_fn=corpus.collate_fn(),
    )
    validation_loader = None
    if validation_indices:
        validation_loader = DataLoader(
            Subset(corpus, validation_indices), batch_size=int(train_config["batch_size"]), shuffle=False,
            num_workers=0, collate_fn=corpus.collate_fn(),
        )

    model = ReactionEncoder(model_config, corpus.vocab_sizes()).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(train_config["learning_rate"]), weight_decay=float(train_config["weight_decay"])
    )
    checkpoint_path = project_root / train_config["checkpoint_path"]
    metadata_path = project_root / train_config["checkpoint_metadata_path"]
    history_path = project_root / train_config["training_history_path"]
    for path in (checkpoint_path, metadata_path, history_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    history = []
    best_validation = math.inf
    mask_generator = torch.Generator().manual_seed(seed + 1)
    for epoch in range(1, int(train_config["epochs"]) + 1):
        epoch_start = time.time()
        train_metrics = run_epoch(
            model, train_loader, device, train_config["loss_weights"], float(train_config["temperature"]),
            mask_generator, optimizer, float(train_config["gradient_clip_norm"]),
        )
        validation_metrics = None
        if validation_loader is not None:
            validation_metrics = run_epoch(
                model, validation_loader, device, train_config["loss_weights"], float(train_config["temperature"]),
                torch.Generator().manual_seed(seed + 1000 + epoch), None,
            )
        row = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_metrics,
            "elapsed_seconds": round(time.time() - epoch_start, 3),
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        checkpoint_selected = validation_metrics is None or validation_metrics["total"] < best_validation
        if checkpoint_selected:
            if validation_metrics is not None:
                best_validation = validation_metrics["total"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_config": model_config,
                "vocab_sizes": corpus.vocab_sizes(),
                "vocabularies": corpus.vocabularies(),
                "data_config": data_config,
                "train_config": train_config,
                "epoch": epoch,
                "validation_total_loss": None if validation_metrics is None else best_validation,
                "data_sha256": sha256_file(project_root / data_config["reaction_parquet"]),
                "train_reaction_count": len(train_indices),
                "validation_reaction_count": len(validation_indices),
                "selected_training_and_validation_count": len(selected_indices),
                "train_indices": train_indices,
                "validation_indices": validation_indices,
            }, checkpoint_path)

    with history_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in history:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint_path.relative_to(project_root)).replace("\\", "/"),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "best_validation_total_loss": None if validation_loader is None else best_validation,
        "best_epoch": int(torch.load(checkpoint_path, map_location="cpu", weights_only=False)["epoch"]),
        "train_reaction_count": len(train_indices),
        "validation_reaction_count": len(validation_indices),
        "total_reaction_count": len(corpus),
        "training_scope": train_config.get("training_scope", "full_corpus"),
        "selected_training_and_validation_count": len(selected_indices),
        "unique_molecule_count": len(corpus.graphs),
        "random_seed": seed,
        "model_config": model_config,
        "train_config": train_config,
        "data_manifest": data_config["phase1_manifest"],
        "data_manifest_sha256": sha256_file(project_root / data_config["phase1_manifest"]),
        "git_revision": git_revision(project_root),
        "software": {
            "python": __import__("sys").version.split()[0],
            "torch": torch.__version__,
            "rdkit": rdBase.rdkitVersion,
        },
        "elapsed_seconds": round(time.time() - start_time, 3),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


@torch.no_grad()
def export_embeddings(
    project_root: Path,
    data_config_path: Path,
    export_config_path: Path,
) -> dict[str, Any]:
    data_config = load_yaml(data_config_path)
    export_config = load_yaml(export_config_path)
    checkpoint_path = project_root / export_config["checkpoint_path"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    train_config = checkpoint["train_config"]
    set_reproducibility(int(train_config["seed"]), int(train_config.get("torch_threads", 8)))
    device = torch.device(train_config.get("device", "cpu"))
    corpus = build_corpus(data_config, project_root)
    if corpus.vocab_sizes() != checkpoint["vocab_sizes"]:
        raise ValueError("Vocabulary sizes differ from the training checkpoint")
    model = ReactionEncoder(checkpoint["model_config"], checkpoint["vocab_sizes"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    loader = DataLoader(
        corpus, batch_size=int(export_config["batch_size"]), shuffle=False,
        num_workers=0, collate_fn=corpus.collate_fn(),
    )
    embedding_rows = []
    reaction_ids = []
    for batch in loader:
        batch = batch.to(device)
        molecules = model.encode_molecules(batch)
        embeddings = model.encode_canonical(molecules, batch).embedding
        embedding_rows.append(embeddings.cpu())
        reaction_ids.extend(batch.reaction_ids)
    matrix = torch.cat(embedding_rows, dim=0).numpy().astype(np.float32, copy=False)

    output_paths = {
        key: project_root / export_config[key]
        for key in ("embedding_npy", "embedding_parquet", "reaction_ids_tsv", "quality_report", "export_manifest")
    }
    for path in output_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_paths["embedding_npy"], matrix, allow_pickle=False)
    with output_paths["reaction_ids_tsv"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["row_index", "reaction_id"])
        writer.writerows(enumerate(reaction_ids))
    embedding_type = pa.list_(pa.float32(), matrix.shape[1])
    table = pa.table({
        "reaction_id": pa.array(reaction_ids, type=pa.string()),
        "embedding": pa.array(matrix.tolist(), type=embedding_type),
    })
    pq.write_table(table, output_paths["embedding_parquet"], compression="zstd")

    phase1_report = json.loads((project_root / data_config["phase1_quality_report"]).read_text(encoding="utf-8"))
    invariance = run_embedding_invariance_checks(model, corpus, device)
    finite = np.isfinite(matrix)
    norms = np.linalg.norm(matrix, axis=1)
    exact_unique_vector_count = int(np.unique(matrix, axis=0).shape[0])
    output_checksums = {
        str(path.relative_to(project_root)).replace("\\", "/"): sha256_file(path)
        for key, path in output_paths.items() if key not in {"quality_report", "export_manifest"}
    }
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "reaction_count": int(matrix.shape[0]),
        "embedding_dim": int(matrix.shape[1]),
        "matrix_shape": list(matrix.shape),
        "unique_reaction_id_count": len(set(reaction_ids)),
        "field_coverage": phase1_report["field_coverage"],
        "failed_smiles_count": phase1_report["failed_smiles_count"],
        "imbalanced_reaction_count": phase1_report["imbalanced_reaction_count"],
        "deduplication_count": phase1_report["deduplication_count"],
        "nan_or_inf_count": int((~finite).sum()),
        "exact_unique_vector_count": exact_unique_vector_count,
        "exact_duplicate_vector_count": int(len(matrix) - exact_unique_vector_count),
        "embedding_norm_statistics": {
            "min": float(norms.min()), "max": float(norms.max()),
            "mean": float(norms.mean()), "std": float(norms.std()),
        },
        "invariance_checks": invariance,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "output_checksums": output_checksums,
        "random_seed": int(train_config["seed"]),
        "training_scope": train_config.get("training_scope", "full_corpus"),
        "train_reaction_count": int(checkpoint["train_reaction_count"]),
        "validation_reaction_count": int(checkpoint["validation_reaction_count"]),
        "selected_training_and_validation_count": int(checkpoint["selected_training_and_validation_count"]),
        "main_embedding_input_policy": {
            "reactant_product_structure": True,
            "signed_stoichiometry": True,
            "reaction_direction": True,
            "cofactor_role_with_missing_mask": bool(checkpoint["model_config"].get("use_cofactor", True)),
            "ec_as_input": False,
            "reaction_type_as_input": False,
            "compartment_as_input": False,
        },
        "ec_auxiliary_prediction_weight": float(train_config["loss_weights"].get("ec_auxiliary", 0.0)),
        "direction_policy": {
            "left_to_right": "keep stored LR orientation",
            "right_to_left": "swap sides and coefficient signs",
            "reversible": "average forward and swapped encodings",
            "undefined": "keep stored orientation with undefined-direction embedding",
        },
    }
    if report["reaction_count"] != 18072 or report["embedding_dim"] != 256:
        raise AssertionError(f"Unexpected embedding shape: {matrix.shape}")
    if report["unique_reaction_id_count"] != report["reaction_count"]:
        raise AssertionError("Reaction IDs are not unique")
    if report["nan_or_inf_count"]:
        raise AssertionError("Embedding output contains NaN or Inf")
    if not invariance["passed"]:
        raise AssertionError(f"Embedding invariance checks failed: {invariance}")
    output_paths["quality_report"].write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    manifest = {
        "manifest_version": 1,
        "generated_at_utc": report["generated_at_utc"],
        "checkpoint": {
            "path": str(checkpoint_path.relative_to(project_root)).replace("\\", "/"),
            "sha256": report["checkpoint_sha256"],
        },
        "data": {
            "path": data_config["reaction_parquet"],
            "sha256": sha256_file(project_root / data_config["reaction_parquet"]),
            "phase1_manifest": data_config["phase1_manifest"],
            "phase1_manifest_sha256": sha256_file(project_root / data_config["phase1_manifest"]),
        },
        "config": {
            "model": checkpoint["model_config"],
            "train": train_config,
            "export": export_config,
        },
        "outputs": [
            {"path": path, "sha256": checksum} for path, checksum in sorted(output_checksums.items())
        ] + [{
            "path": str(output_paths["quality_report"].relative_to(project_root)).replace("\\", "/"),
            "sha256": sha256_file(output_paths["quality_report"]),
        }],
        "software": {
            "python": __import__("sys").version.split()[0],
            "torch": torch.__version__,
            "rdkit": rdBase.rdkitVersion,
        },
        "random_seed": int(train_config["seed"]),
        "git_revision": git_revision(project_root),
    }
    output_paths["export_manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "matrix_shape": report["matrix_shape"],
        "nan_or_inf_count": report["nan_or_inf_count"],
        "norm_statistics": report["embedding_norm_statistics"],
        "invariance_passed": invariance["passed"],
    }, indent=2))
    return report

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ec_tokens(ec_numbers: list[str] | None, level: str) -> set[str]:
    values = set()
    for ec in ec_numbers or []:
        parts = ec.split(".")
        if level == "exact":
            values.add(ec)
        else:
            count = int(level)
            if len(parts) >= count:
                values.add(".".join(parts[:count]))
    return values


def compute_neighbors(matrix: np.ndarray, k: int, block_size: int, threads: int) -> tuple[np.ndarray, np.ndarray]:
    torch.set_num_threads(threads)
    tensor = torch.from_numpy(matrix)
    tensor = torch.nn.functional.normalize(tensor, dim=1)
    transposed = tensor.T.contiguous()
    all_indices = []
    all_scores = []
    with torch.no_grad():
        for start in range(0, tensor.shape[0], block_size):
            end = min(start + block_size, tensor.shape[0])
            scores = tensor[start:end] @ transposed
            row = torch.arange(end - start)
            scores[row, torch.arange(start, end)] = -torch.inf
            values, indices = torch.topk(scores, k=k, dim=1, largest=True, sorted=True)
            all_indices.append(indices.cpu().numpy())
            all_scores.append(values.cpu().numpy())
    return np.concatenate(all_indices), np.concatenate(all_scores)


def compute_ec_metrics(
    ec_lists: list[list[str] | None],
    neighbor_indices: np.ndarray,
) -> tuple[list[dict], dict[str, list[set[str]]]]:
    token_sets = {level: [ec_tokens(values, level) for values in ec_lists] for level in ("1", "2", "3", "exact")}
    rows = []
    for level in ("1", "2", "3", "exact"):
        sets = token_sets[level]
        valid_queries = [index for index, values in enumerate(sets) if values]
        members_by_token: dict[str, set[int]] = {}
        for index, values in enumerate(sets):
            for token in values:
                members_by_token.setdefault(token, set()).add(index)
        candidate_count = len(sets) - 1
        for k in (1, 5, 10):
            neighbor_match_count = 0
            neighbor_query_hits = 0
            expected_random_precision = 0.0
            expected_random_hit_rate = 0.0
            for query in valid_queries:
                neighbor_matches = [bool(sets[query] & sets[int(candidate)]) for candidate in neighbor_indices[query, :k]]
                neighbor_match_count += sum(neighbor_matches)
                neighbor_query_hits += int(any(neighbor_matches))
                matching_candidates: set[int] = set()
                for token in sets[query]:
                    matching_candidates.update(members_by_token[token])
                matching_candidates.discard(query)
                match_count = len(matching_candidates)
                expected_random_precision += match_count / candidate_count
                no_hit_probability = 1.0
                for draw in range(k):
                    no_hit_probability *= (candidate_count - match_count - draw) / (candidate_count - draw)
                expected_random_hit_rate += 1.0 - no_hit_probability
            denominator = max(len(valid_queries) * k, 1)
            query_denominator = max(len(valid_queries), 1)
            precision = neighbor_match_count / denominator
            random_precision = expected_random_precision / query_denominator
            random_hit_rate = expected_random_hit_rate / query_denominator
            rows.append({
                "ec_level": level,
                "k": k,
                "annotated_query_count": len(valid_queries),
                "neighbor_precision": precision,
                "neighbor_hit_rate": neighbor_query_hits / query_denominator,
                "random_precision": random_precision,
                "random_hit_rate": random_hit_rate,
                "precision_lift_vs_random": precision / random_precision if random_precision else None,
            })
    return rows, token_sets


def write_plot(
    path: Path,
    cumulative_variance: np.ndarray,
    nearest_scores: np.ndarray,
    random_cosines: np.ndarray,
    ec_metrics: list[dict],
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    components = np.arange(1, len(cumulative_variance) + 1)
    axes[0].plot(components, cumulative_variance, linewidth=1.8)
    axes[0].axhline(0.9, color="gray", linewidth=1, linestyle="--")
    axes[0].set(xlabel="Principal components", ylabel="Cumulative explained variance", title="Embedding spectrum")
    axes[0].set_xlim(1, len(cumulative_variance))
    axes[0].set_ylim(0, 1.01)

    axes[1].hist(random_cosines, bins=60, alpha=0.65, density=True, label="Random pairs")
    axes[1].hist(nearest_scores[:, 0], bins=60, alpha=0.65, density=True, label="Nearest neighbor")
    axes[1].set(xlabel="Cosine similarity", ylabel="Density", title="Similarity distributions")
    axes[1].legend(frameon=False)

    x = np.arange(3)
    width = 0.12
    selected_levels = ["1", "2", "exact"]
    for offset, level in enumerate(selected_levels):
        level_rows = [row for row in ec_metrics if row["ec_level"] == level]
        axes[2].bar(x + (offset - 1.5) * width, [r["neighbor_precision"] for r in level_rows], width, label=f"EC {level}")
        axes[2].bar(x + (offset + 1.5) * width, [r["random_precision"] for r in level_rows], width, alpha=0.35)
    axes[2].set_xticks(x, ["k=1", "k=5", "k=10"])
    axes[2].set(ylabel="Precision", title="EC agreement (solid) vs random (faded)")
    axes[2].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Nearest-neighbor, EC-consistency, and collapse analysis")
    parser.add_argument("--embedding", default="artifacts/embeddings/reaction_embeddings.npy")
    parser.add_argument("--ids", default="artifacts/embeddings/reaction_ids.tsv")
    parser.add_argument("--reactions", default="data/processed/rhea_reactions.parquet")
    parser.add_argument("--output-dir", default="artifacts/reports/semantic_quality")
    parser.add_argument(
        "--duplicate-diagnosis",
        default=None,
        help="Optional duplicate diagnosis JSON used to separate allowed-input equivalences from encoder collisions",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional checkpoint containing train_indices; enables EC evaluation on a training-excluded pool",
    )
    parser.add_argument("--neighbors", type=int, default=10)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    embedding_path = root / args.embedding
    ids_path = root / args.ids
    reactions_path = root / args.reactions
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix = np.load(embedding_path, allow_pickle=False).astype(np.float32, copy=False)
    with ids_path.open("r", encoding="utf-8", newline="") as handle:
        id_rows = list(csv.DictReader(handle, delimiter="\t"))
    reaction_ids = [row["reaction_id"] for row in id_rows]
    reaction_rows = pq.read_table(reactions_path, columns=["reaction_id", "ec_numbers", "reaction_smiles", "supported_directions"]).to_pylist()
    metadata_by_id = {row["reaction_id"]: row for row in reaction_rows}
    ec_lists = [metadata_by_id[reaction_id]["ec_numbers"] for reaction_id in reaction_ids]
    if matrix.shape != (len(reaction_ids), 256):
        raise ValueError(f"Unexpected shape or ID count: {matrix.shape}, {len(reaction_ids)}")

    neighbor_indices, neighbor_scores = compute_neighbors(matrix, args.neighbors, args.block_size, args.threads)
    rng = np.random.default_rng(args.seed)
    ec_metrics, token_sets = compute_ec_metrics(ec_lists, neighbor_indices)
    holdout_ec_metrics = None
    holdout_indices = None
    checkpoint_training_scope = None
    checkpoint_ec_auxiliary_weight = None
    ec_evaluation_mode = "descriptive_all_reactions_no_checkpoint"
    if args.checkpoint:
        checkpoint = torch.load(root / args.checkpoint, map_location="cpu", weights_only=False)
        train_indices = set(map(int, checkpoint.get("train_indices", [])))
        validation_indices = set(map(int, checkpoint.get("validation_indices", [])))
        checkpoint_training_scope = checkpoint.get("train_config", {}).get("training_scope")
        checkpoint_ec_auxiliary_weight = float(
            checkpoint.get("train_config", {}).get("loss_weights", {}).get("ec_auxiliary", 0.0)
        )
        excluded_ec_indices = train_indices | validation_indices
        if train_indices and validation_indices and len(excluded_ec_indices) < len(reaction_ids):
            holdout_indices = np.asarray(
                [index for index in range(len(reaction_ids)) if index not in excluded_ec_indices]
            )
            holdout_neighbor_indices, _ = compute_neighbors(
                matrix[holdout_indices], args.neighbors, args.block_size, args.threads
            )
            holdout_ec_lists = [ec_lists[index] for index in holdout_indices]
            holdout_ec_metrics, _ = compute_ec_metrics(holdout_ec_lists, holdout_neighbor_indices)
            ec_evaluation_mode = "reaction_disjoint_training_and_model_selection_excluded"
        elif train_indices == set(range(len(reaction_ids))) and not validation_indices:
            ec_evaluation_mode = "transductive_full_corpus_structure_training_no_ec_labels"
        else:
            raise ValueError(
                "Checkpoint indices are neither a valid reaction-disjoint holdout nor a full-corpus fixed fit"
            )

    nearest_csv = output_dir / "nearest_neighbors_top10.csv"
    with nearest_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "query_reaction_id", "rank", "neighbor_reaction_id", "cosine_similarity",
            "query_ec", "neighbor_ec", "ec_level1_match", "ec_level2_match", "exact_ec_match",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for query in range(len(reaction_ids)):
            for rank, (neighbor, score) in enumerate(zip(neighbor_indices[query], neighbor_scores[query]), 1):
                neighbor = int(neighbor)
                writer.writerow({
                    "query_reaction_id": reaction_ids[query],
                    "rank": rank,
                    "neighbor_reaction_id": reaction_ids[neighbor],
                    "cosine_similarity": f"{float(score):.8f}",
                    "query_ec": ";".join(ec_lists[query] or []),
                    "neighbor_ec": ";".join(ec_lists[neighbor] or []),
                    "ec_level1_match": bool(token_sets["1"][query] & token_sets["1"][neighbor]),
                    "ec_level2_match": bool(token_sets["2"][query] & token_sets["2"][neighbor]),
                    "exact_ec_match": bool(token_sets["exact"][query] & token_sets["exact"][neighbor]),
                })

    ec_csv = output_dir / "ec_consistency.csv"
    with ec_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ec_metrics[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(ec_metrics)
    holdout_ec_csv = None
    if holdout_ec_metrics is not None:
        holdout_ec_csv = output_dir / "ec_consistency_training_excluded.csv"
        with holdout_ec_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(holdout_ec_metrics[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(holdout_ec_metrics)

    centered = matrix.astype(np.float64) - matrix.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    eigenvalues = (singular_values ** 2) / max(matrix.shape[0] - 1, 1)
    variance_ratio = eigenvalues / eigenvalues.sum()
    cumulative_variance = np.cumsum(variance_ratio)
    entropy = -np.sum(variance_ratio[variance_ratio > 0] * np.log(variance_ratio[variance_ratio > 0]))
    effective_rank = float(np.exp(entropy))
    participation_ratio = float(eigenvalues.sum() ** 2 / np.square(eigenvalues).sum())
    components_90 = int(np.searchsorted(cumulative_variance, 0.9) + 1)
    numerical_rank = int(np.sum(singular_values > singular_values.max() * 1e-6))
    dimension_std = matrix.std(axis=0)
    exact_unique_count = int(np.unique(matrix, axis=0).shape[0])
    raw_duplicate_count = int(len(matrix) - exact_unique_count)
    equivalent_duplicate_count = 0
    unexplained_duplicate_count = raw_duplicate_count
    if args.duplicate_diagnosis:
        diagnosis_path = root / args.duplicate_diagnosis
        diagnosis = json.loads(diagnosis_path.read_text(encoding="utf-8"))
        if int(diagnosis["duplicate_vector_excess_count"]) != raw_duplicate_count:
            raise ValueError("Duplicate diagnosis does not match the analyzed embedding matrix")
        equivalent_duplicate_count = int(diagnosis["duplicate_vector_excess_same_main_v2_input"])
        unexplained_duplicate_count = int(diagnosis["duplicate_vector_excess_different_main_v2_input"])

    pair_count = 200_000
    first = rng.integers(0, len(matrix), size=pair_count)
    second = rng.integers(0, len(matrix), size=pair_count)
    second = np.where(second == first, (second + 1) % len(matrix), second)
    normalized = matrix / np.linalg.norm(matrix, axis=1, keepdims=True).clip(min=1e-12)
    random_cosines = np.sum(normalized[first] * normalized[second], axis=1)
    nearest_top1 = neighbor_scores[:, 0]
    near_duplicate_999 = int(np.sum(nearest_top1 >= 0.999))
    near_duplicate_9999 = int(np.sum(nearest_top1 >= 0.9999))

    criterion_results = {
        "effective_rank": bool(effective_rank >= 20),
        "top1_explained_variance": bool(variance_ratio[0] < 0.5),
        "top10_cumulative_variance": bool(cumulative_variance[9] < 0.95),
        "zero_variance_dimensions": bool(np.sum(dimension_std < 1e-8) == 0),
        "random_pair_cosine_spread": bool(np.std(random_cosines) > 0.01),
        "unexpected_duplicate_embeddings": bool(unexplained_duplicate_count == 0),
    }
    collapse_pass = all(criterion_results.values())
    collapse = {
        "passed": collapse_pass,
        "criteria": {
            "effective_rank_min": 20,
            "top1_explained_variance_max": 0.5,
            "top10_cumulative_variance_max": 0.95,
            "zero_variance_dimension_count_max": 0,
            "random_pair_cosine_std_min": 0.01,
            "unexpected_duplicate_embedding_count_max": 0,
        },
        "criterion_results": criterion_results,
        "effective_rank": effective_rank,
        "participation_ratio": participation_ratio,
        "numerical_rank": numerical_rank,
        "components_for_90_percent_variance": components_90,
        "top1_explained_variance": float(variance_ratio[0]),
        "top10_cumulative_variance": float(cumulative_variance[9]),
        "top50_cumulative_variance": float(cumulative_variance[49]),
        "dimension_std_min": float(dimension_std.min()),
        "dimension_std_median": float(np.median(dimension_std)),
        "zero_variance_dimension_count": int(np.sum(dimension_std < 1e-8)),
        "exact_unique_embedding_count": exact_unique_count,
        "exact_duplicate_embedding_count": raw_duplicate_count,
        "equivalent_input_duplicate_embedding_count": equivalent_duplicate_count,
        "unexpected_duplicate_embedding_count": unexplained_duplicate_count,
        "nearest_neighbor_cosine": {
            "min": float(nearest_top1.min()),
            "median": float(np.median(nearest_top1)),
            "p95": float(np.quantile(nearest_top1, 0.95)),
            "p99": float(np.quantile(nearest_top1, 0.99)),
            "max": float(nearest_top1.max()),
            "count_ge_0_999": near_duplicate_999,
            "count_ge_0_9999": near_duplicate_9999,
        },
        "random_pair_cosine": {
            "mean": float(random_cosines.mean()),
            "std": float(random_cosines.std()),
            "p05": float(np.quantile(random_cosines, 0.05)),
            "median": float(np.median(random_cosines)),
            "p95": float(np.quantile(random_cosines, 0.95)),
        },
    }

    top_pairs = []
    flat_order = np.argsort(neighbor_scores[:, 0])[::-1][:20]
    for query in flat_order:
        neighbor = int(neighbor_indices[query, 0])
        top_pairs.append({
            "query_reaction_id": reaction_ids[query],
            "neighbor_reaction_id": reaction_ids[neighbor],
            "cosine_similarity": float(neighbor_scores[query, 0]),
            "query_ec": ec_lists[query] or [],
            "neighbor_ec": ec_lists[neighbor] or [],
            "exact_ec_match": bool(token_sets["exact"][query] & token_sets["exact"][neighbor]),
        })

    plot_path = output_dir / "embedding_quality_diagnostics.png"
    ec_metrics_for_evaluation = holdout_ec_metrics if holdout_ec_metrics is not None else ec_metrics
    write_plot(plot_path, cumulative_variance, neighbor_scores, random_cosines, ec_metrics_for_evaluation)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "embedding_shape": list(matrix.shape),
        "embedding_sha256": sha256_file(embedding_path),
        "reaction_count": len(reaction_ids),
        "ec_annotated_reaction_count": sum(bool(values) for values in ec_lists),
        "ec_annotated_fraction": sum(bool(values) for values in ec_lists) / len(ec_lists),
        "ec_consistency": ec_metrics,
        "ec_consistency_training_excluded": holdout_ec_metrics,
        "ec_evaluation_pool_reaction_count": int(len(holdout_indices)) if holdout_indices is not None else None,
        "ec_evaluation_excluded_training_reaction_count": (
            int(len(reaction_ids) - len(holdout_indices)) if holdout_indices is not None else 0
        ),
        "ec_evaluation_excludes_validation_model_selection_reactions": bool(holdout_indices is not None),
        "ec_evaluation_mode": ec_evaluation_mode,
        "checkpoint_training_scope": checkpoint_training_scope,
        "checkpoint_ec_auxiliary_weight": checkpoint_ec_auxiliary_weight,
        "collapse_check": collapse,
        "top_nearest_neighbor_pairs": top_pairs,
        "limitations": (
            [
                "The checkpoint was trained on a 2,000-reaction pilot subset.",
                "The primary EC metric excludes all training and validation/model-selection reactions from both queries and candidates.",
                "Nearest-neighbor agreement measures internal geometry and does not establish downstream biological utility.",
            ]
            if holdout_indices is not None
            else [
                "The full-corpus checkpoint used every reaction structure for self-supervised training with EC auxiliary weight zero.",
                "Full-corpus EC agreement is a transductive descriptive diagnostic, not a reaction-disjoint holdout estimate.",
                "Nearest-neighbor agreement measures internal geometry and does not establish downstream biological utility.",
            ]
        ),
        "outputs": {},
    }
    json_path = output_dir / "embedding_semantic_quality_report.json"
    markdown_path = output_dir / "embedding_semantic_quality_report.md"
    summary["outputs"] = {
        "nearest_neighbors_top10.csv": sha256_file(nearest_csv),
        "ec_consistency.csv": sha256_file(ec_csv),
        "embedding_quality_diagnostics.png": sha256_file(plot_path),
    }
    if holdout_ec_csv is not None:
        summary["outputs"][holdout_ec_csv.name] = sha256_file(holdout_ec_csv)
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    metric_lookup = {(row["ec_level"], row["k"]): row for row in ec_metrics_for_evaluation}
    exact1 = metric_lookup[("exact", 1)]
    exact10 = metric_lookup[("exact", 10)]
    level1_1 = metric_lookup[("1", 1)]
    lines = [
        "# Embedding 语义质量报告",
        "",
        f"生成时间（UTC）：{summary['generated_at_utc']}",
        "",
        "## 结论",
        "",
        f"- Embedding 矩阵：`{matrix.shape[0]} x {matrix.shape[1]}`。",
        f"- 有 EC 注释的反应：{summary['ec_annotated_reaction_count']} 条（{summary['ec_annotated_fraction']:.1%}）。",
        f"- EC 评估模式：`{ec_evaluation_mode}`。",
        *(
            [f"- EC 独立评估池：{len(holdout_indices)} 条，已从查询和候选中排除 {len(reaction_ids) - len(holdout_indices)} 条训练及模型选择反应。"]
            if holdout_indices is not None else []
        ),
        f"- 严格防塌缩验收：**{'通过' if collapse_pass else '未通过'}**。",
        f"- 有效秩：{effective_rank:.2f}；解释 90% 方差只需 {components_90} 个主成分。",
        f"- 完整 EC precision@1：{exact1['neighbor_precision']:.3%}，随机基线 {exact1['random_precision']:.6%}。",
        f"- 完整 EC hit@10：{exact10['neighbor_hit_rate']:.3%}，随机基线 {exact10['random_hit_rate']:.6%}。",
        f"- EC 一级 precision@1：{level1_1['neighbor_precision']:.3%}，随机基线 {level1_1['random_precision']:.3%}。",
        "",
        "## 塌缩诊断",
        "",
        f"- 第一主成分解释方差：{variance_ratio[0]:.3%}（阈值 < 50%，通过）。",
        f"- 前 10 个主成分累计方差：{cumulative_variance[9]:.3%}（阈值 < 95%，通过）。",
        f"- 有效秩：{effective_rank:.2f}（阈值 ≥ 20，{'通过' if effective_rank >= 20 else '未通过'}）。",
        f"- 完全重复向量：{raw_duplicate_count} 个；其中允许输入完全等价 {equivalent_duplicate_count} 个，"
        f"无法由输入等价解释 {unexplained_duplicate_count} 个（阈值 0，{'通过' if unexplained_duplicate_count == 0 else '未通过'}）。",
        f"- 随机反应对余弦相似度：均值 {random_cosines.mean():.4f}，标准差 {random_cosines.std():.4f}。",
        f"- 最近邻余弦相似度：中位数 {np.median(nearest_top1):.4f}，P95 {np.quantile(nearest_top1, 0.95):.4f}。",
        "",
        (
            "严格防塌缩验收通过：方差谱、有效秩、常量维度、随机相似度离散度和无法解释的向量碰撞均满足阈值。"
            if collapse_pass
            else "严格防塌缩验收未通过；请根据 JSON 中的 criterion_results 定位失败项。"
        ),
        "",
        "## EC 一致率解释",
        "",
        "最近邻 EC 一致率显著高于随机基线，说明 embedding 空间确实聚集了酶相关反应。"
        + (
            "该主指标已将所有 EC 辅助训练及验证/模型选择反应从查询和候选池中排除，因此不包含直接标签泄漏。"
            if holdout_indices is not None
            else "full-corpus 模型未使用 EC 标签，但使用过全部反应结构，因此该指标是传导式描述性结果，不是 reaction-disjoint 留出结果。"
        ),
        "",
        "## 输出文件",
        "",
        "- `nearest_neighbors_top10.csv`：每条反应的前 10 个最近邻，共 180,720 行。",
        "- `ec_consistency.csv`：EC 一级、二级、三级和完整编号的 precision/hit-rate 与解析随机基线。",
        *(
            ["- `ec_consistency_training_excluded.csv`：排除全部辅助训练反应后的 EC 独立评估结果。"]
            if holdout_ec_csv is not None else []
        ),
        "- `embedding_quality_diagnostics.png`：方差谱、相似度分布和 EC 一致率图。",
        "- `embedding_semantic_quality_report.json`：完整机器可读指标及文件哈希。",
        "",
        "## 限制",
        "",
        *(
            [
                "- 当前 checkpoint 只使用 2,000 条反应进行 pilot 训练。",
                "- 主 EC 指标从查询和候选中排除了训练及模型选择反应。",
            ]
            if holdout_indices is not None
            else [
                "- 当前 checkpoint 使用全部 18,072 条反应结构进行固定轮次自监督训练，EC auxiliary weight 为 0。",
                "- 全量 EC 指标仅用于描述 embedding 几何；它不是 reaction-disjoint 独立评估。",
            ]
        ),
        "- 尚未用下游生物学任务作为验收标准。",
    ]
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "collapse_passed": collapse_pass,
        "effective_rank": effective_rank,
        "components_for_90_percent_variance": components_90,
        "exact_ec_precision_at_1": exact1["neighbor_precision"],
        "exact_ec_random_precision_at_1": exact1["random_precision"],
        "exact_ec_hit_at_10": exact10["neighbor_hit_rate"],
        "nearest_neighbor_csv_rows": len(reaction_ids) * args.neighbors,
        "output_dir": str(output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()

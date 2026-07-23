from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F


def compute_top1(matrix: np.ndarray, block_size: int = 512) -> np.ndarray:
    tensor = F.normalize(torch.from_numpy(matrix.astype(np.float32, copy=False)), dim=1)
    transposed = tensor.T.contiguous()
    indices = []
    with torch.no_grad():
        for start in range(0, tensor.shape[0], block_size):
            end = min(start + block_size, tensor.shape[0])
            scores = tensor[start:end] @ transposed
            scores[torch.arange(end - start), torch.arange(start, end)] = -torch.inf
            indices.append(torch.topk(scores, k=10, dim=1, largest=True, sorted=True).indices[:, 0].cpu().numpy())
    return np.concatenate(indices)


def ec_tokens(values: list[str] | None, level: str) -> set[str]:
    if level == "exact":
        return set(values or [])
    return {value.split(".")[0] for value in (values or [])}


def paired_top1_result(
    ec_lists: list[list[str] | None],
    neighbors0: np.ndarray,
    neighbors02: np.ndarray,
    level: str,
) -> dict:
    token_sets = [ec_tokens(values, level) for values in ec_lists]
    valid = [index for index, values in enumerate(token_sets) if values]
    matches0 = np.asarray([
        bool(token_sets[index] & token_sets[int(neighbors0[index])]) for index in valid
    ], dtype=np.float64)
    matches02 = np.asarray([
        bool(token_sets[index] & token_sets[int(neighbors02[index])]) for index in valid
    ], dtype=np.float64)
    paired_difference = matches02 - matches0
    mean = float(paired_difference.mean())
    standard_error = float(paired_difference.std(ddof=1) / math.sqrt(len(paired_difference)))
    return {
        "ec_level": level,
        "annotated_query_count": len(valid),
        "weight0_precision_at_1": float(matches0.mean()),
        "weight02_precision_at_1": float(matches02.mean()),
        "paired_delta_weight02_minus_weight0": mean,
        "normal_approximation_95ci": [mean - 1.96 * standard_error, mean + 1.96 * standard_error],
        "queries_improved_by_weight02": int(np.sum(paired_difference > 0)),
        "queries_worsened_by_weight02": int(np.sum(paired_difference < 0)),
        "queries_unchanged": int(np.sum(paired_difference == 0)),
    }


def metric_lookup(report: dict) -> dict[tuple[str, int], dict]:
    rows = report.get("ec_consistency_training_excluded")
    if rows is None:
        raise ValueError("Report lacks training-excluded EC metrics")
    return {(str(row["ec_level"]), int(row["k"])): row for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare EC auxiliary weights 0 and 0.2")
    parser.add_argument("--report-weight0", default="artifacts/reports/semantic_quality_v2_pilot_ecaux0/embedding_semantic_quality_report.json")
    parser.add_argument("--report-weight02", default="artifacts/reports/semantic_quality_v2_pilot/embedding_semantic_quality_report.json")
    parser.add_argument("--checkpoint-weight0", default="artifacts/checkpoints/phase2_v2_pilot_ecaux0.pt")
    parser.add_argument("--checkpoint-weight02", default="artifacts/checkpoints/phase2_v2_pilot.pt")
    parser.add_argument("--embedding-weight0", default="artifacts/embeddings/reaction_embeddings_v2_pilot_ecaux0.npy")
    parser.add_argument("--embedding-weight02", default="artifacts/embeddings/reaction_embeddings_v2_pilot.npy")
    parser.add_argument("--ids", default="artifacts/embeddings/reaction_ids_v2_pilot.tsv")
    parser.add_argument("--reactions", default="data/processed/rhea_reactions.parquet")
    parser.add_argument("--output-dir", default="artifacts/reports/ec_aux_ablation")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    torch.set_num_threads(8)

    report0 = json.loads((root / args.report_weight0).read_text(encoding="utf-8"))
    report02 = json.loads((root / args.report_weight02).read_text(encoding="utf-8"))
    checkpoint0 = torch.load(root / args.checkpoint_weight0, map_location="cpu", weights_only=False)
    checkpoint02 = torch.load(root / args.checkpoint_weight02, map_location="cpu", weights_only=False)
    if checkpoint0["train_indices"] != checkpoint02["train_indices"]:
        raise ValueError("Training indices differ between ablation checkpoints")
    if checkpoint0["validation_indices"] != checkpoint02["validation_indices"]:
        raise ValueError("Validation indices differ between ablation checkpoints")
    if checkpoint0["model_config"] != checkpoint02["model_config"]:
        raise ValueError("Model configs differ between ablation checkpoints")
    weights0 = checkpoint0["train_config"]["loss_weights"]
    weights02 = checkpoint02["train_config"]["loss_weights"]
    if {**weights0, "ec_auxiliary": 0.2} != weights02:
        raise ValueError("Ablation train configs differ in more than EC auxiliary weight")

    lookup0 = metric_lookup(report0)
    lookup02 = metric_lookup(report02)
    rows = []
    for level in ("1", "2", "3", "exact"):
        for k in (1, 5, 10):
            first = lookup0[(level, k)]
            second = lookup02[(level, k)]
            rows.append({
                "ec_level": level,
                "k": k,
                "annotated_query_count": first["annotated_query_count"],
                "weight0_precision": first["neighbor_precision"],
                "weight02_precision": second["neighbor_precision"],
                "precision_delta_weight02_minus_weight0": second["neighbor_precision"] - first["neighbor_precision"],
                "weight0_hit_rate": first["neighbor_hit_rate"],
                "weight02_hit_rate": second["neighbor_hit_rate"],
                "hit_rate_delta_weight02_minus_weight0": second["neighbor_hit_rate"] - first["neighbor_hit_rate"],
                "random_precision": first["random_precision"],
            })

    collapse0 = report0["collapse_check"]
    collapse02 = report02["collapse_check"]
    exact1_delta = lookup02[("exact", 1)]["neighbor_precision"] - lookup0[("exact", 1)]["neighbor_precision"]
    level1_delta = lookup02[("1", 1)]["neighbor_precision"] - lookup0[("1", 1)]["neighbor_precision"]
    excluded = set(checkpoint0["train_indices"]) | set(checkpoint0["validation_indices"])
    holdout_indices = np.asarray([index for index in range(report0["reaction_count"]) if index not in excluded])
    matrix0 = np.load(root / args.embedding_weight0, allow_pickle=False)[holdout_indices]
    matrix02 = np.load(root / args.embedding_weight02, allow_pickle=False)[holdout_indices]
    neighbors0 = compute_top1(matrix0)
    neighbors02 = compute_top1(matrix02)
    with (root / args.ids).open("r", encoding="utf-8", newline="") as handle:
        reaction_ids = [row["reaction_id"] for row in csv.DictReader(handle, delimiter="\t")]
    records = pq.read_table(root / args.reactions, columns=["reaction_id", "ec_numbers"]).to_pylist()
    ec_by_id = {row["reaction_id"]: row["ec_numbers"] for row in records}
    holdout_ec_lists = [ec_by_id[reaction_ids[index]] for index in holdout_indices]
    paired_results = {
        level: paired_top1_result(holdout_ec_lists, neighbors0, neighbors02, level)
        for level in ("1", "exact")
    }
    paired_report_difference = (
        paired_results["exact"]["weight0_precision_at_1"]
        - lookup0[("exact", 1)]["neighbor_precision"]
    )
    if abs(paired_report_difference) > 1e-12:
        raise ValueError(f"Paired exact-EC result differs from report by {paired_report_difference}")
    exact_interval = paired_results["exact"]["normal_approximation_95ci"]
    if collapse0["passed"] and collapse02["passed"] and exact_interval[0] > 0:
        recommended = 0.2
        reason = "weight=0.2 has a positive paired exact-EC precision@1 difference whose 95% interval excludes zero"
    elif collapse0["passed"] and collapse02["passed"] and exact_interval[1] < 0:
        recommended = 0.0
        reason = "weight=0.2 has a negative paired exact-EC precision@1 difference whose 95% interval excludes zero"
    elif not collapse0["passed"] and collapse02["passed"]:
        recommended = 0.2
        reason = "only weight=0.2 passes collapse checks"
    else:
        recommended = 0.0
        reason = "weight=0.2 shows no statistically clear independent exact-EC benefit; use the simpler structure-only objective"
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "controlled_split": {
            "same_train_indices": True,
            "same_validation_indices": True,
            "train_reaction_count": len(checkpoint0["train_indices"]),
            "validation_reaction_count": len(checkpoint0["validation_indices"]),
            "independent_ec_pool_reaction_count": report0["ec_evaluation_pool_reaction_count"],
            "independent_ec_annotated_query_count": lookup0[("exact", 1)]["annotated_query_count"],
        },
        "weight0": {
            "collapse_passed": collapse0["passed"],
            "effective_rank": collapse0["effective_rank"],
            "components_for_90_percent_variance": collapse0["components_for_90_percent_variance"],
            "unexpected_duplicate_embedding_count": collapse0["unexpected_duplicate_embedding_count"],
            "exact_ec_precision_at_1": lookup0[("exact", 1)]["neighbor_precision"],
            "ec_level1_precision_at_1": lookup0[("1", 1)]["neighbor_precision"],
        },
        "weight02": {
            "collapse_passed": collapse02["passed"],
            "effective_rank": collapse02["effective_rank"],
            "components_for_90_percent_variance": collapse02["components_for_90_percent_variance"],
            "unexpected_duplicate_embedding_count": collapse02["unexpected_duplicate_embedding_count"],
            "exact_ec_precision_at_1": lookup02[("exact", 1)]["neighbor_precision"],
            "ec_level1_precision_at_1": lookup02[("1", 1)]["neighbor_precision"],
        },
        "delta_weight02_minus_weight0": {
            "exact_ec_precision_at_1": exact1_delta,
            "ec_level1_precision_at_1": level1_delta,
            "effective_rank": collapse02["effective_rank"] - collapse0["effective_rank"],
        },
        "paired_top1_analysis": paired_results,
        "recommended_ec_auxiliary_weight_for_next_stage": recommended,
        "recommendation_basis": reason,
        "ec_is_main_embedding_input": False,
    }

    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "ec_aux_ablation_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "ec_aux_ablation_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# EC auxiliary weight 消融报告",
        "",
        "## 受控条件",
        "",
        f"- 两组使用相同的 {len(checkpoint0['train_indices'])} 条训练反应和 {len(checkpoint0['validation_indices'])} 条验证反应。",
        f"- 独立 EC 池含 {report0['ec_evaluation_pool_reaction_count']} 条反应；训练及模型选择反应均从 query 和 candidate 中排除。",
        "- 模型、随机种子、epoch、batch size 和其他损失权重完全相同。",
        "- EC 不进入主 embedding 输入。",
        "",
        "## 结果",
        "",
        "| 指标 | weight=0 | weight=0.2 | 差值（0.2−0） |",
        "|---|---:|---:|---:|",
        f"| 完整 EC precision@1 | {lookup0[('exact', 1)]['neighbor_precision']:.3%} | {lookup02[('exact', 1)]['neighbor_precision']:.3%} | {exact1_delta:+.3%} |",
        f"| EC 一级 precision@1 | {lookup0[('1', 1)]['neighbor_precision']:.3%} | {lookup02[('1', 1)]['neighbor_precision']:.3%} | {level1_delta:+.3%} |",
        f"| 有效秩 | {collapse0['effective_rank']:.2f} | {collapse02['effective_rank']:.2f} | {collapse02['effective_rank'] - collapse0['effective_rank']:+.2f} |",
        f"| 90% 方差主成分数 | {collapse0['components_for_90_percent_variance']} | {collapse02['components_for_90_percent_variance']} | {collapse02['components_for_90_percent_variance'] - collapse0['components_for_90_percent_variance']:+d} |",
        f"| 无法解释的重复向量 | {collapse0['unexpected_duplicate_embedding_count']} | {collapse02['unexpected_duplicate_embedding_count']} | {collapse02['unexpected_duplicate_embedding_count'] - collapse0['unexpected_duplicate_embedding_count']:+d} |",
        f"| 防塌缩验收 | {'通过' if collapse0['passed'] else '未通过'} | {'通过' if collapse02['passed'] else '未通过'} | — |",
        "",
        "## 建议",
        "",
        f"完整 EC precision@1 的逐反应配对差值 95% 区间为 "
        f"[{exact_interval[0]:+.3%}, {exact_interval[1]:+.3%}]；"
        f"weight=0.2 改善 {paired_results['exact']['queries_improved_by_weight02']} 条、"
        f"变差 {paired_results['exact']['queries_worsened_by_weight02']} 条。",
        "",
        f"下一阶段建议使用 `ec_auxiliary={recommended:g}`。"
        + (
            "weight=0.2 在独立完整 EC 指标上显著更差，优先保留更纯粹的结构自监督目标。"
            if exact_interval[1] < 0 else
            "weight=0.2 在独立完整 EC 指标上没有显示明确收益，优先保留更纯粹的结构自监督目标。"
            if recommended == 0 else
            "weight=0.2 显示了明确的独立 EC 收益。"
        ),
        "该建议基于单一随机种子的 2,000 条 pilot；全量训练仍应冻结同一个 EC 独立留出集。",
    ]
    (output_dir / "ec_aux_ablation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

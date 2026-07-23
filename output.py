import pandas as pd
import numpy as np
from pathlib import Path

# 路径
input_path = Path(r"D:\Codex\2026-07-21\artifacts\embeddings\reaction_embeddings_v2_pilot_ecaux0.parquet")
output_path = input_path.with_suffix(".csv")

# 读取数据（只读必要列更高效）
df = pd.read_parquet(input_path, columns=["reaction_id", "embedding"])

print(df.shape)
print(df.columns)
print(df.head(3))

# 转换 embedding（更快 & 更安全）
emb_array = np.vstack(df["embedding"].values)

# 自动获取维度（避免写死256）
dim = emb_array.shape[1]

embedding_columns = pd.DataFrame(
    emb_array,
    columns=[f"embedding_{i}" for i in range(dim)]
)

# ⚠️ 重置索引，避免潜在错位
output = pd.concat(
    [df[["reaction_id"]].reset_index(drop=True), embedding_columns],
    axis=1
)

# 保存
output.to_csv(output_path, index=False)

print(f"Saved to: {output_path}")
#!/usr/bin/env bash
set -euo pipefail

# 1) 生成会暴露问题的输入 brain_sample.csv
python - <<'PY'
import numpy as np
data = np.zeros((20, 20), dtype=int)
data[-1, :] = 1  # 仅最后一行为 1
np.savetxt("brain_sample.csv", data, fmt='%d', delimiter=',')
PY

# 2) 跑当前版本的脚本，生成 brain_average.csv
# 注意：如果脚本文件名不是 sagittal_brain.py，请改成你的实际文件名
python sagittal_brain.py

# 3) 读取结果的第一行
line="$(head -n1 brain_average.csv | tr -d '\r\n')"

# 4) 计算期望输出（20 个 0.1 逗号分隔）
expected="$(python - <<'PY'
arr = ["0.1"] * 20
print(",".join(arr))
PY
)"

# 5) 和期望比对：相同 -> 返回 0（good），否则返回 1（bad）
if [ "$line" = "$expected" ]; then
  exit 0
else
  echo "Expected: $expected"
  echo "Got     : $line"
  exit 1
fi

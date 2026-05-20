# Node01 产物清理候选清单（不执行删除）

日期：2026-05-10

本清单只用于后续人工确认清理范围。本次任务不删除任何产物，不移动当前稳定 NeoVerse run，不覆盖当前 eval matrix，也不覆盖已有 preview 目录。

## 保留

- `D:\node01_spin_runtime_ascii\mvp-demo\output\neoverse_fused_runs\2026-04-26_j10_yp20_r02params_r01`
  - 当前 `j10 static` 稳定 NeoVerse fused 4D 基线，仍被 `node01_neoverse_fused_4d_eval_matrix_v1` 引用。
- `D:\node01_spin_runtime_ascii\mvp-demo\output\neoverse_fused_runs\node01_neoverse_fused_4d_eval_matrix_v1`
  - 当前 Node01 `3 identities x 2 scenes` NeoVerse 4D eval matrix 的几何产物目录。
- `research/plans/tri_camera_node_3d_aware_reid/benchmarks/node01_neoverse_fused_4d_eval_matrix_v1.json`
  - 当前正式三相机矩阵 manifest。
- 当前 6 个 scene：
  - `mj_node01_j10_spin_static_yp20_a`
  - `mj_node01_j10_spin_circle_yp_b`
  - `mj_node01_uav1_spin_static_yp_a`
  - `mj_node01_uav1_spin_circle_yp_b`
  - `mj_node01_su34_spin_static_yp_a`
  - `mj_node01_su34_spin_circle_yp_b`
- `mvp-demo/output/evals/node01_neoverse_fused_4d_eval_matrix_v1`
  - 当前 RGB-only 与 RGB+NeoVerse fused 4D 的正式评测结果，结论为可接入可评测，GT-mask bootstrap 下与 RGB-only 持平。

## 保留或归档后再删

- `D:\node01_spin_runtime_ascii\mvp-demo\output\neoverse_fused_runs\2026-04-26_j10_yp20_dense_points_r01`
  - `research/plans/ACTIVE_PLAN.md`、`research/plans/tri_camera_node_3d_aware_reid/master_plan_zh.md`、`research/guides/node01_neoverse_fused_4d_reid_zh.md` 中把它作为负向消融证据引用。
  - 若未来清理，建议先把关键 `meta/index/report` 或指标摘要归档到 `research/`，再考虑删除大体积点云与 preview。

## 暂不删

- `D:\node01_spin_runtime_ascii\mvp-demo\output\neoverse_fused_runs\2026-04-26_j10_yp20_thin_r01`
  - 收益：删除后可减少旧参数探索目录和 preview 噪声。
  - 风险：它仍可作为稀疏/薄点云路线的历史对照；如果没有把参数、失败表现和与 `r02params` 的差异写入文档，直接删除会降低可追溯性。
  - 建议：暂不删；若后续确认无引用价值，先归档 `fusion_meta.json`、`dynamic_constraint_meta.json`、关键 preview 缩略图和运行参数。

## 可删但收益小

- `mvp-demo/output/evals/node01_neoverse_fused_4d_r02params_smoke`
  - 旧 smoke eval，可由正式 eval matrix 覆盖；删除收益主要是减少目录噪声。
- `mvp-demo/output/evals/node01_recon_spin_v1_abs_2026_04_11`
  - 4 月 11 日小 eval JSON，已不是当前主线结果。
- `mvp-demo/output/evals/node01_recon_spin_v1_j10_su34_subset`
  - 旧 subset eval，当前 6-scene matrix 覆盖主线用途。
- `mvp-demo/output/evals/node01_recon_spin_subset_j10_su34_v1`
  - 旧 subset eval，保留价值低于当前 eval matrix。
- 明显临时 preview smoke：
  - `D:\node01_spin_runtime_ascii\mvp-demo\output\neoverse_fused_runs\2026-04-26_j10_yp20_r02params_r01\mj_node01_j10_spin_static_yp20_a\preview\local_target_compare_v1_smoke`
  - `D:\node01_spin_runtime_ascii\mvp-demo\output\neoverse_fused_runs\2026-04-26_j10_yp20_r02params_r01\mj_node01_j10_spin_static_yp20_a\preview\local_target_rgb_points_compare_v1_smoke`
  - 本次不删除；若清理，建议只删带 `_smoke` 的临时 preview，不动正式 preview 子目录。

## Git 噪声

- 已确认 `third_party/Track4World/` 是当前 `git status --short -uall -- third_party` 中的 untracked 来源。
- 不删除 `third_party/Track4World/`。
- 建议在 `.gitignore` 中加入 `third_party/Track4World/`，与现有 `third_party/gsplat_pypi/`、`third_party/dggt/` 的本地第三方仓库忽略规则保持一致。

## 本次执行状态

- 删除：未执行。
- 移动/归档：未执行。
- 覆盖当前稳定 NeoVerse run：未执行。
- 覆盖当前 eval matrix：未执行。
- 覆盖已有 preview 目录：未执行。

# 2024-CGAD PyG 重构版（移除 DGL）

本补丁将原仓库中的 DGL 随机游走替换为基于 PyG `edge_index` 的随机游走重启（RWR）采样逻辑，避免 `dgl==0.4.1` 在新显卡/CUDA 环境中的兼容问题。

## 替换内容

- 删除 `import dgl`、`dgl.random.seed`、`adj_to_dgl_graph`、`dgl.contrib.sampling.random_walk_with_restart`。
- 新增 `utils.build_neighbor_lists()`：从 PyG `edge_index` 构建邻接表。
- 新增 `utils.generate_rwr_subgraph()`：使用 PyG 图结构进行 RWR 子图采样，并保持“目标节点放在子图最后一位”的原始约定。
- 修复 GPU 推理阶段 `.detach().numpy()` 报错，统一改为 `.detach().cpu().numpy()`。
- 修复 `math.ceil(nb_nodes // batch_size)` 的批次数计算问题，改为 `math.ceil(nb_nodes / batch_size)`。
- 重构导入路径，补充 `__init__.py`，避免 `from run import *` 这类不稳定导入。
- `get_scores()` 改为 `scikit-learn` 实现，减少对旧版 PyGOD API 的依赖。

## 使用方式

将本目录中的文件覆盖到原 `2024-CGAD` 仓库根目录：

```bash
python main.py --dataset inj_cora --data_dir ./dataset --device cuda:0
```

如果数据文件仍在仓库根目录，可省略 `--data_dir`：

```bash
python main.py --dataset inj_cora --device cuda:0
```

数据文件仍按原仓库约定读取：

- `<dataset>.pt`
- `<dataset>.json`
- `<dataset>.txt`

## 说明

RWR 采样在 CPU 上执行，训练张量仍放在 `--device` 指定的 GPU/CPU 上。这种设计避免依赖 `torch-cluster` 的 CUDA random_walk 扩展，兼容性更好，尤其适合新 CUDA/50 系显卡环境。

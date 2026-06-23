# CGAD
A PyTorch implementation of "CGAD: A Novel Contrastive Learning-Based Framework for Anomaly Detection in Attributed Networks".

## Requirements:
```bash
uv venv -p 3.12
uv pip install torch==2.11.0 torch_geometric scikit-learn --torch-backend=cu128
```

## 运行示例

默认数据目录就是 `~/datasets/GAD/mat`：

```bash
cd /path/to/2024-CGAD
python main.py --dataset ACM --device cuda:0
```

指定数据目录：

```bash
python main.py --dataset ACM --data_dir ~/datasets/GAD/mat --device cuda:0
```

重新生成社区文件和 coef 文件：

```bash
python main.py --dataset ACM --force_preprocess
```

若 Louvain 社区过多，可以限制社区数：

```bash
python main.py --dataset ACM --max_communities 10
```

或使用更快的连通分量作为社区：

```bash
python main.py --dataset ACM --community_method components
```

## 输出缓存

首次训练会自动生成：

```text
~/datasets/GAD/mat/cgad_preprocess/<dataset>.json
~/datasets/GAD/mat/cgad_preprocess/<dataset>.pkl
```

之后默认直接复用缓存。使用 `--force_preprocess` 可重新生成。

## 注意

`coef` 是 `N x N` 的余弦相似度矩阵，大图会占用较多内存。例如 50,000 节点的 dense float64 相似度矩阵会非常大。若数据集特别大，建议先把 `utils/utils.py` 中 `generate_coef` 改为 top-k 相似度或分块计算。

## Citing Our Work

If you compare, build on, or use aspects of our framework, please cite the following paper:
```
@article{wan2024cgad,
  title={CGAD: A Novel Contrastive Learning-Based Framework for Anomaly Detection in Attributed Networks},
  author={Wan, Yun and Zhang, Dapeng and Liu, Dong and Xiao, Feng},
  journal={Neurocomputing},
  pages={128379},
  year={2024},
  publisher={Elsevier}
}

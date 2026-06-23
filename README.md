# CGAD
A PyTorch implementation of "CGAD: A Novel Contrastive Learning-Based Framework for Anomaly Detection in Attributed Networks".

## Requirements:
```bash
uv venv -p 3.12
uv pip install torch==2.11.0 torch_geometric scikit-learn optuna pandas --torch-backend=cu128
```

## 示例命令

```bash
python main.py \
  --dataset cora \
  --data_dir ~/datasets/GAD/mat \
  --train_dir ./runs/cora \
  --runs 5 \
  --num_epoch 100 \
  --save_score_run 1
```

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

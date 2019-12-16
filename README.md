# Higher-order Weighted Graph Convolutional Networks

Higher-order Weighted Graph Convolutional Networks (Songtao Liu et.al.): https://arxiv.org/abs/1911.04129

The is a TensorFlow implementation of Higher-order Weighted Graph Convolutional Networks, we reference the implementation of [GCN](https://github.com/tkipf/gcn)

![comp](https://user-images.githubusercontent.com/33899401/70893840-d3a6ee00-2026-11ea-8747-4f9a6e31d247.jpg)
## Requirements

- tensorflow (>0.12)
- networkx
- osqp

## Run the code

```bash
python train.py --dataset cora --max_order 2
```

## Cite

If you use the code, please cite our paper: 

```
@article{liu2019higher,
  title={Higher-order Weighted Graph Convolutional Networks},
  author={Liu, Songtao and Chen, Lingwei and Dong, Hanze and Wang, Zihao and Wu, Dinghao and Huang, Zengfeng},
  journal={arXiv preprint arXiv:1911.04129},
  year={2019}
} 
```

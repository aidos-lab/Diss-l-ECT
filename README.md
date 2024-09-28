# Diss-l-ECT: Dissecting Graph Data with local Euler Characteristic Transforms

## Overview of local_ect.py
This module provides two primary functions for working with graph datasets, specifically leveraging **Local Euler Characteristic Transform (l-ECT)** and **XGBoost** for node classification tasks for featured graphs. The `compute_local_ect` function computes the ECT for local graph neighborhoods, and `xgb_model` uses the (l-ECT) information to train a machine learning model using **XGBoost**.

### Dependencies
- `torch`
- `torch_geometric`
- `numpy`
- `matplotlib`
- `sklearn`
- `xgboost`

### Dataset
The code is compatible with graph datasets from `torch_geometric`. It particularly supports node classification tasks using datasets like **WebKB**, but other datasets from `torch_geometric.datasets` can be used with minimal modifications.

---

## Function: `compute_local_ect`

### Description:
This function computes the Local Euler Characteristic Transform (l-ECT) for each node in a graph dataset. The ECT captures structural information within a local neighborhood of a node up to a given radius (i.e., the number of hops). The result is a feature vector that encodes this local structure.

### Parameters:
- **`dataset`**: `torch_geometric` dataset  
  The input graph dataset for which to compute the ECT. It should contain node features, edges, and labels.
  
- **`radius`**: `int`, default=1  
  The number of hops to consider when extracting the local graph neighborhood for each node.
  
- **`ECT_TYPE`**: `str`, default='points'  
  The type of structural information for the ECT calculation. Possible values are:
  - `'points'`: Information about the nodes.
  - `'edges'`: Information about the edges.

- **`NUM_THETAS`**: `int`, default=64  
  The dimensionality of the ECT approximation. The final ECT vector has dimensions `NUM_THETAS x NUM_THETAS`.

- **`DEVICE`**: `str`, default='cpu'  
  The device to be used for computation. Set it to `'cuda'` if GPU computation is needed.

- **`subsample_size`**: `int`, default=None  
  The number of randomly sampled nodes to compute the ECT for. If `None`, the ECT is computed for all nodes.

### Returns:
- **`ect`**: `torch.Tensor`  
  A tensor of shape `(num_nodes, NUM_THETAS * NUM_THETAS)`, containing the ECT feature vectors for each node.

### Example:
```python
dataset = WebKB(root='/tmp/Wisconsin', name='Wisconsin')
ect_features = compute_local_ect(dataset, radius=1, ECT_TYPE='points', NUM_THETAS=64)
```

---

## Function: `xgb_model`

### Description:
This function computes the local ECT features for a graph dataset and uses these features along with node attributes to train an XGBoost classifier. It supports evaluation using either accuracy or ROC AUC metrics.

### Parameters:
- **`dataset`**: `torch_geometric` dataset  
  The input graph dataset, which should contain node features, edges, labels, and train/test masks.

- **`radius1`**: `bool`, default=True  
  Whether to compute ECT for 1-hop neighborhoods.

- **`radius2`**: `bool`, default=True  
  Whether to compute ECT for 2-hop neighborhoods.

- **`ECT_TYPE`**: `str`, default='points'  
  Type of structural information used for the ECT calculation. See `compute_local_ect` for details.

- **`NUM_THETAS`**: `int`, default=64  
  The dimensionality of the ECT approximation.

- **`DEVICE`**: `str`, default='cpu'  
  The device to be used for computation (`'cpu'` or `'cuda'`).

- **`metric`**: `str`, default='accuracy'  
  The metric for evaluating the classifier. Options:
  - `'accuracy'`: Evaluates model performance using accuracy.
  - `'roc'`: Evaluates model performance using ROC AUC score.

- **`subsample_size`**: `int`, default=None  
  The number of randomly sampled nodes to compute the ECT for. If `None`, the model is trained using all nodes.

### Returns:
- **`acc` or `roc`**: `float`  
  The classification performance evaluated using the chosen metric.

### Example:
```python
dataset = WebKB(root='/tmp/Wisconsin', name='Wisconsin')
accuracy = xgb_model(dataset, radius1=True, radius2=False, ECT_TYPE='points', metric='accuracy')


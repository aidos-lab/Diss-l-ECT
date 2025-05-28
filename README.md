# Diss-l-ECT: Dissecting Graph Data with local Euler Characteristic Transforms

[![GitHub contributors](https://img.shields.io/github/contributors/aidos-lab/Diss-l-ECT)](https://github.com/aidos-lab/Diss-l-ECT/graphs/contributors) [![License](https://img.shields.io/github/license/aidos-lab/Diss-l-ECT)](/LICENSE.md)

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
```

---

## Overview of rotation_invariant_metric.py

This module implements a rotation-invariant metric utilizing the ECT (Euler Characteristic Transform). It includes functionality for computing the ECT, defining a neural network model, and performing operations for spatial alignment of two data spaces.

### Dependencies

This module requires the following libraries:

- `torch`: The core PyTorch library for tensor operations and neural network functionalities.
- `torch_geometric`: A library for geometric deep learning that extends PyTorch with tools for handling graph-structured data.
- `geotorch`: A library for geometric deep learning and optimization on manifolds.
- `sklearn`: For KDTree implementation to facilitate neighbor searches.

### Constants

- `dim`: Dimensionality of the input and output features (default is 768).
- `in_size`: Input size, equal to `dim`.
- `out_size`: Output size, equal to `dim`.
- `DEVICE`: Device to run the computations, defaults to `'cpu'`.

## Functions

### `compute_ect(X, NUM_THETAS=64)`

Computes the ECT transformation of the input tensor `X`.

**Parameters:**
- `X` (Tensor): The input tensor with features to be transformed.
- `NUM_THETAS` (int): Number of theta values for the ECT layer (default is 64).

**Returns:**
- Tensor: The transformed tensor after applying the ECT layer.

### `rotation_inv_distance(X, X_rot)`

Calculates the rotation invariant distance between the input tensor `X` and its rotated version `X_rot` using the orthogonal model.

**Parameters:**
- `X` (Array-like): The original input data.
- `X_rot` (Array-like): The rotated version of the input data.

**Returns:**
- Tuple: A tuple containing:
  - The loss value as a tensor.
  - The reconstructed input tensor after transformations.
  - The weight matrix of the linear transformation.

## Classes

### `OrthogonalModel`

A neural network model that applies an orthogonal transformation to input data.

#### Methods

- **`__init__()`**
  Initializes the model with a linear layer. The weights of the layer are initialized to an identity matrix, and geometric constraints for orthogonality are applied.

- **`forward(x)`**
  Performs a forward pass through the model, applying the linear transformation and computing the ECT transformation.

**Parameters:**
- `x` (Tensor): Input tensor of shape `(batch_size, in_size)`.

**Returns:**
- Tensor: The output after applying the linear transformation and ECT layer.

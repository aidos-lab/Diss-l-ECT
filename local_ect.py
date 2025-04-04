import torch
from torch_geometric.data import Batch, Data
from torch_geometric.datasets import Planetoid, HeterophilousGraphDataset, Amazon, Reddit, WebKB, WikipediaNetwork, Actor, LINKXDataset, WikiCS, Coauthor
from torch.utils.data import TensorDataset, DataLoader, random_split

import numpy as np
from torch_geometric.utils import k_hop_subgraph
from matplotlib import pyplot

from layers.ect import EctLayer
from layers.config import EctConfig


from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score
import xgboost as xgb


def get_class_ratios(labels):
    class_counts = np.bincount(labels)
    class_ratios = class_counts / class_counts.sum()

    return class_ratios


def compute_local_ect(dataset,
                      radius=1,
                      ECT_TYPE='points',
                      NUM_THETAS = 64,
                      DEVICE = 'cpu',
                      subsample_idx=None
):
    '''
    dataset: pytorch geometric graph dataset
    radius: number of hops for the local graph neighborhoods (i.e. `k` in the `k_hop_subgraph` function)
    ECT_TYPE: type of structural information used for the ECT calculation; can be 'points', 'edges' or 'faces'
    NUM_THETAS: the approximation parameter for the resulting ECT; the computation outputs a NUM_THETAS*NUM_THETAS dimensional vector.
    DEVICE: device to be used for the computation
    subsample_idx: indices of randomly sampled nodes in `dataset` to compute the local ECT for; default is None which means that local ECT is determined for all nodes in the input graph.
    '''

    data = dataset[0]
    features = data.x
    if subsample_idx is not None:
        sub_nodes = np.array(range(len(data.x)))[subsample_idx]
    else:
        sub_nodes = np.array(range(len(data.x)))

    batch = Batch.from_data_list(
        [
            Data(x=(data.x)[k_hop_subgraph(int(i), radius, data.edge_index, relabel_nodes=True)[0]],
                 edge_index=k_hop_subgraph(int(i), radius, data.edge_index, relabel_nodes=True)[1])
            for i in list(sub_nodes)
        ]
    ).to(DEVICE)

    CONFIG = EctConfig(num_thetas=NUM_THETAS, bump_steps=NUM_THETAS,
                       normalized=True, device=DEVICE, num_features=features.shape[1], ect_type=ECT_TYPE)

    ectlayer = EctLayer(config=CONFIG)

    ect = ectlayer(batch)
    ect = ect.reshape(ect.shape[0], ect.shape[1] * ect.shape[2])

    return ect


def xgb_model(dataset,
              radius1=True,
              radius2=True,
              ECT_TYPE='points',
              NUM_THETAS = 64,
              DEVICE = 'cpu',
              metric='accuracy',
              subsample_size=None,
              seed=None
):
    '''
    dataset: pytorch geometric graph dataset
    radius1: if True, compute local ECT w.r.t. 1-hop neighborhoods
    radius2: if True, compute local ECT w.r.t. 2-hop neighborhoods
    ECT_TYPE: type of structural information used for the ECT calculation; can be 'points', 'edges' or 'faces'
    NUM_THETAS: the approximation parameter for the resulting ECT; the computation outputs a NUM_THETAS*NUM_THETAS dimensional vector.
    DEVICE: device to be used for the computation
    metric: choose metric for the evaluation of the classification; can be either `accuracy` or `roc'.
    '''
    data = dataset[0]
    all_labels = data.y
    features = data.x
    try:
        if (len(data.train_mask.shape)>1)&(len(data.test_mask.shape)>1):
            train_mask = data.train_mask[:,0]
            test_mask = data.test_mask[:,0]
        elif (len(data.train_mask.shape)>1)&(not(len(data.test_mask.shape)>1)):
            train_mask = data.train_mask[:, 0]
            test_mask = data.test_mask
        else:
            train_mask = data.train_mask
            test_mask = data.test_mask
    except AttributeError:
        bool_list = [True,False]
        p = [.75,.25]
        train_mask = np.random.choice(bool_list,len(data.x),p=p)
        test_mask = [not x for x in train_mask]

    idx = None

    if subsample_size is not None:
        if seed is not None:
            np.random.seed(seed)

        idx = np.random.choice(
            range(len(data.x)),
            replace=False,
            size=subsample_size,
        )

        all_labels = all_labels[idx]
        features = features[idx]
        train_mask = train_mask[idx]
        test_mask = test_mask[idx]

    if radius1:
        ect = compute_local_ect(dataset,
                                 radius=1,
                                 ECT_TYPE=ECT_TYPE,
                                 NUM_THETAS=NUM_THETAS,
                                 DEVICE=DEVICE,
                                 subsample_idx=idx)

        ect_train = ect[train_mask]
        ect_test = ect[test_mask]

    if radius2:
        ect = compute_local_ect(dataset,
                              radius=2,
                              ECT_TYPE=ECT_TYPE,
                              NUM_THETAS=NUM_THETAS,
                              DEVICE=DEVICE,
                              subsample_idx=idx)
        ect_train_2 = ect[train_mask]
        ect_test_2 = ect[test_mask]

    sub_labels = np.array(all_labels)
    sub_features = features

    train = sub_features[train_mask]
    test = sub_features[test_mask]

    if (radius1 and radius2):
        train = torch.cat((ect_train, ect_train_2,train), 1)
        test = torch.cat((ect_test,ect_test_2,test),1)
    elif radius1:
        train = torch.cat((ect_train, train), 1)
        test = torch.cat((ect_test, test), 1)
    elif radius2:
        train = torch.cat((ect_train_2, train), 1)
        test = torch.cat((ect_test_2, test), 1)

    train_labels = sub_labels[train_mask]
    train_labels = torch.tensor(train_labels)
    test_labels = sub_labels[test_mask]
    test_labels = torch.tensor(test_labels)
    # Train an XGBoost model


    le = LabelEncoder()
    train_labels = le.fit_transform(train_labels)
    test_labels = le.fit_transform(test_labels)

    n_classes = len(le.classes_)

    objective = "multi:softprob" if n_classes > 2 else "binary:logistic"
    eval_metric = "auc" if n_classes > 2 else "error"

    if n_classes == 2:
        scale_pos_weight = get_class_ratios(all_labels)
        scale_pos_weight = scale_pos_weight[0] / scale_pos_weight[1]
    else:
        scale_pos_weight = None

    model = xgb.XGBClassifier(objective=objective,
                              eval_metric=eval_metric,
                              scale_pos_weight=scale_pos_weight,
                              max_depth=5)


    model.fit(train, train_labels)
    # Predict probabilities for the test set
    y_score = model.predict(test)
    print(f'Feature Importance: {model.feature_importances_}')
    # plot

    # pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)
    # pyplot.title('Feature Importances')
    # pyplot.show()
    if metric=='accuracy':
        acc = accuracy_score(test_labels, y_score)
        print(f'Accuracy: {acc:.4f}')
        return acc
    elif metric=='roc':
        roc = roc_auc_score(test_labels, y_score)
        print(f'ROC AUC: {roc:.4f}')
        return roc


def xgb_model_minibatch(
    dataset,
    radius1=True,
    radius2=True,
    ECT_TYPE='points',
    NUM_THETAS=64,
    DEVICE='cpu',
    metric='accuracy',
    subsample_size=None,
    batch_size=64,
    num_epochs=100,
):
    data = dataset[0]
    all_labels = data.y
    features = data.x

    try:
        if (len(data.train_mask.shape) > 1) & (len(data.test_mask.shape) > 1):
            train_mask = data.train_mask[:, 0]
            test_mask = data.test_mask[:, 0]
        elif (len(data.train_mask.shape) > 1) & (not (len(data.test_mask.shape) > 1)):
            train_mask = data.train_mask[:, 0]
            test_mask = data.test_mask
        else:
            train_mask = data.train_mask
            test_mask = data.test_mask
    except AttributeError:
        bool_list = [True, False]
        p = [.75, .25]
        train_mask = np.random.choice(bool_list, len(data.x), p=p)
        test_mask = [not x for x in train_mask]

    if subsample_size is not None:

        print("Class ratios before subsampling:", get_class_ratios(all_labels))

        # FIXME
        #np.random.seed(42)
        idx = np.random.choice(len(data.x), size=subsample_size, replace=False)
        all_labels = all_labels[idx]
        features = features[idx]
        train_mask = train_mask[idx]
        test_mask = test_mask[idx]

        print("Class ratios after subsampling:", get_class_ratios(all_labels))

    print("Featurizing data...")

    ect_features = []
    if radius1:
        ect = compute_local_ect(dataset, radius=1, ECT_TYPE=ECT_TYPE, NUM_THETAS=NUM_THETAS, DEVICE=DEVICE, subsample_size=subsample_size)
        ect_features.append(ect)
    if radius2:
        ect = compute_local_ect(dataset, radius=2, ECT_TYPE=ECT_TYPE, NUM_THETAS=NUM_THETAS, DEVICE=DEVICE, subsample_size=subsample_size)
        ect_features.append(ect)

    print("...finished.")

    features = torch.cat(ect_features + [features], dim=1) if ect_features else features

    train_features = features[train_mask]
    test_features = features[test_mask]
    train_labels = all_labels[train_mask]
    test_labels = all_labels[test_mask]

    # Encode labels
    le = LabelEncoder()
    train_labels = torch.tensor(le.fit_transform(train_labels))
    test_labels = torch.tensor(le.transform(test_labels))

    train_loader = DataLoader(TensorDataset(train_features, train_labels), batch_size=batch_size, shuffle=True)

    scale_pos_weight = get_class_ratios(all_labels)
    scale_pos_weight = scale_pos_weight[0] / scale_pos_weight[1]

    print("scale_pos_weight =", scale_pos_weight)

    booster = None
    for epoch in range(num_epochs):
        all_train_preds = []
        all_train_preds_proba = []
        all_train_labels = []

        for batch_features, batch_labels in train_loader:
            dtrain = xgb.DMatrix(batch_features.numpy(), label=batch_labels.numpy())
            if booster is None:
                booster = xgb.train(
                    params={'objective': 'binary:logistic',
                            'eval_metric': 'auc', 'tree_method': 'hist',
                            'max_depth': 4, 'alpha': 0.5,
                            'min_child_weight': 3, 'subsample': 0.5,
                            'scale_pos_weight': scale_pos_weight},
                    dtrain=dtrain,
                    num_boost_round=3,
                )
            else:
                booster = xgb.train(
                    params={ 'objective': 'binary:logistic',
                            'eval_metric': 'auc', 'tree_method': 'hist',
                            'max_depth': 4,
                            'scale_pos_weight': scale_pos_weight},
                    dtrain=dtrain,
                    num_boost_round=3,
                    xgb_model=booster
                )

            y_pred_batch_proba = booster.predict(dtrain)
            y_pred_batch = (y_pred_batch_proba >= 0.5).astype(int)
            all_train_preds.extend(y_pred_batch.tolist())
            all_train_preds_proba.extend(y_pred_batch_proba.tolist())
            all_train_labels.extend(batch_labels.numpy())

        if metric == "accuracy":
            epoch_acc = accuracy_score(all_train_labels, all_train_preds)
            print(f"Epoch {epoch+1} Train Accuracy: {epoch_acc:.4f}")
        elif metric == "roc":
            epoch_acc = roc_auc_score(all_train_labels, all_train_preds_proba, average="weighted")
            print(f"Epoch {epoch+1} Train ROC: {epoch_acc:.4f}")

    dtest = xgb.DMatrix(test_features.numpy())
    y_pred_proba = booster.predict(dtest)
    y_pred = (y_pred_proba >= 0.5).astype(int)

    if metric == 'accuracy':
        acc = accuracy_score(test_labels.numpy(), y_pred)
        print(f'Accuracy: {acc:.4f}')
        return acc
    elif metric == 'roc':
        #FIXME: roc = roc_auc_score(test_labels.numpy(), y_pred, multi_class='ovr')
        roc = roc_auc_score(test_labels, y_pred_proba, average="weighted")
        print(f'ROC AUC: {roc:.4f}')
        return roc

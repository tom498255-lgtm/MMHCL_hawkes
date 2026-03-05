import torch
import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F


# --- 来源：MMHCL (norm.py & load_data.py) ---

def build_sim(context):
    """
    计算特征相似度
    来源: MMHCL norm.py
    """
    # MMHCL: context_norm = context.div(torch.norm(context, p=2, dim=-1, keepdim=True))
    context_norm = context.div(torch.norm(context, p=2, dim=-1, keepdim=True) + 1e-8)
    sim = torch.mm(context_norm, context_norm.transpose(1, 0))
    return sim


def build_knn_normalized_graph(adj, top_k):
    """
    构建KNN归一化图
    来源: MMHCL norm.py
    """
    device = adj.device
    knn_val, knn_ind = torch.topk(adj, top_k, dim=-1)

    # 构造稀疏矩阵的 indices 和 values
    # MMHCL 原文逻辑: tuple_list = [[row, int(col)] ...]
    # 为了效率，这里使用向量化实现，但逻辑保持一致
    n_nodes = adj.shape[0]
    row = torch.arange(n_nodes, device=device).unsqueeze(1).repeat(1, top_k).flatten()
    col = knn_ind.flatten()

    indices = torch.stack([row, col], dim=0)
    values = knn_val.flatten()

    shape = adj.shape
    return torch.sparse_coo_tensor(indices, values, shape).coalesce()


def scipy_sparse_mat_to_torch_sparse_tensor(sparse_mx):
    """
    Scipy 稀疏矩阵转 Torch 稀疏张量
    来源: MMHCL load_data.py (sparse_mx_to_torch_sparse_tensor)
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def _to_numpy(x):
    if hasattr(x, "to_numpy"):        # pandas
        return x.to_numpy()
    elif torch.is_tensor(x):          # torch
        return x.cpu().numpy()
    else:                             # numpy
        return x

def get_u2u_mat(inter_feat, n_users, n_items):
    """
    构建 U2U 共同交互图
    来源: MMHCL load_data.py (get_U2U_mat 逻辑复现)
    原文使用 A @ A.T
    """
    # 构造交互矩阵 R
    user_np = _to_numpy(inter_feat['user_id'])
    item_np = _to_numpy(inter_feat['item_id'])
    data = np.ones_like(user_np, dtype=np.float32)

    # R: [n_users, n_items]
    R = sp.csr_matrix((data, (user_np, item_np)), shape=(n_users, n_items))

    # R @ R.T 计算共同交互
    U2U = R.dot(R.T)
    U2U.setdiag(0)  # 去除自环

    # 归一化 (Row Normalization) - 参考 MMHCL get_U2U_mat(norm_type='rw')
    rowsum = np.array(U2U.sum(1))
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    norm_U2U = d_mat_inv.dot(U2U)

    return scipy_sparse_mat_to_torch_sparse_tensor(norm_U2U)
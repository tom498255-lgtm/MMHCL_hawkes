import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole_custom.model.abstract_recommender import SequentialRecommender
from recbole_custom.model.loss import BPRLoss
from recbole_custom.model.init import xavier_uniform_initialization

# 引入自定义工具
from recbole_custom.utils.mmhcl_utils import build_sim, build_knn_normalized_graph, get_u2u_mat


# --- 来源：HyperHawkes (hyperhawkes.py SoftKMeans) ---
class SoftKMeans(nn.Module):
    def __init__(self, n_clusters, hidden_size, temp):
        super().__init__()
        self.n_clusters = n_clusters
        self.centers = nn.Parameter(torch.randn(n_clusters, hidden_size))
        self.temp = temp

    def forward(self, x):
        # 计算欧氏距离
        dist = torch.cdist(x, self.centers)
        # 软分配
        soft_assign = F.softmax(-dist / self.temp, dim=-1)
        return soft_assign


class MMHyperHawkes(SequentialRecommender):
    def __init__(self, config, dataset):
        super(MMHyperHawkes, self).__init__(config, dataset)
        self.dataset = dataset
        self.n_users = dataset.num(self.USER_ID)
        self.n_items = dataset.num(self.ITEM_ID)
        # ================= 配置参数 =================
        self.device = config['device']

        # MMHCL 参数
        self.embedding_size = config['embedding_size']
        self.n_ui_layers = config['n_ui_layers']  # GCN 层数
        self.top_k = config['top_k']

        # HyperHawkes 参数
        self.n_clusters = config['n_clusters']  # 意图数量
        self.temp_cluster = config['temp_cluster']
        self.time_scalar = config['time_scalar']  # 时间缩放因子
        self.sub_time_delta = config['sub_time_delta']
        self.emb_dropout_prob = config['emb_dropout_prob']

        # ================= 模型组件初始化 =================

        # 1. Embeddings (MMHCL 风格)
        # MMHCL 使用多个 Embedding，这里我们做融合设计
        # user_ui_embedding / item_ui_embedding 对应基础图嵌入
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)

        # 多模态投影 (New Framework MLP)
        self.image_mlp = nn.Linear(config['image_feat_dim'], self.embedding_size)
        self.text_mlp = nn.Linear(config['text_feat_dim'], self.embedding_size)

        # 2. 意图聚类 (HyperHawkes 风格)
        self.soft_kmeans = SoftKMeans(self.n_clusters, self.embedding_size, self.temp_cluster)
        self.register_buffer('cluster_prob', torch.zeros(self.n_items, self.n_clusters))

        # 3. Hawkes 参数网络 (HyperHawkes 风格)
        # 来源: hyperhawkes.py __init__
        self.global_alpha = nn.Parameter(torch.tensor(0.1))
        self.intent_dist = nn.Sequential(
            nn.Linear(self.n_clusters + self.embedding_size * 2, self.embedding_size),
            nn.ReLU(),
            nn.Linear(self.embedding_size, 5)  # 输出 alpha, beta, mu, sigma, pi
        )

        # 4. 短期兴趣 Attention (HyperHawkes Short-term)
        self.W_q = nn.Linear(self.embedding_size, self.embedding_size)
        self.W_k = nn.Linear(self.embedding_size, self.embedding_size)
        self.LayerNorm = nn.LayerNorm(self.embedding_size)

        # ================= 图构建 (预处理) =================
        # 注意：为了性能，实际通常在 Dataset 中构建，这里为了展示逻辑放在初始化中
        # 假设 dataset 已经加载了 features
        self._build_mmhcl_graphs(dataset)

        self.loss_fct = BPRLoss()
        self.apply(xavier_uniform_initialization)

    def _build_mmhcl_graphs(self, dataset):
        """
        构建 MMHCL 所需的 U2U 和 I2I 图
        """
        # 1. 加载特征 (模拟)
        # image_feats = dataset.image_features ...
        # 这里假设已经存在 self.image_feats, self.text_feats

        # 2. 构建 I2I 超图 (MMHCL Models.py + norm.py)
        # 视觉图
        sim_img = build_sim(dataset.image_features)
        graph_img = build_knn_normalized_graph(sim_img, self.top_k)
        # 文本图
        sim_txt = build_sim(dataset.text_features)
        graph_txt = build_knn_normalized_graph(sim_txt, self.top_k)
        # 融合 I2I (MMHCL 逻辑: torch.cat or mean)
        # 为了 GCN 传播方便，取平均
        self.i2i_mat = (graph_img + graph_txt) / 2
        self.i2i_mat = self.i2i_mat.to(self.device)

        # 3. 构建 U2U 超图 (MMHCL load_data.py)
        # 需要交互数据
        self.u2u_mat = get_u2u_mat(dataset.inter_feat, self.n_users, self.n_items)
        self.u2u_mat = self.u2u_mat.to(self.device)

    # --- 来源：MMHCL (Models.py forward) ---
    def mmhcl_encoder(self):
        """
        图卷积编码器，严格复现 MMHCL 的 sparse.mm 传播逻辑
        """
        # 1. 初始特征融合
        img_emb = self.image_mlp(self.dataset.image_features.to(self.device))
        txt_emb = self.text_mlp(self.dataset.text_features.to(self.device))

        # Base ID Embeddings
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight + img_emb + txt_emb  # 简单的多模态融合

        # 2. U2U 传播 (针对用户)
        # 来源: MMHCL Models.py -> if args.user_loss_ratio != 0: uu_emb = torch.sparse.mm(...)
        u_embs = [user_emb]
        ego_u = user_emb
        for i in range(self.n_ui_layers):
            side_u = torch.sparse.mm(self.u2u_mat, ego_u)
            ego_u = side_u
            u_embs.append(ego_u)
        u_final = torch.stack(u_embs, dim=1).mean(dim=1)

        # 3. I2I 传播 (针对物品)
        # 来源: MMHCL Models.py -> ii_emb = torch.sparse.mm(I2I_mat, ii_emb)
        i_embs = [item_emb]
        ego_i = item_emb
        for i in range(self.n_ui_layers):
            side_i = torch.sparse.mm(self.i2i_mat, ego_i)
            ego_i = side_i
            i_embs.append(ego_i)
        i_final = torch.stack(i_embs, dim=1).mean(dim=1)

        return u_final, i_final

    # --- 来源：HyperHawkes (hyperhawkes.py e_step) ---
    def e_step(self):
        """
        E-M 算法 E步：更新聚类概率
        """
        with torch.no_grad():
            _, item_embs = self.mmhcl_encoder()
            self.cluster_prob = self.soft_kmeans(item_embs.detach())

    # --- 来源：HyperHawkes (hyperhawkes.py intent_excitation) ---
    def get_hawkes_excitation(self, target_item, item_seq, time_seq, target_time, user_rep):
        """
        Hawkes-based intent excitation (fixed broadcast version)

        target_item: [B]
        item_seq:    [B, L]
        time_seq:    [B, L]
        target_time: [B]
        user_rep:    [B, D]
        """

        B, L = item_seq.size()

        # ===== 1. Intent (cluster) probabilities =====
        target_cluster_probs = self.cluster_prob[target_item]  # [B, K]
        seq_cluster_probs = self.cluster_prob[item_seq]  # [B, L, K]

        # ===== 2. KL-divergence intent matching =====
        kl_div = F.kl_div(
            seq_cluster_probs.log(),
            target_cluster_probs.unsqueeze(1).log(),
            reduction='none',
            log_target=True
        ).sum(dim=2)  # [B, L]

        intent_mask = (kl_div < 1e-12) & (item_seq > 0)  # [B, L]

        # ===== 3. Time difference =====
        delta_t = target_time.unsqueeze(1) - time_seq  # [B, L]
        delta_mask = (delta_t > self.sub_time_delta) & intent_mask
        delta_t = (delta_t / self.time_scalar) * delta_mask.float()  # [B, L]

        mask = (delta_t > 0).float()  # [B, L]

        # ===== 4. Distribution parameter prediction =====
        target_item_emb = self.item_embedding(target_item)  # [B, D]

        dist_input = torch.cat(
            (target_cluster_probs, target_item_emb, user_rep), dim=1
        )  # [B, *]

        dist_params = self.intent_dist(dist_input)  # [B, 5]

        mus = dist_params[:, 0].clamp(1e-10, 10).unsqueeze(1)  # [B, 1]
        sigmas = dist_params[:, 1].clamp(1e-10, 10).unsqueeze(1)  # [B, 1]
        alphas = (self.global_alpha + dist_params[:, 2]).unsqueeze(1)  # [B, 1]
        betas = (dist_params[:, 3] + 1).clamp(1e-10, 10).unsqueeze(1)  # [B, 1]
        pis = (dist_params[:, 4] + 0.5).clamp(1e-10, 1).unsqueeze(1)  # [B, 1]

        # ===== 5. Hawkes kernel mixture =====
        exp_dist = torch.distributions.Exponential(betas)
        norm_dist = torch.distributions.Normal(mus, sigmas)

        excitation = (
                pis * exp_dist.log_prob(delta_t + 1e-9).exp()
                + (1 - pis) * norm_dist.log_prob(delta_t + 1e-9).exp()
        )  # [B, L]

        excitation = alphas * excitation * mask  # [B, L]

        return excitation.sum(dim=1)  # [B]

    # --- 来源：HyperHawkes (hyperhawkes.py short-term attention) ---
    def get_short_term_rep(self, item_seq_emb, mask):
        # 多头注意力的简化实现 (Q, K projection)
        Q = self.W_q(item_seq_emb)
        K = self.W_k(item_seq_emb)
        V = item_seq_emb

        # Scaled Dot-Product
        attn_scores = torch.matmul(Q, K.transpose(-1, -2)) / (self.embedding_size ** 0.5)
        attn_scores = attn_scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        attn_weights = F.softmax(attn_scores, dim=-1)

        output = torch.matmul(attn_weights, V)
        return self.LayerNorm(output.mean(dim=1))  # Mean pooling of attention output

    def forward(self, item_seq, item_seq_len, target_item, time_seq, target_time):
        # 1. 获取图增强表示 (MMHCL Encoder)
        u_g_embeddings, i_g_embeddings = self.mmhcl_encoder()

        # 获取 Batch 数据
        seq_emb = i_g_embeddings[item_seq]  # [B, L, D]
        target_emb = i_g_embeddings[target_item]  # [B, D]

        # 2. 短期兴趣 (HyperHawkes Short-term)
        mask = (item_seq > 0).float()
        short_term_rep = self.get_short_term_rep(seq_emb, mask)

        # 3. 基础强度 AURA (Base Intensity)
        # 用 user_embedding (经 U2U 增强) 代表用户长期静态偏好
        # 这里为了简化，假设 batch 内每个序列对应一个用户，实际应用需传入 user_id
        # 暂时用 short_term_rep 作为 User Query 的近似
        base_score = torch.mul(short_term_rep, target_emb).sum(dim=1)

        # 4. 意图激发 (Hawkes Process)
        # 假设我们有 user_rep (可以是 short_term_rep + static user emb)
        hawkes_score = self.get_hawkes_excitation(target_item, item_seq, time_seq, target_time, short_term_rep)

        # 5. 融合分数
        final_score = base_score + hawkes_score
        return final_score

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        pos_items = interaction[self.POS_ITEM_ID]
        neg_items = interaction[self.NEG_ITEM_ID]

        # 获取时间信息 (RecBole 字段)
        time_seq = interaction['timestamp_list']  # 假设 dataset 处理了时间序列
        target_time = interaction['timestamp']

        pos_score = self.forward(item_seq, item_seq_len, pos_items, time_seq, target_time)
        neg_score = self.forward(item_seq, item_seq_len, neg_items, time_seq, target_time)

        return self.loss_fct(pos_score, neg_score)

    def predict(self, interaction):
        # 与 calculate_loss 类似，只计算 pos_item (即 target item)
        pass

    def full_sort_predict(self, interaction):
        """
        Full sort prediction for validation / test phase
        Return scores: [B, n_items]
        """

        device = self.device

        # ===== 1. Required fields =====
        item_seq = interaction[self.ITEM_SEQ]  # [B, L]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]  # [B]
        time_seq = interaction['timestamp_list']  # [B, L]
        target_time = interaction['timestamp']  # [B]

        B, L = item_seq.size()
        n_items = self.n_items

        # ===== 2. MMHCL graph encoder (same as forward) =====
        _, i_g_embeddings = self.mmhcl_encoder()  # [n_items, D]

        # ===== 3. Short-term user representation =====
        seq_emb = i_g_embeddings[item_seq]  # [B, L, D]
        mask = (item_seq > 0).float()  # [B, L]
        short_term_rep = self.get_short_term_rep(seq_emb, mask)  # [B, D]

        # ===== 4. Base score for all items (vectorized) =====
        # short_term_rep: [B, D]
        # i_g_embeddings: [n_items, D]
        base_scores = torch.matmul(
            short_term_rep, i_g_embeddings.transpose(0, 1)
        )  # [B, n_items]

        # ===== 5. Hawkes excitation (vectorized over items) =====
        # Expand tensors for broadcasting
        item_seq_expand = item_seq.unsqueeze(1).expand(B, n_items, L)  # [B, n_items, L]
        time_seq_expand = time_seq.unsqueeze(1).expand(B, n_items, L)  # [B, n_items, L]
        target_time_expand = target_time.unsqueeze(1).expand(B, n_items)  # [B, n_items]

        # Flatten for Hawkes computation
        flat_item_seq = item_seq_expand.reshape(-1, L)  # [B*n_items, L]
        flat_time_seq = time_seq_expand.reshape(-1, L)  # [B*n_items, L]
        flat_target_time = target_time_expand.reshape(-1)  # [B*n_items]
        flat_user_rep = short_term_rep.unsqueeze(1).expand(
            B, n_items, -1
        ).reshape(-1, self.embedding_size)  # [B*n_items, D]

        flat_target_item = torch.arange(
            n_items, device=device
        ).unsqueeze(0).expand(B, -1).reshape(-1)  # [B*n_items]

        hawkes_scores = self.get_hawkes_excitation(
            target_item=flat_target_item,
            item_seq=flat_item_seq,
            time_seq=flat_time_seq,
            target_time=flat_target_time,
            user_rep=flat_user_rep
        ).view(B, n_items)  # [B, n_items]

        # ===== 6. Final score =====
        scores = base_scores + hawkes_scores

        # Padding item (id=0) should never be recommended
        scores[:, 0] = -1e12

        return scores

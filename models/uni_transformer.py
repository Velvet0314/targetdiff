import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import radius_graph, knn_graph
from torch_scatter import scatter_softmax, scatter_sum

from models.common import GaussianSmearing, MLP, batch_hybrid_edge_connection, outer_product

# 基于图注意力机制的节点特征更新层
class BaseX2HAttLayer(nn.Module):
    """
    将 x 特征转换为 h 特征的注意力层 —— 将空间信息(X)转换为隐含特征(H)
    x: 3D坐标信息
    h: 节点特征
    """
    def __init__(self, input_dim, hidden_dim, output_dim, n_heads, edge_feat_dim, r_feat_dim,
                 act_fn='relu', norm=True, ew_net_type='r', out_fc=True):
        """初始化注意力层参数和网络结构"""
        super().__init__()
        self.input_dim = input_dim  # 输入特征维度
        self.hidden_dim = hidden_dim  # 隐层维度
        self.output_dim = output_dim  # 输出维度
        self.n_heads = n_heads  # 注意力头数
        self.act_fn = act_fn  # 激活函数
        self.edge_feat_dim = edge_feat_dim  # 边特征维度
        self.r_feat_dim = r_feat_dim  # 距离特征维度
        self.ew_net_type = ew_net_type  # 边权重网络类型
        self.out_fc = out_fc  # 是否使用输出全连接层

        # MLP：多层感知机

        # attention key func

        # 计算Key和Value的输入维度:
        # - input_dim * 2: 源节点和目标节点的特征
        # - edge_feat_dim: 边特征维度
        # - r_feat_dim: 距离特征维度
        kv_input_dim = input_dim * 2 + edge_feat_dim + r_feat_dim 
        # Key生成网络:将边信息映射为注意力键
        # 输入: 节点对特征、边特征、距离特征的拼接
        # 输出: output_dim维度的Key向量
        self.hk_func = MLP(kv_input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)

        # attention value func

        # Value生成网络:将边信息映射为注意力值
        # 结构与Key网络相同,但学习不同的映射关系
        self.hv_func = MLP(kv_input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)

        # attention query func

        # Query生成网络:将单个节点特征映射为查询向量
        # 输入维度较小,只包含节点自身特征
        self.hq_func = MLP(input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)
        
        # 边权重预测网络:为每条边分配重要性权重
        if ew_net_type == 'r':
            # 基于距离特征的边权重
            # 将r_feat_dim维的距离特征映射为标量权重并归一化到[0,1]
            self.ew_net = nn.Sequential(nn.Linear(r_feat_dim, 1), nn.Sigmoid())
        elif ew_net_type == 'm':
            # 基于节点消息的边权重
            # 将output_dim维的节点消息映射为标量权重
            self.ew_net = nn.Sequential(nn.Linear(output_dim, 1), nn.Sigmoid())

        if self.out_fc:
            # 输出变换层:融合更新后的特征与原始特征
            # 将拼接的2*hidden_dim维特征映射回hidden_dim维
            self.node_output = MLP(2 * hidden_dim, hidden_dim, hidden_dim, norm=norm, act_fn=act_fn)

    def forward(self, h, r_feat, edge_feat, edge_index, e_w=None):
        """
        前向传播
        Args:
            h: 节点特征
            r_feat: 距离特征
            edge_feat: 边特征
            edge_index: 边的连接信息
            e_w: 预定义的边权重
        """
        N = h.size(0)  # 获取图中节点总数
        src, dst = edge_index  # 解析源节点(src)和目标节点(dst)的索引
        hi, hj = h[dst], h[src]  # 获取目标节点和源节点的特征向量

        # multi-head attention
        # decide inputs of k_func and v_func

        # 第一步:准备注意力计算的输入
        # 将距离特征、目标节点特征、源节点特征拼接
        # [num_edges, r_feat_dim + input_dim * 2]
        kv_input = torch.cat([r_feat, hi, hj], -1)  # 组合距离特征和节点特征
        if edge_feat is not None:
            # 如果存在边特征,则将其加入拼接
            # [num_edges, edge_feat_dim + r_feat_dim + input_dim * 2]
            kv_input = torch.cat([edge_feat, kv_input], -1)  # 添加边特征

        # 第二步:计算注意力机制的Key和Value

        # compute k
        # Key: 将边信息转换为查询键 [num_edges, n_heads, output_dim//n_heads]
        k = self.hk_func(kv_input).view(-1, self.n_heads, self.output_dim // self.n_heads)
        
        # compute v 
        v = self.hv_func(kv_input)

        # 第三步:计算边权重
        if self.ew_net_type == 'r':
            # 使用距离特征计算边权重
            e_w = self.ew_net(r_feat)  # [num_edges, 1]
        elif self.ew_net_type == 'm':
            # 使用节点消息计算边权重
            e_w = self.ew_net(v[..., :self.hidden_dim])  # [num_edges, 1]
        elif e_w is not None:
            # 使用预定义的边权重
            e_w = e_w.view(-1, 1)
        else:
            # 默认边权重为1
            e_w = 1.
        
        # 应用边权重到值向量
        v = v * e_w # [num_edges, output_dim]
        # 重塑值向量用于多头注意力 [num_edges, n_heads, output_dim//n_heads]
        v = v.view(-1, self.n_heads, self.output_dim // self.n_heads)

        # compute q
        # 第四步:计算Query并得到注意力权重
        # Query: 将节点特征转换为查询向量 [num_nodes, n_heads, output_dim//n_heads]
        q = self.hq_func(h).view(-1, self.n_heads, self.output_dim // self.n_heads)

        # compute attention weights
        # 计算注意力分数并进行softmax归一化
        alpha = scatter_softmax((q[dst] * k / np.sqrt(k.shape[-1])).sum(-1), dst, dim=0,
                                dim_size=N)  # [num_edges, n_heads]

        # 第五步:消息传递和特征聚合
        # 将注意力权重应用到值向量 [num_edges, n_heads, output_dim//n_heads]
        
        # perform attention-weighted message-passing
        m = alpha.unsqueeze(-1) * v  # (E, heads, H_per_head)
        # 聚合来自相邻节点的消息 [num_nodes, n_heads, output_dim//n_heads]
        output = scatter_sum(m, dst, dim=0, dim_size=N)  # (N, heads, H_per_head)
        # 重塑输出特征 [num_nodes, output_dim]
        output = output.view(-1, self.output_dim)
        
        # 第六步:特征转换和残差连接
        if self.out_fc:
            # 将更新特征与原始特征拼接并转换
            output = self.node_output(torch.cat([output, h], -1))
        # 添加残差连接以缓解梯度消失问题
        output = output + h
        
        return output


class BaseH2XAttLayer(nn.Module):
    """
    将节点特征(H)映射到3D空间坐标(X)的注意力层
    实现了从节点特征到空间位置的更新
    这个过程允许模型根据节点的隐含特征动态调整其在3D空间中的位置
    """
    def __init__(self, input_dim, hidden_dim, output_dim, n_heads, edge_feat_dim, r_feat_dim,
                 act_fn='relu', norm=True, ew_net_type='r'):
        super().__init__()
        self.input_dim = input_dim  # 输入节点特征维度,表示每个节点的特征向量长度
        self.hidden_dim = hidden_dim    # 隐藏层维度,用于信息转换的中间表示
        self.output_dim = output_dim    # 输出维度(通常为3,对应xyz坐标),表示3D空间中的位置信息
        self.n_heads = n_heads  # 注意力头数,多头注意力机制用于捕获不同方面的特征关系
        self.edge_feat_dim = edge_feat_dim  # 边特征维度,描述节点间关系的特征向量长度
        self.r_feat_dim = r_feat_dim    # 距离特征维度,编码节点间空间距离信息的特征长度
        self.act_fn = act_fn    # 激活函数类型,用于引入非线性变换
        # norm: 是否使用归一化,用于稳定训练过程
        self.ew_net_type = ew_net_type  # 边权重网络类型('r':基于距离特征计算/'m':基于节点消息计算)

        # 计算注意力机制的输入维度
        # 包含源节点特征、目标节点特征、边特征和距离特征的组合
        kv_input_dim = input_dim * 2 + edge_feat_dim + r_feat_dim

        # 定义三个核心转换网络 - 多层感知机(MLP)结构

        # Key网络:边信息->注意力键
        # 将边的综合信息转换为用于计算注意力分数的键向量
        self.xk_func = MLP(kv_input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)
        
        # Value网络:边信息->坐标偏移缩放因子
        # 将边信息转换为用于调整相对坐标的缩放系数
        self.xv_func = MLP(kv_input_dim, self.n_heads, hidden_dim, norm=norm, act_fn=act_fn)
        
        # Query网络:节点特征->查询向量
        # 将节点特征转换为用于计算注意力分数的查询向量
        self.xq_func = MLP(input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)
        
        # 边权重网络(可选):用于计算边的重要性权重
        if ew_net_type == 'r':
            # 将距离特征转换为标量权重并通过Sigmoid归一化到[0,1]
            self.ew_net = nn.Sequential(nn.Linear(r_feat_dim, 1), nn.Sigmoid())

    def forward(self, h, rel_x, r_feat, edge_feat, edge_index, e_w=None):
        """
        前向传播:更新节点的空间坐标
        Args:
            h: 节点特征矩阵 [num_nodes, input_dim]
            rel_x: 相对坐标矩阵 (xi - xj),表示节点间的相对位置 [num_edges, 3]
            r_feat: 距离特征矩阵,编码节点间的距离信息 [num_edges, r_feat_dim]
            edge_feat: 边特征矩阵,描述节点间的关系 [num_edges, edge_feat_dim]
            edge_index: 边的连接关系,存储源节点和目标节点的索引 [2, num_edges]
            e_w: 预定义边权重(可选),用于手动控制边的重要性
        Returns:
            Tensor: 节点的坐标偏移量,表示每个节点在3D空间中的位置调整量 [num_nodes, 3]
        """

        N = h.size(0)   # 节点数量,即图中节点的总数
        src, dst = edge_index   # 源节点和目标节点索引,用于建立节点间的连接关系
        hi, hj = h[dst], h[src] # 获取相连节点的特征,hi为目标节点特征,hj为源节点特征

        # multi-head attention
        # decide inputs of k_func and v_func
        kv_input = torch.cat([r_feat, hi, hj], -1)  # 组合距离特征和节点特征,形成边的综合表示
        if edge_feat is not None:
            kv_input = torch.cat([edge_feat, kv_input], -1) # 添加边特征,进一步丰富边的表示

        # 计算注意力机制的Key和Value
        # Key用于计算注意力分数,维度重组用于多头注意力机制
        k = self.xk_func(kv_input).view(-1, self.n_heads, self.output_dim // self.n_heads)
        # Value用于生成坐标偏移的缩放因子
        v = self.xv_func(kv_input)

        # 计算边权重,用于调整边的重要性
        if self.ew_net_type == 'r':
             # 基于距离特征计算边权重,反映空间距离的影响
            e_w = self.ew_net(r_feat)
        elif self.ew_net_type == 'm':
            # 不使用边权重,所有边具有相同的重要性
            e_w = 1.
        elif e_w is not None:
            # 使用预定义权重,允许外部控制边的重要性
            e_w = e_w.view(-1, 1)
        else:
            e_w = 1.
        # 应用边权重到Value向量
        v = v * e_w

        # 计算坐标偏移量
        # 将Value向量与相对坐标相乘,生成方向性的坐标调整
        v = v.unsqueeze(-1) * rel_x.unsqueeze(1)  # (xi - xj) [n_edges, n_heads, 3]
        # 计算Query向量,用于后续的注意力权重计算
        q = self.xq_func(h).view(-1, self.n_heads, self.output_dim // self.n_heads)

        # Compute attention weights
        # 计算注意力权重,使用点积注意力机制并进行softmax归一化
        alpha = scatter_softmax((q[dst] * k / np.sqrt(k.shape[-1])).sum(-1), dst, dim=0, dim_size=N)  # (E, heads)

        # Perform attention-weighted message-passing
        # 加权聚合坐标偏移,将注意力权重应用到坐标调整量上
        m = alpha.unsqueeze(-1) * v  # (E, heads, 3)
        # 汇总所有边对每个节点的贡献,得到最终的坐标偏移
        output = scatter_sum(m, dst, dim=0, dim_size=N)  # (N, heads, 3)
        # 返回平均坐标偏移,即所有注意力头的平均结果
        return output.mean(1)  # [num_nodes, 3]


class AttentionLayerO2TwoUpdateNodeGeneral(nn.Module):
    """
    实现基于O2对称性的双向注意力机制
    """
    def __init__(self, hidden_dim, n_heads, num_r_gaussian, edge_feat_dim, act_fn='relu', norm=True,
                 num_x2h=1, num_h2x=1, r_min=0., r_max=10., num_node_types=8,
                 ew_net_type='r', x2h_out_fc=True, sync_twoup=False):
        """初始化模型参数和网络结构"""
        super().__init__()
        # 模型基础参数
        self.hidden_dim = hidden_dim          # 隐藏层维度
        self.n_heads = n_heads                # 注意力头数
        self.edge_feat_dim = edge_feat_dim    # 边特征维度
        self.num_r_gaussian = num_r_gaussian  # 高斯核数量
        self.norm = norm                      # 是否使用归一化
        self.act_fn = act_fn                  # 激活函数类型
        self.num_x2h = num_x2h                # X->H转换层数
        self.num_h2x = num_h2x                # H->X转换层数
        self.r_min, self.r_max = r_min, r_max # 距离范围
        self.num_node_types = num_node_types  # 节点类型数
        self.ew_net_type = ew_net_type        # 边权重网络类型
        self.x2h_out_fc = x2h_out_fc          # 是否使用X2H输出全连接层
        self.sync_twoup = sync_twoup          # 是否同步更新

        # 距离编码层
        self.distance_expansion = GaussianSmearing(self.r_min, self.r_max, num_gaussians=num_r_gaussian)

        # X->H特征转换层
        self.x2h_layers = nn.ModuleList()
        for i in range(self.num_x2h):
            self.x2h_layers.append(
                BaseX2HAttLayer(hidden_dim, hidden_dim, hidden_dim, n_heads, edge_feat_dim,
                                r_feat_dim=num_r_gaussian * 4,    # 4种边类型的距离特征
                                act_fn=act_fn, norm=norm,
                                ew_net_type=self.ew_net_type, out_fc=self.x2h_out_fc)
            )

        # H->X坐标更新层
        self.h2x_layers = nn.ModuleList()
        for i in range(self.num_h2x):
            self.h2x_layers.append(
                BaseH2XAttLayer(hidden_dim, hidden_dim, hidden_dim, n_heads, edge_feat_dim,
                                r_feat_dim=num_r_gaussian * 4,    # 4种边类型的距离特征
                                act_fn=act_fn, norm=norm,
                                ew_net_type=self.ew_net_type)
            )

    def forward(self, h, x, edge_attr, edge_index, mask_ligand, e_w=None, fix_x=False):
        # 获取边的源节点和目标节点索引
        src, dst = edge_index
        # 处理边特征
        if self.edge_feat_dim > 0:
            edge_feat = edge_attr  # shape: [#edges_in_batch, #bond_types]
        else:
            edge_feat = None

        # 计算相对空间信息
        rel_x = x[dst] - x[src]  # 相对坐标
        dist = torch.norm(rel_x, p=2, dim=-1, keepdim=True)  # 欧氏距离

        # X->H特征更新阶段
        h_in = h
        # 4 separate distance embedding for p-p, p-l, l-p, l-l
        for i in range(self.num_x2h):
            dist_feat = self.distance_expansion(dist)  # 距离编码
            dist_feat = outer_product(edge_attr, dist_feat)  # 边类型与距离特征的外积
            h_out = self.x2h_layers[i](h_in, dist_feat, edge_feat, edge_index, e_w=e_w)
            h_in = h_out
        x2h_out = h_in

        # H->X坐标更新阶段
        new_h = h if self.sync_twoup else x2h_out
        for i in range(self.num_h2x):
            dist_feat = self.distance_expansion(dist)  # 距离编码
            dist_feat = outer_product(edge_attr, dist_feat)  # 边类型与距离特征的外积
            delta_x = self.h2x_layers[i](new_h, rel_x, dist_feat, edge_feat, edge_index, e_w=e_w)
            if not fix_x:
                x = x + delta_x * mask_ligand[:, None]  # only ligand positions will be updated
            rel_x = x[dst] - x[src]  # 更新相对坐标
            dist = torch.norm(rel_x, p=2, dim=-1, keepdim=True)  # 更新距离

        return x2h_out, x  # 返回更新后的特征和坐标


class UniTransformerO2TwoUpdateGeneral(nn.Module):
    def __init__(self, num_blocks, num_layers, hidden_dim, n_heads=1, k=32,
                 num_r_gaussian=50, edge_feat_dim=0, num_node_types=8, act_fn='relu', norm=True,
                 cutoff_mode='radius', ew_net_type='r',
                 num_init_x2h=1, num_init_h2x=0, num_x2h=1, num_h2x=1, r_max=10., x2h_out_fc=True, sync_twoup=False):
        super().__init__()
        # Build the network
        self.num_blocks = num_blocks  # 模型块数,每个块包含多个注意力层
        self.num_layers = num_layers  # 每个块中的层数,控制特征提取深度
        self.hidden_dim = hidden_dim  # 隐藏层维度,决定特征表示能力
        self.n_heads = n_heads  # 注意力头数,增加特征捕获能力
        self.num_r_gaussian = num_r_gaussian  # 高斯核数量,用于距离编码
        self.edge_feat_dim = edge_feat_dim  # 边特征维度
        self.act_fn = act_fn  # 激活函数类型
        self.norm = norm  # 是否使用归一化
        self.num_node_types = num_node_types  # 节点类型数量

        # radius graph / knn graph
        self.cutoff_mode = cutoff_mode  # [radius, none] - 图构建方式
        self.k = k  # KNN图中的邻居数
        self.ew_net_type = ew_net_type  # [r, m, none] - 边权重计算方式

        # X->H和H->X转换层配置
        self.num_x2h = num_x2h  # 特征提取层数
        self.num_h2x = num_h2x  # 坐标更新层数 
        self.num_init_x2h = num_init_x2h  # 初始特征提取层数
        self.num_init_h2x = num_init_h2x  # 初始坐标更新层数
        self.r_max = r_max  # 最大截断距离
        self.x2h_out_fc = x2h_out_fc  # 是否使用X->H输出全连接层
        self.sync_twoup = sync_twoup  # 是否同步更新特征和坐标

        # 距离编码层
        self.distance_expansion = GaussianSmearing(0., r_max, num_gaussians=num_r_gaussian)

        # 全局边权重预测层(可选)
        if self.ew_net_type == 'global':
            self.edge_pred_layer = MLP(num_r_gaussian, 1, hidden_dim)

        # 构建模型核心组件
        self.init_h_emb_layer = self._build_init_h_layer()  # 初始特征嵌入层
        self.base_block = self._build_share_blocks()  # 共享块结构

    def _build_init_h_layer(self):
        """构建初始特征嵌入层"""
        layer = AttentionLayerO2TwoUpdateNodeGeneral(
            self.hidden_dim, self.n_heads, self.num_r_gaussian, self.edge_feat_dim, act_fn=self.act_fn, norm=self.norm,
            num_x2h=self.num_init_x2h, num_h2x=self.num_init_h2x, r_max=self.r_max, num_node_types=self.num_node_types,
            ew_net_type=self.ew_net_type, x2h_out_fc=self.x2h_out_fc, sync_twoup=self.sync_twoup,
        )
        return layer

    def _build_share_blocks(self):
        # Equivariant layers - 构建等变层
        base_block = []
        for l_idx in range(self.num_layers):
            layer = AttentionLayerO2TwoUpdateNodeGeneral(
                self.hidden_dim, self.n_heads, self.num_r_gaussian, self.edge_feat_dim, act_fn=self.act_fn,
                norm=self.norm,
                num_x2h=self.num_x2h, num_h2x=self.num_h2x, r_max=self.r_max, num_node_types=self.num_node_types,
                ew_net_type=self.ew_net_type, x2h_out_fc=self.x2h_out_fc, sync_twoup=self.sync_twoup,
            )
            base_block.append(layer)
        return nn.ModuleList(base_block)

    def forward(self, h, x, mask_ligand, batch, return_all=False, fix_x=False):
        """前向传播:实现特征提取和坐标预测"""
        # 存储所有中间状态
        all_x = [x]
        all_h = [h]

        # 多个块的迭代更新
        for b_idx in range(self.num_blocks):
            edge_index = self._connect_edge(x, mask_ligand, batch)
            src, dst = edge_index

            # edge type (dim: 4) - 构建边类型编码
            edge_type = self._build_edge_type(edge_index, mask_ligand)
            if self.ew_net_type == 'global':
                # 计算全局边权重
                dist = torch.norm(x[dst] - x[src], p=2, dim=-1, keepdim=True)
                dist_feat = self.distance_expansion(dist)
                logits = self.edge_pred_layer(dist_feat)
                e_w = torch.sigmoid(logits)
            else:
                e_w = None

            # 依次通过每一层进行特征和坐标更新
            for l_idx, layer in enumerate(self.base_block):
                h, x = layer(h, x, edge_type, edge_index, mask_ligand, e_w=e_w, fix_x=fix_x)
            all_x.append(x)
            all_h.append(h)

        # 准备返回结果
        outputs = {'x': x, 'h': h}
        if return_all:
            outputs.update({'all_x': all_x, 'all_h': all_h})
        return outputs

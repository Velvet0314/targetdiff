import argparse  # 导入命令行参数解析库
import os
import shutil
import time

import numpy as np
import torch
from torch_geometric.data import Batch
from torch_geometric.transforms import Compose
from torch_scatter import scatter_sum, scatter_mean  # 导入scatter函数，用于聚合操作
from tqdm.auto import tqdm  # 导入tqdm库，用于显示进度条

import utils.misc as misc
import utils.transforms as trans
from datasets import get_dataset
from datasets.pl_data import FOLLOW_BATCH
from models.molopt_score_model import ScorePosNet3D, log_sample_categorical
from utils.evaluation import atom_num

def unbatch_v_traj(ligand_v_traj, n_data, ligand_cum_atoms):
    # 将批处理的配体 v 拆分为单个样本
    all_step_v = [[] for _ in range(n_data)]  # 初始化每个样本的 v 列表
    for v in ligand_v_traj:  # 遍历轨迹中的每个时间步
        v_array = v.cpu().numpy()  # 将张量转换为numpy数组
        for k in range(n_data):
            # 根据累积原子数量拆分每个样本的v值
            all_step_v[k].append(v_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]])
    # 将每个样本的 v 堆叠为数组
    all_step_v = [np.stack(step_v) for step_v in all_step_v]  # num_samples * [num_steps, num_atoms_i]
    return all_step_v  # 返回拆分后的 v 列表

# 采样算法关键部分
def sample_diffusion_ligand(model, data, num_samples, batch_size=16, device='cuda:0',
                            num_steps=None, pos_only=False, center_pos_mode='protein',
                            sample_num_atoms='prior'):
    # 使用扩散模型采样配体
    all_pred_pos, all_pred_v = [], []  # 存储所有预测的配体位置和v值
    all_pred_pos_traj, all_pred_v_traj = [], []  # 存储所有位置和v的轨迹
    all_pred_v0_traj, all_pred_vt_traj = [], []  # 存储初始和最终的 v 
    time_list = []  # 存储每个批次的耗时
    num_batch = int(np.ceil(num_samples / batch_size))  # 计算需要的批次数
    current_i = 0  # 当前处理的样本索引
    for i in tqdm(range(num_batch)):  # 遍历每个批次，显示进度条
        n_data = batch_size if i < num_batch - 1 else num_samples - batch_size * (num_batch - 1)
        # 为当前批次复制数据，形成批处理
        batch = Batch.from_data_list([data.clone() for _ in range(n_data)], follow_batch=FOLLOW_BATCH).to(device)

        t1 = time.time()  # 记录批次开始时间
        with torch.no_grad():  # 禁用梯度计算，加速推理
            batch_protein = batch.protein_element_batch  # 获取蛋白质的批次索引
            # 步骤一：确定原子数量
            # 这里有三种方式，其中第一种对应算法中的步骤
            if sample_num_atoms == 'prior':
                # 根据先验分布采样配体原子数量
                pocket_size = atom_num.get_space_size(data.protein_pos.detach().cpu().numpy())  # 计算口袋大小
                ligand_num_atoms = [atom_num.sample_atom_num(pocket_size).astype(int) for _ in range(n_data)]  # 采样原子数量
                batch_ligand = torch.repeat_interleave(torch.arange(n_data), torch.tensor(ligand_num_atoms)).to(device)  # 生成配体批次索引
            elif sample_num_atoms == 'range':
                # 按顺序指定配体原子数量
                ligand_num_atoms = list(range(current_i + 1, current_i + n_data + 1))  # 生成原子数量列表
                batch_ligand = torch.repeat_interleave(torch.arange(n_data), torch.tensor(ligand_num_atoms)).to(device)  # 生成配体批次索引
            elif sample_num_atoms == 'ref':
                # 使用参考数据的原子数量
                batch_ligand = batch.ligand_element_batch  # 获取配体的批次索引
                ligand_num_atoms = scatter_sum(torch.ones_like(batch_ligand), batch_ligand, dim=0).tolist()  # 计算每个样本的原子数量
            else:
                raise ValueError  # 抛出异常
            
            # 步骤二：初始化配体位置？
            center_pos = scatter_mean(batch.protein_pos, batch_protein, dim=0)  # 计算每个蛋白质的中心位置
            batch_center_pos = center_pos[batch_ligand]  # 获取每个配体原子的中心位置
            # 步骤三：采样初始化——原子位置
            init_ligand_pos = batch_center_pos + torch.randn_like(batch_center_pos)  # 添加随机噪声，初始化配体位置

            # 步骤三：采样初始化—原子类型
            if pos_only:
                # 如果仅采样位置，使用初始的配体特征
                init_ligand_v = batch.ligand_atom_feature_full
            else:
                # 否则，从均匀分布中采样初始v值
                # 算法中对应的步骤
                uniform_logits = torch.zeros(len(batch_ligand), model.num_classes).to(device)  # 创建均匀分布的logits
                init_ligand_v = log_sample_categorical(uniform_logits)  # 采样v值

            # 调用模型的采样函数，执行扩散采样过程
            r = model.sample_diffusion(
                protein_pos=batch.protein_pos,
                protein_v=batch.protein_atom_feature.float(),
                batch_protein=batch_protein,

                init_ligand_pos=init_ligand_pos,
                init_ligand_v=init_ligand_v,
                batch_ligand=batch_ligand,
                num_steps=num_steps,
                pos_only=pos_only,
                center_pos_mode=center_pos_mode
            )

            # 获取采样结果
            ligand_pos, ligand_v, ligand_pos_traj, ligand_v_traj = r['pos'], r['v'], r['pos_traj'], r['v_traj']
            ligand_v0_traj, ligand_vt_traj = r['v0_traj'], r['vt_traj']

            # 将批处理的配体位置拆分为单个样本
            ligand_cum_atoms = np.cumsum([0] + ligand_num_atoms)  # 计算累积原子数量
            ligand_pos_array = ligand_pos.cpu().numpy().astype(np.float64)  # 转换为numpy数组
            all_pred_pos += [ligand_pos_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]] for k in range(n_data)]  # 存储每个样本的配体位置

            # 拆分位置轨迹
            all_step_pos = [[] for _ in range(n_data)]  # 初始化位置轨迹列表
            for p in ligand_pos_traj:  # 遍历每个时间步的配体位置
                p_array = p.cpu().numpy().astype(np.float64)  # 转换为numpy数组
                for k in range(n_data):
                    all_step_pos[k].append(p_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]])  # 添加到对应样本的轨迹中
            all_step_pos = [np.stack(step_pos) for step_pos in all_step_pos]  # 堆叠轨迹
            all_pred_pos_traj += [p for p in all_step_pos]  # 存储位置轨迹

            # 将批处理的配体v值拆分为单个样本
            ligand_v_array = ligand_v.cpu().numpy()  # 转换为numpy数组
            all_pred_v += [ligand_v_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]] for k in range(n_data)]  # 存储v值

            # 拆分 v 
            all_step_v = unbatch_v_traj(ligand_v_traj, n_data, ligand_cum_atoms)  # 调用拆分函数
            all_pred_v_traj += [v for v in all_step_v]  # 存储 v 

            if not pos_only:
                # 如果采样v值，拆分v0和vt的轨迹
                all_step_v0 = unbatch_v_traj(ligand_v0_traj, n_data, ligand_cum_atoms)
                all_pred_v0_traj += [v for v in all_step_v0]
                all_step_vt = unbatch_v_traj(ligand_vt_traj, n_data, ligand_cum_atoms)
                all_pred_vt_traj += [v for v in all_step_vt]

        t2 = time.time()  # 记录批次结束时间
        time_list.append(t2 - t1)  # 计算批次耗时
        current_i += n_data  # 更新当前处理的样本索引
    # 返回采样结果
    return all_pred_pos, all_pred_v, all_pred_pos_traj, all_pred_v_traj, all_pred_v0_traj, all_pred_vt_traj, time_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()  # 创建参数解析器
    parser.add_argument('config', type=str)  # 添加位置参数，配置文件路径
    parser.add_argument('-i', '--data_id', type=int)  # 添加可选参数，数据ID
    parser.add_argument('--device', type=str, default='cuda:0')  # 添加可选参数，指定设备
    parser.add_argument('--batch_size', type=int, default=100)  # 添加可选参数，批处理大小
    parser.add_argument('--result_path', type=str, default='./outputs')  # 添加可选参数，结果保存路径
    args = parser.parse_args()  # 解析命令行参数

    logger = misc.get_logger('sampling')  # 获取用于记录日志的logger

    # 加载配置文件
    config = misc.load_config(args.config)
    logger.info(config)  # 输出配置信息
    misc.seed_all(config.sample.seed)  # 设置随机种子，确保结果可复现

    # 加载模型检查点
    ckpt = torch.load(config.model.checkpoint, map_location=args.device)  # 加载模型参数
    logger.info(f"Training Config: {ckpt['config']}")  # 输出训练配置

    # 定义数据变换
    protein_featurizer = trans.FeaturizeProteinAtom()  # 蛋白质原子特征化
    ligand_atom_mode = ckpt['config'].data.transform.ligand_atom_mode  # 获取配体原子模式
    ligand_featurizer = trans.FeaturizeLigandAtom(ligand_atom_mode)  # 配体原子特征化
    transform = Compose([
        protein_featurizer,  # 蛋白质特征化
        ligand_featurizer,  # 配体特征化
        trans.FeaturizeLigandBond(),  # 配体键特征化
    ])

    # 加载数据集
    dataset, subsets = get_dataset(
        config=ckpt['config'].data,
        transform=transform
    )
    train_set, test_set = subsets['train'], subsets['test']  # 获取训练集和测试集
    logger.info(f'Successfully load the dataset (size: {len(test_set)})!')  # 输出数据集大小

    # 初始化模型
    model = ScorePosNet3D(
        ckpt['config'].model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,  # 蛋白质特征维度
        ligand_atom_feature_dim=ligand_featurizer.feature_dim  # 配体特征维度
    ).to(args.device)
    model.load_state_dict(ckpt['model'])  # 加载模型参数
    logger.info(f'Successfully load the model! {config.model.checkpoint}')  # 输出模型加载成功

    data = test_set[args.data_id]  # 获取指定ID的数据样本
    # 调用采样函数，生成配体
    pred_pos, pred_v, pred_pos_traj, pred_v_traj, pred_v0_traj, pred_vt_traj, time_list = sample_diffusion_ligand(
        model, data, config.sample.num_samples,  # 样本数量
        batch_size=args.batch_size, device=args.device,
        num_steps=config.sample.num_steps,  # 扩散步骤数
        pos_only=config.sample.pos_only,  # 是否仅采样位置
        center_pos_mode=config.sample.center_pos_mode,  # 中心位置模式
        sample_num_atoms=config.sample.sample_num_atoms  # 配体原子数量采样模式
    )
    # 将结果打包成字典
    result = {
        'data': data,
        'pred_ligand_pos': pred_pos,  # 预测的配体位置
        'pred_ligand_v': pred_v,  # 预测的配体v值
        'pred_ligand_pos_traj': pred_pos_traj,  # 位置轨迹
        'pred_ligand_v_traj': pred_v_traj,  # v值轨迹
        'time': time_list  # 耗时列表
    }
    logger.info('Sample done!')  # 输出采样完成

    # 保存结果
    result_path = args.result_path  # 获取结果保存路径
    os.makedirs(result_path, exist_ok=True)  # 创建目录
    shutil.copyfile(args.config, os.path.join(result_path, 'sample.yml'))  # 复制配置文件到结果目录
    torch.save(result, os.path.join(result_path, f'result_{args.data_id}.pt'))  # 保存结果到文件

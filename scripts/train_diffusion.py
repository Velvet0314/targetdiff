import argparse
import os
import shutil

import numpy as np
import torch
import torch.utils.tensorboard
from sklearn.metrics import roc_auc_score
from torch.nn.utils import clip_grad_norm_
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from tqdm.auto import tqdm

import utils.misc as misc
import utils.train as utils_train
import utils.transforms as trans
from datasets import get_dataset
from datasets.pl_data import FOLLOW_BATCH
from models.molopt_score_model import ScorePosNet3D

# 计算多类别的平均 AUROC
def get_auroc(y_true, y_pred, feat_mode):
    y_true = np.array(y_true)
    # y_pred 是模型的预测结果，它是一个二维数组，其中每一行表示一个样本的所有类别的预测概率
    y_pred = np.array(y_pred)
    avg_auroc = 0.  # 初始化平均 AUROC
    possible_classes = set(y_true)  # 获取数据集中所有可能的类别
    for c in possible_classes:  # 遍历每一个类别
        """
        roc_auc_score 需要两个参数：
            1.真实标签（这里是 y_true == c，表示当前类别的真实标签）
            2.预测概率（这里是 y_pred[:, c]，表示当前类别的预测概率）
        y_pred[:, c] 使用切片操作提取 y_pred 中第 c 列（即所有样本对类别 c 的预测概率）
        """
        auroc = roc_auc_score(y_true == c, y_pred[:, c])
        avg_auroc += auroc * np.sum(y_true == c)    # 按类别样本数加权累加 AUROC
        # 定义类别特征的映射关系
        mapping = {
            'basic': trans.MAP_INDEX_TO_ATOM_TYPE_ONLY,
            'add_aromatic': trans.MAP_INDEX_TO_ATOM_TYPE_AROMATIC,
            'full': trans.MAP_INDEX_TO_ATOM_TYPE_FULL
        }
        print(f'atom: {mapping[feat_mode][c]} \t auc roc: {auroc:.4f}') # 输出当前类别的 AUROC
    return avg_auroc / len(y_true)  # 返回样本加权后的平均 AUROC


if __name__ == '__main__':

    # 定义命令行参数解析器
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)  # 配置文件路径
    parser.add_argument('--device', type=str, default='cuda')  # 运行设备，默认为 CUDA
    parser.add_argument('--logdir', type=str, default='./logs_diffusion')  # 日志目录
    parser.add_argument('--tag', type=str, default='')  # 可选标签
    parser.add_argument('--train_report_iter', type=int, default=200)  # 日志打印频率
    args = parser.parse_args()  # 解析命令行参数

    # Load configs
    config = misc.load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    misc.seed_all(config.train.seed)

    # Logging
    log_dir = misc.get_new_log_dir(args.logdir, prefix=config_name, tag=args.tag)
    ckpt_dir = os.path.join(log_dir, 'checkpoints')  # 检查点保存目录
    os.makedirs(ckpt_dir, exist_ok=True)
    vis_dir = os.path.join(log_dir, 'vis')  # 可视化输出目录
    os.makedirs(vis_dir, exist_ok=True)
    logger = misc.get_logger('train', log_dir)  # 定义日志记录器
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)  # 定义 TensorBoard 日志
    logger.info(args)  # 打印命令行参数信息
    logger.info(config)  # 打印配置文件信息
    shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))  # 备份配置文件
    shutil.copytree('./models', os.path.join(log_dir, 'models'))  # 备份模型代码

    # Transforms
    # 特征化器和数据变换定义
    
    # 蛋白质特征化器
    # 将每个原子转换为一个特征向量并将其赋值给 data.protein_atom_feature
    protein_featurizer = trans.FeaturizeProteinAtom()  
    ligand_featurizer = trans.FeaturizeLigandAtom(config.data.transform.ligand_atom_mode)  # 配体特征化器
    transform_list = [
        protein_featurizer,  # 加载蛋白质特征化
        ligand_featurizer,  # 加载配体特征化
        trans.FeaturizeLigandBond(),  # 加载配体键特征化
    ]
    if config.data.transform.random_rot:  # 若配置中启用随机旋转
        transform_list.append(trans.RandomRotation())
    transform = Compose(transform_list)  # 将所有变换组合成一个整体

    # Datasets and loaders
    logger.info('Loading dataset...')
    dataset, subsets = get_dataset(
        config=config.data,
        transform=transform
    )
    train_set, val_set = subsets['train'], subsets['test']
    logger.info(f'Training: {len(train_set)} Validation: {len(val_set)}')

    # follow_batch = ['protein_element', 'ligand_element']
    collate_exclude_keys = ['ligand_nbh_list']  # 数据加载时排除某些键
    train_iterator = utils_train.inf_iterator(DataLoader(  # 定义无限循环的数据迭代器
        train_set,
        batch_size=config.train.batch_size,  # 每批次大小
        shuffle=True,  # 随机打乱
        num_workers=config.train.num_workers,  # 使用多线程加载数据
        follow_batch=FOLLOW_BATCH,  # 特定的批处理字段
        exclude_keys=collate_exclude_keys  # 排除某些特定键
    ))
    # 定义验证集加载器
    val_loader = DataLoader(val_set, config.train.batch_size, shuffle=False,
                            follow_batch=FOLLOW_BATCH, exclude_keys=collate_exclude_keys)

    # Model
    logger.info('Building model...')
    model = ScorePosNet3D(
        config.model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,    # 蛋白质特征维度
        ligand_atom_feature_dim=ligand_featurizer.feature_dim   # 配体特征维度
    ).to(args.device)
    # print(model)
    print(f'protein feature dim: {protein_featurizer.feature_dim} ligand feature dim: {ligand_featurizer.feature_dim}')
    logger.info(f'# trainable parameters: {misc.count_parameters(model) / 1e6:.4f} M')

    # Optimizer and scheduler
    optimizer = utils_train.get_optimizer(config.train.optimizer, model)
    scheduler = utils_train.get_scheduler(config.train.scheduler, optimizer)


    def train(it):
        model.train()   # 设置模型为训练模式
        optimizer.zero_grad()   # 清空梯度
        """
        梯度累积
        通常情况下，深度学习模型每处理一个批次数据，就会执行一次反向传播（计算梯度），并通过 optimizer.step() 更新模型参数
        然而，在内存受限的情况下，如果直接使用较大的批次，可能会导致显存不足
        因此，梯度累积通过分批处理和延迟更新权重的方式，模拟大批次训练的效果，而不会占用过多内存
        train.n_acc_batch 控制了模型在进行一次参数更新之前，积累多少个小批次的梯度

        通过梯度累积，训练过程实际上是在进行多个小批次的训练后，一次性更新模型权重。假设每个小批次的大小为 batch_size，那么：
        每次训练迭代会处理 n_acc_batch 个小批次的梯度
        每个小批次的损失会被归一化，并通过 loss.backward() 累加梯度
        直到处理完 n_acc_batch 个批次后，才调用 optimizer.step() 更新模型参数
        这相当于用 n_acc_batch * batch_size 大小的批次进行训练，但内存只占用了 batch_size 大小的批次
        因此，train.n_acc_batch 控制了 模型参数更新的频率，在训练过程中，只有在累积了多个小批次的梯度后，才会进行一次参数更新
        """
        for _ in range(config.train.n_acc_batch):
            batch = next(train_iterator).to(args.device)

            # 添加噪声
            protein_noise = torch.randn_like(batch.protein_pos) * config.train.pos_noise_std
            # 得到加噪的蛋白质位置
            gt_protein_pos = batch.protein_pos + protein_noise
            # 调用模型计算损失
            results = model.get_diffusion_loss(
                protein_pos=gt_protein_pos,
                protein_v=batch.protein_atom_feature.float(),
                batch_protein=batch.protein_element_batch,

                ligand_pos=batch.ligand_pos,
                ligand_v=batch.ligand_atom_feature_full,
                batch_ligand=batch.ligand_element_batch
            )
            loss, loss_pos, loss_v = results['loss'], results['loss_pos'], results['loss_v']
            loss = loss / config.train.n_acc_batch  # 损失归一化
            loss.backward() # 反向传播计算梯度
        orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)    # 梯度裁剪
        optimizer.step()    # 更新模型参数

        # 打印和记录训练过程
        if it % args.train_report_iter == 0:
            logger.info(
                '[Train] Iter %d | Loss %.6f (pos %.6f | v %.6f) | Lr: %.6f | Grad Norm: %.6f' % (
                    it, loss, loss_pos, loss_v, optimizer.param_groups[0]['lr'], orig_grad_norm
                )
            )
            for k, v in results.items():
                if torch.is_tensor(v) and v.squeeze().ndim == 0:
                    writer.add_scalar(f'train/{k}', v, it)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
            writer.add_scalar('train/grad', orig_grad_norm, it)
            writer.flush()


    def validate(it):
        # fix time steps
        sum_loss, sum_loss_pos, sum_loss_v, sum_n = 0, 0, 0, 0
        sum_loss_bond, sum_loss_non_bond = 0, 0
        all_pred_v, all_true_v = [], []
        all_pred_bond_type, all_gt_bond_type = [], []
        with torch.no_grad():
            model.eval()    # 设置模型为评估模式
            # 通过 tqdm 包装 val_loader，可以在验证过程中显示进度条，便于监控验证过程的进度
            for batch in tqdm(val_loader, desc='Validate'):
                batch = batch.to(args.device)
                batch_size = batch.num_graphs
                t_loss, t_loss_pos, t_loss_v = [], [], []
                # 生成一个从 0 到 num_timesteps - 1 的等间距的时间步序列，这里取了 10 个时间步
                for t in np.linspace(0, model.num_timesteps - 1, 10).astype(int):
                    # 为当前批次的每个样本创建一个时间步张量。t 是时间步的当前值，batch_size 表示当前批次中样本的数量
                    # 这个时间步会被传递给模型，用来计算不同时间步的损失
                    time_step = torch.tensor([t] * batch_size).to(args.device)
                    results = model.get_diffusion_loss(
                        protein_pos=batch.protein_pos, # 蛋白质的位置
                        protein_v=batch.protein_atom_feature.float(),   # 蛋白质的特征
                        batch_protein=batch.protein_element_batch,  # 蛋白质的批次元素

                        ligand_pos=batch.ligand_pos,    # 配体的位置
                        ligand_v=batch.ligand_atom_feature_full,    # 配体的特征
                        batch_ligand=batch.ligand_element_batch,    # 配体的批次元素
                        time_step=time_step # 当前的时间步，控制模型在不同时间步的表现
                    )
                    loss, loss_pos, loss_v = results['loss'], results['loss_pos'], results['loss_v']

                    # 累积损失
                    sum_loss += float(loss) * batch_size
                    sum_loss_pos += float(loss_pos) * batch_size
                    sum_loss_v += float(loss_v) * batch_size
                    sum_n += batch_size
                    all_pred_v.append(results['ligand_v_recon'].detach().cpu().numpy())
                    all_true_v.append(batch.ligand_atom_feature_full.detach().cpu().numpy())

        avg_loss = sum_loss / sum_n
        avg_loss_pos = sum_loss_pos / sum_n
        avg_loss_v = sum_loss_v / sum_n
        atom_auroc = get_auroc(np.concatenate(all_true_v), np.concatenate(all_pred_v, axis=0),
                               feat_mode=config.data.transform.ligand_atom_mode)

        # 学习率调整配置
        if config.train.scheduler.type == 'plateau':
            scheduler.step(avg_loss)
        elif config.train.scheduler.type == 'warmup_plateau':
            scheduler.step_ReduceLROnPlateau(avg_loss)
        else:
            scheduler.step()

        logger.info(
            '[Validate] Iter %05d | Loss %.6f | Loss pos %.6f | Loss v %.6f e-3 | Avg atom auroc %.6f' % (
                it, avg_loss, avg_loss_pos, avg_loss_v * 1000, atom_auroc
            )
        )
        writer.add_scalar('val/loss', avg_loss, it)
        writer.add_scalar('val/loss_pos', avg_loss_pos, it)
        writer.add_scalar('val/loss_v', avg_loss_v, it)
        writer.flush()
        return avg_loss


    try:
        # 初始化最佳损失和迭代次数
        best_loss, best_iter = None, None
        for it in range(1, config.train.max_iters + 1):
            # with torch.autograd.detect_anomaly():
            train(it)
            """
            验证模型：每经过一定次数的迭代，或者在最后一次迭代时，都会执行一次验证
            config.train.val_freq 是一个配置参数，表示每 val_freq 次迭代执行一次验证
            it == config.train.max_iters 确保在最后一次迭代时也进行验证
            """
            if it % config.train.val_freq == 0 or it == config.train.max_iters:
                val_loss = validate(it)
                # 在每次验证后，如果当前验证损失 val_loss 小于之前记录的 best_loss（即当前模型表现更好）
                if best_loss is None or val_loss < best_loss:
                    logger.info(f'[Validate] Best val loss achieved: {val_loss:.6f}')
                    best_loss, best_iter = val_loss, it
                    ckpt_path = os.path.join(ckpt_dir, '%d.pt' % it)
                    torch.save({
                        'config': config,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'iteration': it,
                    }, ckpt_path)
                else:
                    logger.info(f'[Validate] Val loss is not improved. '
                                f'Best val loss: {best_loss:.6f} at iter {best_iter}')
    except KeyboardInterrupt:
        logger.info('Terminating...')

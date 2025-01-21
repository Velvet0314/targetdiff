import os
import numpy as np
import torch
from glob import glob
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from collections import defaultdict
import sys
sys.path.append("/home/aita8180/data/mntdata/mjt/odemolcraft/")

def tanimoto_sim(mol, ref):
    fp1 = Chem.RDKFingerprint(ref)
    fp2 = Chem.RDKFingerprint(mol)
    return DataStructs.TanimotoSimilarity(fp1,fp2)

def calc_pairwise_sim(mols):
    n = len(mols)
    sims = []
    for i in range(n):
        for j in range(i + 1, n):
            sims.append(tanimoto_sim(mols[i], mols[j]))
    return np.array(sims)

def computer_diversity(mols):
    div_all = []
    # for result in tqdm(results):
    div_all.append(np.mean(1 - calc_pairwise_sim(mols)))

    div_all = np.array(div_all)
    div_all = div_all[~np.isnan(div_all)]
    return div_all

if __name__ == '__main__':
    # eval_path = '/home/aita8180/data/mntdata/mjt/testmolcraft/logs/aita8180_bfn_sbdd/official/default/test_outputs_v124/20241028-152757'
    # results_fn_list = glob(os.path.join(eval_path, 'vina_docked.pt'))
    eval_path = '/home/aita8180/data/mntdata/mjt/odemolcraft/test/samples/'
    results_fn_list = glob(os.path.join(eval_path, 'mix_vina_docked.pt'))
    print("num of results.pt: ", len(results_fn_list))

    # 字典存储每个目标蛋白质及其生成的分子的 SMILES
    protein_ligand_dict = defaultdict(list)

    # 遍历加载每个results文件
    for results_fn in results_fn_list:
        # 加载 .pt 文件
        results = torch.load(results_fn)

        # 假设每个结果文件中有 'ligand_filename' 和 'smiles'
        for result in results:
            if 'ligand_filename' in result and 'mol' in result:
                ligand_filename = result['ligand_filename']
                smiles = result['mol']
                # 根据 ligand_filename 对生成的分子进行分组
                protein_ligand_dict[ligand_filename].append(smiles)

    # 存储所有目标蛋白质的diversity值
    protein_diversities = []

    # 计算每个目标蛋白质的多样性
    for ligand_filename, smiles_list in protein_ligand_dict.items():
        diversity = computer_diversity(smiles_list)
        protein_diversities.append(diversity)
        print(f"{ligand_filename} 的diversity: {diversity}")
    print(len(protein_ligand_dict))
    # 计算所有目标蛋白质的diversity的平均值和中位数
    mean_diversity = np.mean(protein_diversities)
    median_diversity = np.median(protein_diversities)

    print(f"所有目标蛋白质的多样性平均值: {mean_diversity}")
    print(f"所有目标蛋白质的多样性中位数: {median_diversity}")
# def calculate_tanimoto_similarity(smiles_list):
#     # 将 SMILES 转换为分子指纹
#     fingerprints = []
#     for smi in smiles_list:
#         mol = Chem.MolFromSmiles(smi)
#         if mol is not None:
#             # 使用 Morgan fingerprint （类似于 ECFC4）
#             fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
#             fingerprints.append(fp)
#         else:
#             print(f"无效的 SMILES: {smi}")
    
#     # 计算分子间的 Tanimoto 相似性
#     num_fps = len(fingerprints)
#     if num_fps < 2:
#         # 如果生成分子小于2个，无法计算多样性，返回0
#         return 0
    
#     similarity_matrix = np.zeros((num_fps, num_fps))
#     for i in range(num_fps):
#         for j in range(i+1, num_fps):

#             sim = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
#             similarity_matrix[i, j] = sim
#             similarity_matrix[j, i] = sim
    
#     # 计算多样性，基于 (1 - 相似性)
#     diversity_scores = 1 - similarity_matrix[np.triu_indices(num_fps, 1)]
#     return np.mean(diversity_scores)
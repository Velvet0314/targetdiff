import lmdb
import pickle

# 必要的参数，否则会报错
env = lmdb.open('./data/crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb',
            map_size=10*(1024*1024*1024),
            subdir=False,
            readonly=True,
            lock=False,
        )

with env.begin() as txn:
    cursor = txn.cursor()
    count = 0
    for key, value in cursor:
        data = pickle.loads(value)
        print("\n值解析后的字典键:")
        for k in data.keys():
            print(f"- {k}: {type(data[k])}")
        
        print("\n复合物ID:", key.decode())
        print("-" * 50)
        
        # 打印所有字符串类型数据
        print("\n字符串类型数据:")
        print(f"蛋白质分子名称: {data['protein_molecule_name']}")
        print(f"配体SMILES: {data['ligand_smiles']}")
        print(f"蛋白质文件: {data['protein_filename']}")
        print(f"配体文件: {data['ligand_filename']}")
        
        # 打印所有张量类型数据
        print("\n张量类型数据(形状):")
        print(f"蛋白质元素: {data['protein_element'].shape}")
        print(f"蛋白质位置: {data['protein_pos'].shape}")
        print(f"蛋白质主链标记: {data['protein_is_backbone'].shape}")
        print(f"蛋白质原子到氨基酸类型映射: {data['protein_atom_to_aa_type'].shape}")
        print(f"配体元素: {data['ligand_element'].shape}")
        print(f"配体位置: {data['ligand_pos'].shape}")
        print(f"配体键连接索引: {data['ligand_bond_index'].shape}")
        print(f"配体键类型: {data['ligand_bond_type'].shape}")
        print(f"配体质心: {data['ligand_center_of_mass'].shape}")
        print(f"配体原子特征: {data['ligand_atom_feature'].shape}")
        
        # 打印列表类型数据
        print("\n列表类型数据(前5项):")
        print(f"蛋白质原子名称: {data['protein_atom_name'][:5]}")
        print(f"配体杂化类型: {data['ligand_hybridization'][:5]}")
        
        # 打印字典类型数据
        nbh_items = list(data['ligand_nbh_list'].items())[:3]
        for k, v in nbh_items:
            print(f"原子{k}的邻接原子: {v}")
            
        print("-" * 50)

        count += 1
        if count >= 1:  # 只显示第一个键值对
            break
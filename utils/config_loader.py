import json
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.cnn1d import CNN1D, CNN1D_KAT
from models.res_sa import Res_SA
from torch.utils.data import DataLoader, random_split
from models.dataset import h5Dataset, KATDataset,CWRUDataset


def get_dataset(file_path):
    if file_path.endswith('h5data'):
        return h5Dataset(file_path)
    elif file_path.endswith('KAT'):
        return KATDataset(file_path)
    elif file_path.endswith('12k Drive End Bearing Fault Data'):
        return CWRUDataset(file_path)
    else:
        raise ValueError(f"不支持的文件格式: {file_path}")
    
def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def get_model(config):
    """根据配置创建模型"""
    model_name = config['model']['name']
    model_params = config['model']['params']
    
    if model_name == "CNN1D":
        return CNN1D()
    elif model_name == "CNN1D_KAT":
        return CNN1D_KAT()
    elif model_name == "Res_SA":
        return Res_SA(**model_params)
    else:
        raise ValueError(f"不支持的模型类型: {model_name}")

def setup_training(config):
    """设置训练环境和参数"""
    # 设置设备
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model = get_model(config)
    model = model.to(device)
    
    # 设置优化器
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config['training']['learning_rate']
    )
    
    # 设置学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config['training']['scheduler']['factor'],
        patience=config['training']['scheduler']['patience'],
        verbose=True
    )
    
    return model, optimizer, scheduler, device



def setup_dataset(config):
    """设置数据集和数据加载器"""
    # 从配置中获取数据相关参数
    file_path = config['data']['file_path']
    batch_size = config['data']['batch_size']
    train_ratio = config['data']['train_ratio']
    
    # 创建数据集
    dataset = get_dataset(file_path)
    
    # 添加数据集大小检查
    total_size = len(dataset)
    if total_size == 0:
        raise ValueError("数据集为空！请检查数据集路径和加载过程。")
    
    # 计算训练集和验证集的大小
    train_size = int(0.8 * total_size)  # 假设使用80%作为训练集
    val_size = total_size - train_size
    
    # 确保两个子集都不为空
    if train_size == 0 or val_size == 0:
        raise ValueError(f"数据集划分错误：训练集大小={train_size}，验证集大小={val_size}")
    
    # 分割数据集
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size]
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    return train_loader, val_loader

get_dataset('data/h5data')
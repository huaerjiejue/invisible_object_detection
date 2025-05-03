import torch
from utils.config import Config
from models.model import InvisibleObjectDetector


def main():
    # 加载配置
    config = Config('configs/model_config.yaml', 'configs/train_config.yaml')
    
    # 创建模型
    model = InvisibleObjectDetector(config)
    
    # 获取训练配置
    training_config = config.get_training_config()
    
    # 设置设备
    device = torch.device(training_config['device'])
    model = model.to(device)
    
    # 创建优化器
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config['learning_rate'],
        weight_decay=training_config['weight_decay']
    )
    
    # 这里可以添加数据加载和训练循环
    # ...


if __name__ == '__main__':
    main()

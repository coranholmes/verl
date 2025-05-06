#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
知识图谱抽取训练的启动脚本
用于包装Hydra配置，使其更易于从命令行调用
"""

import os
import sys
import argparse


def main():
    """解析命令行参数并启动训练"""
    parser = argparse.ArgumentParser(description="知识图谱抽取的PPO训练")
    
    parser.add_argument("--config_path", type=str, default="/mnt/afs/tanka/shihao/project/data/kg_extraction/kg_extraction_config.yaml",
                        help="自定义配置文件路径，覆盖默认配置")
    
    parser.add_argument("--data_path", type=str, default="/mnt/afs/tanka/shihao/data/kg_extraction/kg_extraction_train.jsonl",
                        help="训练数据路径，覆盖配置文件中的设置")
    
    parser.add_argument("--model_path", type=str, default="/mnt/afs/tanka/shihao/model/Qwen2.5-0.5B-Instruct",
                        help="基础模型路径，覆盖配置文件中的设置")
    
    parser.add_argument("--output_dir", type=str, default="/mnt/afs/tanka/shihao/outputs/kg_extraction_debug",
                        help="输出目录，覆盖配置文件中的设置")
    
    parser.add_argument("--debug", action="store_true",
                        help="调试模式，使用较小的训练配置")
    
    parser.add_argument("--quick_test", action="store_true",
                        help="快速测试模式，仅初始化训练环境但不开始训练")
    
    args = parser.parse_args()
    
    # 打印默认参数
    print("\n" + "="*50)
    print("默认参数配置：")
    print(f"配置文件路径: {args.config_path}")
    print(f"训练数据路径: {args.data_path}")
    print(f"模型路径: {args.model_path}")
    print(f"输出目录: {args.output_dir}")
    print(f"调试模式: {'开启' if args.debug else '关闭'}")
    print(f"快速测试: {'开启' if args.quick_test else '关闭'}")
    print("="*50 + "\n")
    
    # 构建Hydra命令行参数
    hydra_args = [
        f"--config-path={os.path.dirname(args.config_path)}",
        f"--config-name={os.path.basename(args.config_path).split('.')[0]}"
    ]
    
    # 添加覆盖参数
    overrides = []
    
    if args.data_path:
        overrides.append(f"data.train_path={args.data_path}")
    
    if args.model_path:
        overrides.append(f"actor_rollout_ref.model.path={args.model_path}")
        overrides.append(f"critic.model.path={args.model_path}")
    
    if args.output_dir:
        overrides.append(f"trainer.output_dir={args.output_dir}")
    
    # 调试模式设置
    if args.debug:
        overrides.extend([
            "trainer.total_episodes=5",            # 仅处理5个训练样本
            "trainer.eval_episodes=2",             # 仅评估2个样本
            "algorithm.batch_size=2",              # 小批量大小
            "algorithm.epochs=10",                 # 减少训练步数
            "algorithm.lr=1e-5"                    # 设置学习率
        ])
    
    # 快速测试模式
    if args.quick_test:
        overrides.append("quick_test_mode=true")
    
    # 添加所有覆盖参数
    hydra_args.extend(overrides)
    
    # 构建完整的命令行
    from verl.trainer.main_ppo_kg import main
    
    # 备份原始参数
    original_argv = sys.argv.copy()
    
    try:
        # 替换系统参数
        sys.argv = [sys.argv[0]] + hydra_args
        print("运行命令: python", sys.argv[0], " ".join(hydra_args))
        
        # 调用Hydra主函数
        main()
    finally:
        # 恢复原始参数
        sys.argv = original_argv


if __name__ == "__main__":
    main() 
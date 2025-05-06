# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
知识图谱抽取的PPO训练脚本，基于verl框架实现
"""

import os
import argparse
import sys

import hydra
import ray

from verl.trainer.ppo.ray_trainer import RayPPOTrainer
# from verl.trainer.ppo.reward import load_reward_manager
from verl.utils.kg_rewards import compute_kg_extraction_reward

def get_custom_reward_fn(config):
    """
    从指定路径动态加载自定义奖励函数。
    
    参数:
        config: 包含自定义奖励函数配置的字典
        
    返回:
        wrapped_fn: 带有额外参数的包装奖励函数
    """
    import importlib.util
    import sys

    # 获取自定义奖励函数配置
    reward_fn_config = config.get("custom_reward_function") or {}
    file_path = reward_fn_config.get("path")
    if not file_path:
        return None

    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"奖励函数文件 '{file_path}' 未找到.")

    # 动态导入模块
    spec = importlib.util.spec_from_file_location("custom_module", file_path)
    module = importlib.util.module_from_spec(spec)
    try:
        sys.modules["custom_module"] = module
        spec.loader.exec_module(module)
    except Exception as e:
        raise RuntimeError(f"从 '{file_path}' 加载模块时出错: {e}") from e

    # 获取指定的函数
    function_name = reward_fn_config.get("name")
    if not hasattr(module, function_name):
        raise AttributeError(f"奖励函数 '{function_name}' 在文件 '{file_path}' 中未找到.")

    print(f"使用自定义奖励函数 '{function_name}' (来自 '{file_path}')")
    raw_fn = getattr(module, function_name)

    # 获取奖励函数的额外参数
    reward_kwargs = dict(reward_fn_config.get("reward_kwargs", {}))
    
    # 创建包装函数，注入额外参数
    def wrapped_fn(*args, **kwargs):
        return raw_fn(*args, **kwargs, **reward_kwargs)

    return wrapped_fn


@hydra.main(config_path="config", config_name="kg_extraction_config", version_base=None)
def main(config):
    """主入口函数，使用hydra加载配置并启动知识图谱抽取的PPO训练"""
    try:
        # 解析命令行参数，但不使用sys.argv（因为hydra已经处理过这些参数）
        # 仅用于打印调试信息和获取默认值
        args = argparse.ArgumentParser(description="知识图谱抽取的PPO训练")
        args.add_argument("--config_path", type=str, default="/mnt/afs/tanka/shihao/project/data/kg_extraction/kg_extraction_config.yaml",
                         help="自定义配置文件路径，覆盖默认配置")
        args.add_argument("--data_path", type=str, default="/mnt/afs/tanka/shihao/data/kg_extraction/kg_extraction_train.jsonl",
                         help="训练数据路径，覆盖配置文件中的设置")
        args.add_argument("--model_path", type=str, default="/mnt/afs/tanka/shihao/model/Qwen2.5-0.5B-Instruct",
                         help="基础模型路径，覆盖配置文件中的设置")
        args.add_argument("--output_dir", type=str, default="/mnt/afs/tanka/shihao/outputs/kg_extraction_debug",
                         help="输出目录，覆盖配置文件中的设置")
        args.add_argument("--debug", action="store_true",
                         help="调试模式，使用较小的训练配置")
        args.add_argument("--quick_test", action="store_true",
                         help="快速测试模式，仅初始化训练环境但不开始训练")
        args = args.parse_args([])  # 仅获取默认值，不处理实际命令行参数
        
        # 打印调试信息
        print_debug_info(config, args)
    except Exception as e:
        print(f"调试信息处理失败: {e}")
    
    # 应用快速测试模式标志（如果需要）
    if "quick_test_mode" in config and config.quick_test_mode:
        print("启用快速测试模式...")
    
    run_ppo(config)


def run_ppo(config) -> None:
    """
    运行PPO训练的主函数
    
    参数:
        config: 包含训练配置的对象
    """
    # 步骤1: 设置环境变量和初始化Ray
    os.environ["ENSURE_CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not ray.is_initialized():
        # 初始化本地Ray集群
        ray.init(
            runtime_env={
                "env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN", "VLLM_LOGGING_LEVEL": "WARN"}
            },
            num_cpus=config.ray_init.num_cpus,
        )

    # 检查是否是快速测试模式
    quick_test_mode = config.get("quick_test_mode", False)
    if quick_test_mode:
        print("\n===== 快速测试模式 =====")
        print("仅初始化环境，不执行训练流程")
        print("Ray集群已初始化")
        print("配置检查完成")
        print("=========================\n")
        return

    # 步骤2: 创建远程任务运行器并执行
    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)  # 确保主任务不在头节点上调度
class TaskRunner:
    """Ray远程任务运行器，负责执行PPO训练的实际任务"""
    
    async def run(self, config):
        """
        运行PPO训练任务
        
        参数:
            config: 包含训练配置的对象
        """
        # 步骤1: 打印初始配置
        from pprint import pprint
        from omegaconf import OmegaConf
        from verl.utils.fs import copy_to_local

        pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True会计算符号值
        OmegaConf.resolve(config)

        # 步骤2: 从HDFS下载检查点到本地
        local_path = copy_to_local(config.actor_rollout_ref.model.path)

        # 步骤3: 实例化tokenizer和processor
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, use_fast=True)  # 用于多模态LLM，可能为None

        # 步骤4: 根据策略定义工作器类
        if config.actor_rollout_ref.actor.strategy == "fsdp":
            # 使用FSDP (Fully Sharded Data Parallel) 策略
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

            # 根据rollout模式选择actor类
            actor_rollout_cls = (
                AsyncActorRolloutRefWorker
                if config.actor_rollout_ref.rollout.mode == "async"
                else ActorRolloutRefWorker
            )
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            # 使用Megatron策略
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = ActorRolloutRefWorker
            ray_worker_group_cls = NVMegatronRayWorkerGroup

        else:
            raise NotImplementedError

        # 步骤5: 设置资源池和角色映射
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
            Role.Critic: ray.remote(CriticWorker),
        }

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        # 步骤6: 设置奖励模型（如果启用）
        if config.reward_model.enable:
            if config.reward_model.strategy == "fsdp":
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        # 步骤7: 设置参考模型（如果需要）
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        # 步骤8: 加载奖励函数
        reward_manager_name = config.reward_model.get("reward_manager", "naive")
        if reward_manager_name == 'naive':
            from verl.workers.reward_manager import NaiveRewardManager
            reward_manager_cls = NaiveRewardManager
        elif reward_manager_name == 'prime':
            from verl.workers.reward_manager import PrimeRewardManager
            reward_manager_cls = PrimeRewardManager
        else:
            raise NotImplementedError
        
        compute_score = compute_kg_extraction_reward
        reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=0,          
                                       compute_score=compute_score)

        val_reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=1, 
                                           compute_score=compute_score)
        
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        # 步骤9: 创建PPO训练器并开始训练
        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
        )
        trainer.init_workers()
        await trainer.fit()


def print_debug_info(config, args):
    """打印调试信息"""
    print("\n" + "="*50)
    print("调试信息：")
    print(f"配置文件默认路径: {args.config_path}")
    print(f"训练数据默认路径: {args.data_path}")
    print(f"模型默认路径: {args.model_path}")
    print(f"输出目录默认值: {args.output_dir}")
    
    print("\n实际配置值：")
    try:
        print(f"训练数据路径: {config.data.train_path}")
        print(f"模型路径: {config.actor_rollout_ref.model.path}")
        print(f"输出目录: {config.trainer.output_dir}")
    except Exception as e:
        print(f"获取实际配置值失败: {e}")
    
    print("\nHydra命令行参数:")
    for arg in sys.argv[1:]:
        print(f"  {arg}")
    print("="*50 + "\n")


if __name__ == "__main__":
    # 直接调用main()函数，由Hydra处理命令行参数
    main()

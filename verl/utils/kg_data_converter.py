import os
import json
import pandas as pd
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import re

def extract_json_from_answer(api_output: str) -> str:
    """
    Extracts the content inside <answer> ... </answer> tags.
    Returns None if not found.
    """
    match = re.search(r"<answer>(.*?)</answer>", api_output, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def extract_thinking_from_output(api_output: str) -> str:
    """
    Extracts the content inside <think> ... </think> or <reasoning> ... </reasoning> tags.
    Returns empty string if not found.
    """
    # 先尝试提取<think>标签
    match = re.search(r"<think>(.*?)</think>", api_output, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # 如果没有找到，尝试提取<reasoning>标签
    match = re.search(r"<reasoning>(.*?)</reasoning>", api_output, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    return ""

def convert_to_jsonl(
    input_csv_path: str, 
    output_jsonl_path: str,
    prompt_template: str =  \
"""Chunk Text:
{input_text}

Instructions: Analyze the above text, explain your reasoning inside <think> tags, and output a structured knowledge graph in JSON format inside <answer> tags. The JSON should have two keys: 'nodes' and 'edges'. 

## For example:
{{"nodes": [{{"id": 1, "label": "EntityName", "type": "EntityType"}}, ...], 
"edges": [{{"source": 1, "target": 2, "relation": "RELATION_TYPE", "description": "Explanation of the relationship", "keywords": ["key1", "key2","key3"], "strength": 8, "msg_ids": [0, 1, 2]}}, ...]}}

## Requirements:
- Ensure your output strictly follows this format.
- For each relationship/edge, include:
    - source: ID of the source entity
    - target: ID of the target entity
    - relation: Only one type of relationship for each edge
    - description: Detailed explanation of why the entities are related
    - keywords: List of key terms that can be searched or represent important elements of the relationship
    - strength: Numeric score (1-10) indicating relationship strength
    - msg_ids: List of message IDs where this relationship was mentioned
- The output language is English. For specific entities (person name, product name), use the the original language of in the text.
- msg_ids can not be empty.
""",

    data_source_tag: str = "kg_extraction"
) -> None:
    """
    将CSV格式的图抽取数据转换为verl框架所需的JSONL格式
    
    参数:
        input_csv_path: 输入CSV文件路径
        output_jsonl_path: 输出JSONL文件路径
        prompt_template: 提示模板，{input_text}将被替换为实际文本
        data_source_tag: 数据来源标签，用于奖励函数选择
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(os.path.abspath(output_jsonl_path)), exist_ok=True)
    
    # 读取CSV文件
    df = pd.read_csv(input_csv_path)
    
    # 创建JSONL数据
    jsonl_data = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Converting data"):
        input_text = row.get('input_text', '')
        
        # 处理ground truth数据 - 优先使用api_output
        api_output = row.get('api_output', '')
        ground_truth_answer = extract_json_from_answer(api_output) or '{}'
        ground_truth_reasoning = row.get('thinking_process', '') 
            
        # 尝试解析JSON - 确保是有效的JSON
        try:
            json.loads(ground_truth_answer)
        except json.JSONDecodeError:
            print(f"Warning: Invalid JSON in ground_truth_answer, skipping: {ground_truth_answer[:100]}...")
            continue
        
        # 创建verl格式的数据项
        prompt = prompt_template.format(input_text=input_text)
        
        jsonl_item = {
            "prompt": prompt,
            "response": "",  # verl框架会用模型生成的回答填充这里
            "reward_model": {
                "ground_truth_answer": ground_truth_answer,
                "ground_truth_reasoning": ground_truth_reasoning
            },
            "data_source": data_source_tag,
            "extra_info": {
                "original_input": input_text
            }
        }
        
        jsonl_data.append(jsonl_item)
    
    # 写入JSONL文件
    with open(output_jsonl_path, 'w', encoding='utf-8') as f:
        for item in jsonl_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Converted {len(jsonl_data)} items to {output_jsonl_path}")

def create_train_val_split(
    input_jsonl_path: str,
    train_output_path: str,
    val_output_path: str,
    val_ratio: float = 0.1,
    seed: int = 42
) -> None:
    """
    将JSONL数据分割为训练集和验证集
    
    参数:
        input_jsonl_path: 输入JSONL文件路径
        train_output_path: 训练集输出路径
        val_output_path: 验证集输出路径
        val_ratio: 验证集比例
        seed: 随机种子
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(os.path.abspath(train_output_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(val_output_path)), exist_ok=True)
    
    # 读取JSONL文件
    data = []
    with open(input_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    
    # 随机分割
    import random
    random.seed(seed)
    random.shuffle(data)
    
    split_idx = int(len(data) * (1 - val_ratio))
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    # 写入训练集
    with open(train_output_path, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # 写入验证集
    with open(val_output_path, 'w', encoding='utf-8') as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Split data into {len(train_data)} training samples and {len(val_data)} validation samples")

def generate_hydra_config(
    model_path: str,
    train_data_path: str,
    val_data_path: Optional[str] = None,
    output_dir: str = "./outputs",
    reward_function_path: str = "/mnt/afs/tanka/shihao/project/verl/verl/utils/kg_rewards.py",
    reward_function_name: str = "compute_kg_extraction_reward",
    n_gpus_per_node: int = 2,
    nnodes: int = 1,
    output_config_path: str = "/mnt/afs/tanka/shihao/project/verl/verl/trainer/config/kg_extraction.yaml"
) -> None:
    """
    生成verl框架所需的hydra配置文件
    
    参数:
        model_path: 基础模型路径
        train_data_path: 训练数据路径
        val_data_path: 验证数据路径，可选
        output_dir: 输出目录
        reward_function_path: 奖励函数文件路径
        reward_function_name: 奖励函数名称
        n_gpus_per_node: 每个节点的GPU数量
        nnodes: 节点数量
        output_config_path: 输出配置文件路径
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(os.path.abspath(output_config_path)), exist_ok=True)
    
    # 基本配置
    config = f"""
actor_rollout_ref:
  model:
    path: {model_path}
  
  actor:
    strategy: fsdp
    fsdp_config:
      reshard_after_forward: true
      move_params_to_cpu: false
      state_dict_type: full
    use_kl_loss: true
    kl_coef: 0.1

  rollout:
    mode: sync
    use_cache: true
    override_generation_kwargs:
      max_new_tokens: 1024
      temperature: 0.8
      top_p: 0.9
      do_sample: true

critic:
  model:
    path: {model_path}
  
  strategy: fsdp
  fsdp_config:
    state_dict_type: full
    reshard_after_forward: true
    move_params_to_cpu: false

algorithm:
  name: ppo
  epochs: 50
  use_kl_in_reward: true
  kl_coef: 0.1
  gamma: 1.0
  lam: 0.95
  entropy_coef: 0.0

  ratio_clip: 0.2
  value_clip: 0.2
  clip_grad: 1.0
  vf_coef: 0.5

  lr: 5e-6
  lr_schedule: constant
  mini_batch_size: 16
  batch_size: 128
  update_epochs: 4
  target_kl: 0.2
  norm_adv: true
  cliprange_reward: 10
  loss: pg

# 设定自定义奖励函数
custom_reward_function:
  path: {reward_function_path}
  name: {reward_function_name}
  reward_kwargs:
    # 可以在这里添加奖励函数所需的其他参数
    threshold: 0.5

reward_model:
  enable: true
  reward_manager: naive
  strategy: fsdp
  reward_kwargs:
    use_reference_model: false
    sft_ref_scale: 0.0

data:
  train_path: {train_data_path}
  val_path: {val_data_path if val_data_path else ""}
  include_reward_computation_in_dataset: false
  reward_fn_key: data_source
  trust_remote_code: true

trainer:
  nnodes: {nnodes}
  n_gpus_per_node: {n_gpus_per_node}
  batch_log_interval: 1
  eval_interval: 1
  total_episodes: 1000
  profile_gpu_memory: false
  checkpoint_interval: 2
  eval_episodes: 10
  output_dir: {output_dir}

ray_init:
  num_cpus: 32
    """
    
    # 写入配置文件
    with open(output_config_path, 'w', encoding='utf-8') as f:
        f.write(config.strip())
    
    print(f"Generated hydra configuration at {output_config_path}")

# 命令行入口
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="知识图谱抽取数据转换工具")
    parser.add_argument("--input_csv", type=str, required=True, help="输入CSV文件路径")
    parser.add_argument("--output_jsonl", type=str, required=True, help="输出JSONL文件路径")
    parser.add_argument("--train_output", type=str, help="训练集输出路径")
    parser.add_argument("--val_output", type=str, help="验证集输出路径")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--generate_config", action="store_true", help="是否生成配置文件")
    parser.add_argument("--model_path", type=str, help="基础模型路径")
    parser.add_argument("--output_config", type=str, help="输出配置文件路径")
    
    args = parser.parse_args()
    
    # 转换数据
    convert_to_jsonl(args.input_csv, args.output_jsonl)
    
    # 分割数据
    if args.train_output and args.val_output:
        create_train_val_split(
            args.output_jsonl, 
            args.train_output, 
            args.val_output, 
            args.val_ratio
        )
    
    # 生成配置
    if args.generate_config and args.model_path and args.output_config:
        train_path = args.train_output if args.train_output else args.output_jsonl
        val_path = args.val_output if args.val_output else None
        
        generate_hydra_config(
            args.model_path,
            train_path,
            val_path,
            output_config_path=args.output_config
        ) 
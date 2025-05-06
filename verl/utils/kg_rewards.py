import re
import json
import os
from typing import List, Dict, Any, Union, Optional, Type
import torch
from pydantic import BaseModel

# 定义评估所需的模型类
class GraphEvaluationInput(BaseModel):
    """图评估输入"""
    extracted_graph: str
    ground_truth: str

class GraphEvaluationOutput(BaseModel):
    """图评估输出"""
    score: float
    feedback: Optional[str] = None

class ReasoningEvaluationInput(BaseModel):
    """推理评估输入"""
    reasoning: str
    ground_truth: str

class ReasoningEvaluationOutput(BaseModel):
    """推理评估输出"""
    score: float
    feedback: Optional[str] = None

# 辅助函数
def extract_tag_content(text: str, tag: str) -> str:
    """从给定文本中提取被<tag>...</tag>标签包围的内容"""
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else ""

def extract_json_from_answer(answer_text: str) -> str:
    """从<answer>...</answer>标签中提取JSON内容"""
    answer_content = extract_tag_content(answer_text, "answer")
    if answer_content:
        try:
            # 仅验证JSON是否可解析
            json.loads(answer_content)
            return answer_content
        except json.JSONDecodeError:
            return ""
    return ""

def extract_thinking_content(text: str) -> str:
    """从<think>...</think>或<reasoning>...</reasoning>标签中提取思考内容"""
    # 先尝试提取<think>标签内容
    thinking = extract_tag_content(text, "think")
    # 如果没有找到，则尝试提取<reasoning>标签内容
    if not thinking:
        thinking = extract_tag_content(text, "reasoning")
    return thinking

# 格式检查奖励函数
def format_reward(solution_str: str, **kwargs) -> Dict[str, Any]:
    """检查输出是否符合特定格式的奖励函数"""
    # 检查<think>...</think>或<reasoning>...</reasoning>标签
    has_thinking = bool(extract_thinking_content(solution_str))
    
    # 检查<answer>...</answer>标签
    has_answer = bool(extract_tag_content(solution_str, "answer"))
    
    # 计算分数 (0.5分)
    score = 0.5 if (has_thinking and has_answer) else 0.0
    
    return {
        "score": score,
        "format_score": score,
        "has_thinking": has_thinking,
        "has_answer": has_answer
    }

def xml_count_reward(solution_str: str, **kwargs) -> Dict[str, Any]:
    """基于XML标签计数的奖励函数"""
    count = 0.0
    
    # 检查thinking/reasoning标签
    if solution_str.count("<think>") == 1:
        count += 0.125
    elif solution_str.count("<reasoning>") == 1:
        count += 0.125
        
    if solution_str.count("</think>") == 1:
        count += 0.125
    elif solution_str.count("</reasoning>") == 1:
        count += 0.125
    
    # 检查answer标签
    if solution_str.count("<answer>") == 1:
        count += 0.125
    if solution_str.count("</answer>") == 1:
        count += 0.125
    
    # 检查标签之间的顺序是否正确
    if "<think>" in solution_str and "</think>" in solution_str:
        if solution_str.find("<think>") < solution_str.find("</think>"):
            count += 0.125
    elif "<reasoning>" in solution_str and "</reasoning>" in solution_str:
        if solution_str.find("<reasoning>") < solution_str.find("</reasoning>"):
            count += 0.125
            
    if "<answer>" in solution_str and "</answer>" in solution_str:
        if solution_str.find("<answer>") < solution_str.find("</answer>"):
            count += 0.125
    
    # 检查thinking/reasoning在answer之前
    if "</think>" in solution_str and "<answer>" in solution_str:
        if solution_str.find("</think>") < solution_str.find("<answer>"):
            count += 0.125
    elif "</reasoning>" in solution_str and "<answer>" in solution_str:
        if solution_str.find("</reasoning>") < solution_str.find("<answer>"):
            count += 0.125
    
    return {
        "score": count,
        "xml_count": count
    }

# 图正确性奖励函数
def basic_graph_score(extracted_graph: str, ground_truth: str) -> float:
    """基于简单规则对图进行评分"""
    try:
        # 检查JSON有效性
        comp_json = json.loads(extracted_graph)
        truth_json = json.loads(ground_truth)
        
        # 提取基本信息
        comp_nodes = comp_json.get("nodes", [])
        comp_edges = comp_json.get("edges", [])
        truth_nodes = truth_json.get("nodes", [])
        truth_edges = truth_json.get("edges", [])
        
        # 节点数量比较
        if len(comp_nodes) == 0:
            return 0.0
        
        node_ratio = min(len(comp_nodes) / max(1, len(truth_nodes)), 1.0)
        
        # 边数量比较
        if len(truth_edges) == 0:
            edge_ratio = 1.0 if len(comp_edges) == 0 else 0.5
        else:
            edge_ratio = min(len(comp_edges) / max(1, len(truth_edges)), 1.0)
        
        # 计算边属性覆盖率
        edge_attr_score = 0.0
        required_attrs = ["source", "target", "relation", "description", "keywords", "strength", "msg_ids"]
        
        if comp_edges:
            # 检查第一条边是否具有所有必要属性
            first_edge = comp_edges[0]
            attrs_present = sum(1 for attr in required_attrs if attr in first_edge)
            edge_attr_score = attrs_present / len(required_attrs)
        
        # 计算最终分数 (满分2.0)
        score = (node_ratio * 0.4 + edge_ratio * 0.4 + edge_attr_score * 0.2) * 2.0
        return min(score, 2.0)
    
    except Exception as e:
        # JSON解析错误或其他问题
        return 0.0

def graph_correctness_reward(solution_str: str, ground_truth_answer: str, **kwargs) -> Dict[str, Any]:
    """评估抽取的图与标准图的相似度"""
    # 从回答中提取JSON
    extracted_json = extract_tag_content(solution_str, "answer")
    
    # 如果提取失败，返回0分
    if not extracted_json:
        return {
            "score": 0.0,
            "graph_score": 0.0,
            "valid_json": False
        }
    
    # 验证JSON格式
    try:
        json.loads(extracted_json)
        is_valid_json = True
    except json.JSONDecodeError:
        return {
            "score": 0.0,
            "graph_score": 0.0,
            "valid_json": False
        }
    
    # 计算图相似度分数
    score = basic_graph_score(extracted_json, ground_truth_answer) if is_valid_json else 0.0
    
    return {
        "score": score,
        "graph_score": score,
        "valid_json": is_valid_json
    }

# 推理质量奖励函数
def reasoning_quality_reward(solution_str: str, ground_truth_reasoning: str, **kwargs) -> Dict[str, Any]:
    """评估推理的质量"""
    # 提取推理部分
    reasoning = extract_thinking_content(solution_str)
    
    if not reasoning:
        return {
            "score": 0.0,
            "reasoning_score": 0.0,
            "has_reasoning": False
        }
    
    # 计算基础分数 - 使用长度比例作为基础分数
    min_length = 50  # 最小有效长度
    ideal_length = 200  # 理想长度
    
    if len(reasoning) < min_length:
        base_score = 0.2
    else:
        base_score = min(0.8, len(reasoning) / ideal_length * 0.8)
    
    # 关键词检查 - 图相关术语
    graph_keywords = ["node", "edge", "entity", "relationship", "graph"]
    graph_keyword_bonus = sum(1 for kw in graph_keywords if kw.lower() in reasoning.lower()) / len(graph_keywords) * 0.1
    
    # 关键词检查 - 英文专业术语
    domain_keywords = ["relation", "description", "keywords", "strength", "msg_ids"]
    domain_keyword_bonus = sum(1 for kw in domain_keywords if kw.lower() in reasoning.lower()) / len(domain_keywords) * 0.1
    
    # 如果提供了ground_truth_reasoning，可以进行更深入的比较
    comparison_bonus = 0.0
    if ground_truth_reasoning and len(ground_truth_reasoning) > 20:
        # 简单的文本重叠率作为相似度度量
        # 在实际应用中可以使用更复杂的语义相似度算法
        gt_reasoning = ground_truth_reasoning.lower()
        user_reasoning = reasoning.lower()
        
        # 计算词汇重叠率
        gt_words = set(gt_reasoning.split())
        user_words = set(user_reasoning.split())
        if gt_words:
            overlap = len(gt_words.intersection(user_words)) / len(gt_words)
            comparison_bonus = overlap * 0.2
    
    total_score = min(1.0, base_score + graph_keyword_bonus + domain_keyword_bonus + comparison_bonus)
    
    return {
        "score": total_score,
        "reasoning_score": total_score,
        "has_reasoning": True,
        "reasoning_length": len(reasoning)
    }

# 综合奖励函数
def kg_extraction_reward(data_source: str, solution_str: str, ground_truth: dict, extra_info=None) -> Dict[str, Any]:
    """综合奖励函数，整合多个奖励指标"""
    # 获取ground_truth中的答案和推理部分
    ground_truth_answer = ground_truth.get("ground_truth_answer", "{}")
    ground_truth_reasoning = ground_truth.get("ground_truth_reasoning", "")
    
    # 1. 格式奖励
    format_reward_result = format_reward(solution_str)
    
    # 2. XML标签奖励
    xml_reward_result = xml_count_reward(solution_str)
    
    # 3. 图正确性奖励
    graph_reward_result = graph_correctness_reward(solution_str, ground_truth_answer)
    
    # 4. 推理质量奖励
    reasoning_reward_result = reasoning_quality_reward(solution_str, ground_truth_reasoning)
    
    # 计算加权总分
    format_weight = 0.1
    xml_weight = 0.1
    graph_weight = 0.6
    reasoning_weight = 0.2
    
    total_score = (
        format_reward_result["score"] * format_weight +
        xml_reward_result["score"] * xml_weight +
        graph_reward_result["score"] * graph_weight +
        reasoning_reward_result["score"] * reasoning_weight
    )
    
    return {
        "score": total_score,
        "format_score": format_reward_result["score"],
        "xml_count": xml_reward_result["score"],
        "graph_score": graph_reward_result["score"],
        "reasoning_score": reasoning_reward_result["score"]
    }

# 主函数：用于verl框架
def compute_kg_extraction_reward(data_source, solution_str, ground_truth, extra_info=None):
    """知识图谱抽取奖励计算主函数"""
    return kg_extraction_reward(data_source, solution_str, ground_truth, extra_info) 
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

from langgraph.graph import MessagesState

from src.prompts.planner_model import Plan
from src.rag import Resource


class State(MessagesState):
    """State for the agent system, extends MessagesState with next field."""

    # Runtime Variables
    locale: str = "en-US"  # 当前会话的语言环境（如英文、中文），影响交互和内容生成
    observations: list[str] = []  # 研究过程中收集到的观察结果或中间信息（由各agent/工具产生）
    resources: list[Resource] = []  # 研究过程中用到的外部资源（如文献、网页、数据等）
    plan_iterations: int = 0  # 规划器（planner）已进行的规划轮数，用于控制多轮规划
    current_plan: Plan | str = None  # 当前的研究计划（由planner生成，供research_team等执行）
    final_report: str = ""  # 最终生成的研究报告内容（由reporter节点整理输出）
    auto_accepted_plan: bool = False  # 是否自动接受planner生成的计划（否则需要human_feedback人工确认）
    # 是否启用背景调查（由background_investigator节点执行）
    enable_background_investigation: bool = True
    # 背景调查的结果内容（为planner和research_team提供上下文）
    background_investigation_results: str = None

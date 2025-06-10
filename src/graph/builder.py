# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from src.prompts.planner_model import StepType

from .types import State
from .nodes import (
    coordinator_node,
    planner_node,
    reporter_node,
    research_team_node,
    researcher_node,
    coder_node,
    human_feedback_node,
    background_investigation_node,
)


def continue_to_running_research_team(state: State):
    current_plan = state.get("current_plan")#从state中获取当前计划
    if not current_plan or not current_plan.steps:#如果当前计划不存在，则返回planner
        return "planner"
    if all(step.execution_res for step in current_plan.steps):#如果所有步骤都执行成功，则返回到planner
        return "planner"
    for step in current_plan.steps:
        if not step.execution_res:#如果有某个步骤没有得到结果
            break
    if step.step_type and step.step_type == StepType.RESEARCH:#如果步骤类型是研究，则返回researcher
        return "researcher"
    if step.step_type and step.step_type == StepType.PROCESSING:#如果步骤类型是处理，则返回coder
        return "coder"
    return "planner"


def _build_base_graph():
    """Build and return the base state graph with all nodes and edges."""
    # START\END是LangGraph的特殊节点，表示图的开始和结束
    # 其他节点在.nodes中定义
    builder = StateGraph(State)
    builder.add_edge(START, "coordinator")
    builder.add_node("coordinator", coordinator_node)#第一个参数是节点名称，第二个参数是节点函数
    builder.add_node("background_investigator", background_investigation_node)
    builder.add_node("planner", planner_node)
    builder.add_node("reporter", reporter_node)
    builder.add_node("research_team", research_team_node)
    builder.add_node("researcher", researcher_node)
    builder.add_node("coder", coder_node)
    builder.add_node("human_feedback", human_feedback_node)
    builder.add_edge("background_investigator", "planner")
    builder.add_conditional_edges(
        "research_team",
        continue_to_running_research_team,
        ["planner", "researcher", "coder"],
    )#add_conditional_edges用于添加条件边，第一个参数是节点名称，第二个参数是条件函数，第三个参数是条件为真时可以跳转的节点列表
    # 根据前面我们对函数的分析，这里跳转的时候是从列表里三选一，所以三个节点都可能被选中
    builder.add_edge("reporter", END)
    return builder


def build_graph_with_memory():
    """Build and return the agent workflow graph with memory."""
    # use persistent memory to save conversation history
    # TODO: be compatible with SQLite / PostgreSQL
    memory = MemorySaver()

    # build state graph
    builder = _build_base_graph()
    return builder.compile(checkpointer=memory)


def build_graph():
    """Build and return the agent workflow graph without memory."""
    # build state graph
    builder = _build_base_graph()#构建基础图
    return builder.compile()#编译图，用于将图结构转为可运行的graph对象


graph = build_graph()

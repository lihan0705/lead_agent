"""
GAIA Evaluation Example

This example demonstrates how to use the GAIA evaluation tool to evaluate a superagent.
"""

from pathlib import Path

from superagent import create_simple_agent
from superagent.llm import build_qwen_llm

# GAIA官方系统提示词（来自论文）
GAIA_SYSTEM_PROMPT = """You are a general AI assistant. I will ask you a question. Report your thoughts, and finish your answer with the following template: FINAL ANSWER: [YOUR FINAL ANSWER].

YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings.

If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise.

If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise.

If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string."""


class SuperAgentWrapper:
    """Wrapper for superagent to make it compatible with GAIA evaluation tool"""
    
    def __init__(self, 
                 working_dir: str | None = None,
                 auto_approve: bool = False):
        """
        Initialize superagent wrapper
        
        Args:
            working_dir: Working directory for file operations (should be GAIA dataset directory)
            auto_approve: Whether to auto-approve all tool calls
        """
        # Use build_qwen_llm to get the Qwen model
        self.model = build_qwen_llm()
        self.working_dir = working_dir or str(Path.cwd())
        self.auto_approve = auto_approve
        
        # Create superagent with GAIA system prompt
        self.agent = create_simple_agent(
            model=self.model,
            working_dir=self.working_dir,
            auto_approve=auto_approve,
            enable_subagents=True,
            enable_memory=True,
        )
    
    def run(self, question: str) -> str:
        """
        Run the agent on a question
        
        Args:
            question: The question to answer
            
        Returns:
            Agent's response
        """
        # Invoke the agent with the question
        result = self.agent.invoke({
            "messages": [
                {"role": "system", "content": GAIA_SYSTEM_PROMPT},
                {"role": "user", "content": question}
            ]
        })
        
        # Extract the final response
        messages = result.get("messages", [])
        if messages:
            # Get the last message from the assistant
            for msg in reversed(messages):
                if msg.get("role") == "assistant":
                    return msg.get("content", "")
        
        return ""


def evaluate_gaia_with_agent(agent, level=1, max_samples=5):
    """
    Evaluate an agent on GAIA benchmark
    
    Args:
        agent: Agent instance with a run() method
        level: GAIA difficulty level (1, 2, or 3)
        max_samples: Maximum number of samples to evaluate
        
    Returns:
        Evaluation results dictionary
    """
    from evaluation.tools import GAIAEvaluationTool
    
    # 创建GAIA评估工具
    gaia_tool = GAIAEvaluationTool()
    
    # 运行评估
    results = gaia_tool.run(
        agent=agent,
        level=level,
        max_samples=max_samples,
        export_results=True,
        generate_report=True
    )
    
    # 查看结果
    print(f"精确匹配率: {results['exact_match_rate']:.2%}")
    print(f"部分匹配率: {results['partial_match_rate']:.2%}")
    print(f"正确数: {results['exact_matches']}/{results['total_samples']}")
    
    return results


# Example usage with superagent
if __name__ == "__main__":
    # Create superagent wrapper
    # working_dir should be the GAIA dataset directory where you downloaded the data
    # This is typically the path returned by download_gaia.py
    agent = SuperAgentWrapper(
        working_dir="/Users/lihan/Documents/lead_agent/gaia_data",  # GAIA数据集目录
        auto_approve=False  # Set to True to auto-approve all tool calls
    )
    
    # Evaluate on GAIA
    results = evaluate_gaia_with_agent(agent, level=1, max_samples=5)
    
    print("\n" + "="*60)
    print("Evaluation completed!")
    print("="*60)

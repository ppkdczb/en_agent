# config.py
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    max_loop: int = 15
    batchsize: int = 10 # 没用到
    searchdocs: int = 10  #搜索到的相关文档数量
    run_times: int = 1  # current run number (for directory naming)
    database_path: str = Path(__file__).resolve().parent.parent / "data"
    run_directory: str = Path(__file__).resolve().parent.parent / "runs"
    case_dir: str = ""
    max_time_limit: int = 3600 # Max time limit after which the openfoam run will be terminated, in seconds
    file_dependency_threshold: int = 3000 # threshold length on the similar case; see `nodes/architect_node.py` for details
    model_provider: str = "openai"# [openai, ollama, bedrock]
    # model_version should be in ["gpt-4o", "deepseek-r1:32b-qwen-distill-fp16", "qwen2.5:32b-instruct"]
    model_version: str = "deepseek-chat"
    temperature: float = 1.0
    
    
##claude-haiku-4-5-20251001

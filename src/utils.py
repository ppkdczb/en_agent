# utils.py
from optparse import Option
import re
import subprocess
import os
from token import OP
from typing import Optional, Any, Type, TypedDict, List
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from pathlib import Path
from model import ClozeTest, ReadingTask
from sklearn import base
import requests
import time
import random
import shutil
from botocore.exceptions import ClientError
import json

class ResponseWithThinkPydantic(BaseModel):
    think: str = Field(description="Thought process of the LLM")
    response: str = Field(description="Response of the LLM")

def _extract_json_object(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return text
    try:
        json.loads(text)
        return text
    except Exception:
        pass

    # Try to extract the first {...} or [...] block.
    m = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
    if m:
        return m.group(1).strip()
    return text

class LLMService:
    def __init__(self, config: object):
        self.model_version = getattr(config, "model_version", "gpt-4o")
        self.temperature = getattr(config, "temperature", 0)
        self.model_provider = getattr(config, "model_provider", "openai")
        
        # Initialize statistics
        self.total_calls = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.failed_calls = 0
        self.retry_count = 0
        self.thinking = getattr(config, "thinking", False)
        # Initialize the LLM
        try:
            self.llm = init_chat_model(
                self.model_version, 
                model_provider=self.model_provider, 
                temperature=self.temperature,
            )
        except Exception as e:
            raise Exception(f"Failed to initialize LLM: {str(e)}")
    
    def invoke(self, 
              user_prompt: str, 
              system_prompt: Optional[str] = None, 
              pydantic_obj: Optional[Type[BaseModel]] = None,
              max_retries: int = 10, is_thinking: bool = None) -> Any:
        """
        Invoke the LLM with the given prompts and return the response.
        
        Args:
            user_prompt: The user's prompt
            system_prompt: Optional system prompt
            pydantic_obj: Optional Pydantic model for structured output
            max_retries: Maximum number of retries for throttling errors
            
        Returns:
            The LLM response with token usage statistics
        """
        self.total_calls += 1
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        
        # Calculate prompt tokens
        prompt_tokens = 0
        for message in messages:
            prompt_tokens += self.llm.get_num_tokens(message["content"])
        
        retry_count = 0
        while True:
            try:
                if pydantic_obj:
                    try:
                        structured_llm = self.llm.with_structured_output(pydantic_obj)
                        response = structured_llm.invoke(messages)
                    except Exception as e:
                        # Some providers/models reject certain `response_format` / schema modes.
                        # Fallback: ask for strict JSON and validate with Pydantic locally.
                        field_names = list(getattr(pydantic_obj, "model_fields", {}).keys())
                        fallback_system = (
                            "Return ONLY a valid JSON value (no Markdown, no code fences). "
                            f"Top-level keys MUST be: {field_names}."
                        )
                        fallback_messages = messages.copy()
                        fallback_messages.insert(0, {"role": "system", "content": fallback_system})

                        raw = self.llm.invoke(fallback_messages)
                        raw_text = raw.content if hasattr(raw, "content") else str(raw)
                        raw_text = _extract_json_object(raw_text)
                        parsed = json.loads(raw_text)
                        response = pydantic_obj.model_validate(parsed)
                else:
                    if self.model_version.startswith("deepseek"):
                        structured_llm = self.llm.with_structured_output(ResponseWithThinkPydantic)
                        response = structured_llm.invoke(messages)
                        #print(response)
                        # Extract the resposne without the think
                        response = response.response
                    else:
                        response = self.llm.invoke(messages)
                        response = response.content

                # Calculate completion tokens
                response_content = str(response)
                completion_tokens = self.llm.get_num_tokens(response_content)
                total_tokens = prompt_tokens + completion_tokens
                
                # Update statistics
                self.total_prompt_tokens += prompt_tokens
                self.total_completion_tokens += completion_tokens
                self.total_tokens += total_tokens
                
                return response
                
            except ClientError as e:
                if e.response['Error']['Code'] == 'Throttling' or e.response['Error']['Code'] == 'TooManyRequestsException':
                    retry_count += 1
                    self.retry_count += 1
                    
                    if retry_count > max_retries:
                        self.failed_calls += 1
                        raise Exception(f"Maximum retries ({max_retries}) exceeded: {str(e)}")
                    
                    base_delay = 1.0
                    max_delay = 60.0
                    delay = min(max_delay, base_delay * (2 ** (retry_count - 1)))
                    jitter = random.uniform(0, 0.1 * delay)
                    sleep_time = delay + jitter
                    
                    print(f"ThrottlingException occurred: {str(e)}. Retrying in {sleep_time:.2f} seconds (attempt {retry_count}/{max_retries})")
                    time.sleep(sleep_time)
                else:
                    self.failed_calls += 1
                    raise e
            except Exception as e:
                self.failed_calls += 1
                raise e
    
    def get_statistics(self) -> dict:
        """
        Get the current statistics of the LLM service.
        
        Returns:
            Dictionary containing various statistics
        """
        return {
            "total_calls": self.total_calls,
            "failed_calls": self.failed_calls,
            "retry_count": self.retry_count,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "average_prompt_tokens": self.total_prompt_tokens / self.total_calls if self.total_calls > 0 else 0,
            "average_completion_tokens": self.total_completion_tokens / self.total_calls if self.total_calls > 0 else 0,
            "average_tokens": self.total_tokens / self.total_calls if self.total_calls > 0 else 0
        }
    
    def print_statistics(self) -> None:
        """
        Print the current statistics of the LLM service.
        """
        stats = self.get_statistics()
        print("\n<LLM Service Statistics>")
        print(f"Total calls: {stats['total_calls']}")
        print(f"Failed calls: {stats['failed_calls']}")
        print(f"Total retries: {stats['retry_count']}")
        print(f"Total prompt tokens: {stats['total_prompt_tokens']}")
        print(f"Total completion tokens: {stats['total_completion_tokens']}")
        print(f"Total tokens: {stats['total_tokens']}")
        print(f"Average prompt tokens per call: {stats['average_prompt_tokens']:.2f}")
        print(f"Average completion tokens per call: {stats['average_completion_tokens']:.2f}")
        print(f"Average tokens per call: {stats['average_tokens']:.2f}\n")
        print("</LLM Service Statistics>")


class GraphState(TypedDict):
    Cloze: Optional[list[ClozeTest]]
    Reading: Optional[list[ReadingTask]]
    llm_service: Optional['LLMService']
    Cloze_answers: Optional[dict[str, Any]]
    Reading_answers: Optional[dict[str, Any]]


def tokenize(text: str) -> str:
    # Replace underscores with spaces
    text = text.replace('_', ' ')
    # Insert a space between a lowercase letter and an uppercase letter (global match)
    text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)
    return text.lower()

def save_file(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(content)
    print(f"Saved file at {path}")

def read_file(path: str) -> str:
    if os.path.exists(path):
        with open(path, 'r') as f:
            return f.read()
    return ""

def list_case_files(case_dir: str) -> str:
    files = [f for f in os.listdir(case_dir) if os.path.isfile(os.path.join(case_dir, f))]
    return ", ".join(files)

def remove_files(directory: str, prefix: str) -> None:
    for file in os.listdir(directory):
        if file.startswith(prefix):
            os.remove(os.path.join(directory, file))
    print(f"Removed files with prefix '{prefix}' in {directory}")

def remove_file(path: str) -> None:
    if os.path.exists(path):
        os.remove(path)
        print(f"Removed file {path}")

def remove_numeric_folders(case_dir: str) -> None:
    """
    Remove all folders in case_dir that represent numeric values, including those with decimal points,
    except for the "0" folder.
    
    Args:
        case_dir (str): The directory path to process
    """
    for item in os.listdir(case_dir):
        item_path = os.path.join(case_dir, item)
        if os.path.isdir(item_path) and item != "0":
            try:
                # Try to convert to float to check if it's a numeric value
                float(item)
                # If conversion succeeds, it's a numeric folder
                try:
                    shutil.rmtree(item_path)
                    print(f"Removed numeric folder: {item_path}")
                except Exception as e:
                    print(f"Error removing folder {item_path}: {str(e)}")
            except ValueError:
                # Not a numeric value, so we keep this folder
                pass

def check_foam_errors(directory: str) -> list:
    error_logs = []
    # DOTALL mode allows '.' to match newline characters
    pattern = re.compile(r"ERROR:(.*)", re.DOTALL)
    
    for file in os.listdir(directory):
        if file.startswith("log"):
            filepath = os.path.join(directory, file)
            with open(filepath, 'r') as f:
                content = f.read()
            
            match = pattern.search(content)
            if match:
                error_content = match.group(0).strip()
                error_logs.append({"file": file, "error_content": error_content})
            elif "error" in content.lower():
                print(f"Warning: file {file} contains 'error' but does not match expected format.")
    return error_logs

def extract_commands_from_allrun_out(out_file: str) -> list:
    commands = []
    if not os.path.exists(out_file):
        return commands
    with open(out_file, 'r') as f:
        for line in f:
            if line.startswith("Running "):
                parts = line.split(" ")
                if len(parts) > 1:
                    commands.append(parts[1].strip())
    return commands

def parse_case_name(text: str) -> str:
    match = re.search(r'case name:\s*(.+)', text, re.IGNORECASE)
    return match.group(1).strip() if match else "default_case"

def split_subtasks(text: str) -> list:
    header_match = re.search(r'splits into (\d+) subtasks:', text, re.IGNORECASE)
    if not header_match:
        print("Warning: No subtasks header found in the response.")
        return []
    num_subtasks = int(header_match.group(1))
    subtasks = re.findall(r'subtask\d+:\s*(.*)', text, re.IGNORECASE)
    if len(subtasks) != num_subtasks:
        print(f"Warning: Expected {num_subtasks} subtasks but found {len(subtasks)}.")
    return subtasks

# 截取 </think> 之后的内容
def remove_think_tags(text: str) -> str:
    think_end_idx = str(text).find("</think>")
    if think_end_idx != -1:
        text = text[think_end_idx + len("</think>") :]
        print("已去除 </think> 及之前的内容")
    return text

def parse_context(text: str) -> str:
    text = remove_think_tags(text)
    match = re.search(r'FoamFile\s*\{.*?(?=```|$)', text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(0).strip()
    
    print("Warning: Could not parse context; returning original text.")
    return text


def parse_file_name(subtask: str) -> str:
    subtask = remove_think_tags(subtask)    
    match = re.search(r'openfoam\s+(.*?)\s+foamfile', subtask, re.IGNORECASE)
    return match.group(1).strip() if match else ""

def parse_json_content(subtask: str, ) -> str:
    subtask_nothink = remove_think_tags(subtask)    
    match = re.search(r'```(?:json)?\s*(.*?)\s*```', subtask_nothink, re.DOTALL)
    return match.group(1).strip() if match else subtask_nothink

def parse_folder_name(subtask: str) -> str:
    subtask = remove_think_tags(subtask)    
    match = re.search(r'foamfile in\s+(.*?)\s+folder', subtask, re.IGNORECASE)
    return match.group(1).strip() if match else ""

def find_similar_file(description: str, tutorial: str) -> str:
    start_pos = tutorial.find(description)
    if start_pos == -1:
        return "None"
    end_marker = "input_file_end."
    end_pos = tutorial.find(end_marker, start_pos)
    if end_pos == -1:
        return "None"
    return tutorial[start_pos:end_pos + len(end_marker)]

def read_commands(file_path: str) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Commands file not found: {file_path}")
    with open(file_path, 'r') as f:
        # join non-empty lines with a comma
        return ", ".join(line.strip() for line in f if line.strip())

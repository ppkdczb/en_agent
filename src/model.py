import re
from typing import List, Literal, Optional
from pydantic import BaseModel, field_validator, model_validator

# 定义单个题目的选项结构
class OptionGroup(BaseModel):
    A: str
    B: str
    C: str
    D: str

class ClozeTest(BaseModel):
    id: int
    #title: Optional[str] = None
    content: str  # 例如: "I am <question_1> student."
    options: List[OptionGroup]
    answers: List[Literal["A", "B", "C", "D"]]


# 阅读理解 单个问题 ---
class QuestionItem(BaseModel):
    id: int                     # 题号，方便排序或校验
    prompt: str                 # 题干，例如 "作者为什么感到伤心？"
    options: OptionGroup        # 该题的四个选项
    answer: Literal["A", "B", "C", "D"] # 答案
    explanation: Optional[str] = None   # (可选) 答案解析

    # 针对单个问题的校验：自动转大写
    @field_validator('answer', mode='before')
    @classmethod
    def upper_answer(cls, v):
        return str(v).strip().upper() if v else v

# --- 3. 最外层：阅读理解任务 ---
class ReadingTask(BaseModel):
    title: str
    content: str                # 文章正文
    questions: List[QuestionItem] # 问题列表

if __name__ == "__main__":

    # --- 测试运行 ---

    # 1. 正确数据
    data_ok = {
        "id": 1,
        # 注意这里的占位符格式
        "content": "Hello <question_0>, nice to <question_1> you.",
        "options": [
            {"A": "World", "B": "Space", "C": "Moon", "D": "Sun"},
            {"A": "see", "B": "meet", "C": "hit", "D": "kick"}
        ],
        "answers": ["A", "B"] # 小写会被自动转为大写
    }
   # print(data_ok)
    try:
        test = ClozeTest(**data_ok)
        print("✅ 验证通过！数据结构正确。")
    except Exception as e:
        print(e)

    print("-" * 30)

    data = {
        "title": "科技对生活的影响",
        "content": "随着人工智能的发展，我们的生活发生了翻天覆地的变化...",
        "questions": [
            {
                "id": 1,
                "prompt": "文章主要讨论了什么？",
                "options": {
                    "A": "环境保护",
                    "B": "AI 的发展",
                    "C": "历史变迁",
                    "D": "经济危机"
                },
                "answer": "B",  # 小写 b 会自动转大写
                "explanation": "文章第一句就提到了人工智能的发展。"
            },
            {
                "id": 2,
                "prompt": "作者的态度是什么？",
                "options": {
                    "A": "乐观",
                    "B": "悲观",
                    "C": "中立",
                    "D": "愤怒"
                },
                "answer": "A"
            }
        ]
    }

    try:
        task = ReadingTask(**data)
        print(f"《{task.title}》加载成功，包含 {len(task.questions)} 道题。")
        print(f"第一题答案: {task.questions[0].answer}")
        print(f"第一题解析: {task.questions[0].explanation}")
        
    except Exception as e:
        print(f"校验失败: {e}")
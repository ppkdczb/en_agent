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
    title: str
    content: str  # 例如: "I am <question_1> student."
    options: List[OptionGroup]
    answers: List[Literal["A", "B", "C", "D"]]

    # 1. 答案自动转大写
    @field_validator('answers', mode='before')
    @classmethod
    def normalize_answers(cls, v):
        if isinstance(v, list):
            # 转字符串 -> 去空格 -> 转大写
            return [str(item).strip().upper() for item in v]
        return v

    # 2. 核心校验：检查占位符
    @model_validator(mode='after')
    def check_consistency(self) -> 'ClozeTest':
        # 正则解释：
        # <question_  匹配字面量
        # (\d+)       匹配数字并捕获（Capture Group）
        # >           匹配结束符号
        matches = re.findall(r'<question_(\d+)>', self.content)
        
        # 将捕获到的数字转为整数列表，例如 [1, 2, 3]
        question_numbers = [int(n) for n in matches]
        
        blank_count = len(question_numbers)
        options_count = len(self.options)
        answers_count = len(self.answers)

        # 校验 A: 数量必须一致
        if not (blank_count == options_count == answers_count):
            raise ValueError(
                f"数量不匹配！\n"
                f"- 文章挖空数: {blank_count}\n"
                f"- 选项组数: {options_count}\n"
                f"- 答案数: {answers_count}"
            )
        
        # 校验 B (可选但推荐): 检查序号是否从 1 开始且连续
        # 比如 content 里只有 <question_1> 和 <question_3>，少了 2，这里可以报错
        if question_numbers:
            expected_numbers = list(range(1, blank_count + 1))
            if question_numbers != expected_numbers:
                raise ValueError(
                    f"题目序号异常: 期望顺序 {expected_numbers}, "
                    f"实际检测到 {question_numbers}。请检查是否跳号或乱序。"
                )

        return self


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

    # 校验逻辑：检查题号是否重复，是否连续
    @model_validator(mode='after')
    def check_question_ids(self) -> 'ReadingTask':
        if not self.questions:
            raise ValueError("阅读理解必须至少包含一道题目")

        # 提取所有题号
        ids = [q.id for q in self.questions]
        
        # 检查重复
        if len(ids) != len(set(ids)):
            raise ValueError(f"题号存在重复: {ids}")
        
        # 检查是否按顺序 (可选，视需求而定)
        # 比如必须是 1, 2, 3...
        expected = list(range(1, len(ids) + 1))
        # 这里的排序是为了防止输入乱序，先排个序再比对
        if sorted(ids) != expected:
            raise ValueError(f"题号不连续或不从1开始: 实际为 {sorted(ids)}")
            
        return self







if __name__ == "__main__":

    # --- 测试运行 ---

    # 1. 正确数据
    data_ok = {
        "title": "New Format Test",
        # 注意这里的占位符格式
        "content": "Hello <question_0>, nice to <question_1> you.",
        "options": [
            {"A": "World", "B": "Space", "C": "Moon", "D": "Sun"},
            {"A": "see", "B": "meet", "C": "hit", "D": "kick"}
        ],
        "answers": ["A", "B"] # 小写会被自动转为大写
    }
    print(data_ok)
    try:
        test = ClozeTest(**data_ok)
        print("✅ 验证通过！数据结构正确。")
    except Exception as e:
        print(e)

    print("-" * 30)

    # 2. 错误数据（跳过了序号2，直接写了3）
    data_bad = {
        "title": "Bad Index",
        "content": "One <question_1>, Three <question_3>.", # 跳过了 2
        "options": [
            {"A": "1", "B": "1", "C": "1", "D": "1"},
            {"A": "3", "B": "3", "C": "3", "D": "3"}
        ],
        "answers": ["A", "A"]
    }

    try:
        ClozeTest(**data_bad)
    except Exception as e:
        print("❌ 捕获到预期错误：")
        # 打印具体的错误信息（通常在 e.errors() 或 str(e) 中）
        print(e)


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
                "answer": "b",  # 小写 b 会自动转大写
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
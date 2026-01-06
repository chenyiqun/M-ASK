import os
import re
import time
import json
import logging
import asyncio
from typing import List, Dict, Optional, Any
import requests
from qa_manager.llm_api import LLM_Client
import random

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

EXAMPLE_PROMPT = '''
- Example:
Question: When did the simpsons first air on television?
Answer: <final_answer>December 17, 1989</final_answer>

Question: When did the lightning thief book come out?
Answer: <final_answer>2005</final_answer>

Question: Who said i'm late i'm late for a very important date?
Answer: <final_answer>The White Rabbit</final_answer>

Question: Where does the short happy life of francis macomber take place?
Answer: <final_answer>Africa</final_answer>

Question: What was the fourth expansion pack for sims 2?
Answer: <final_answer>Pets</final_answer>

Question: Voice of the snake in the jungle book?
Answer: <final_answer>The Jungle Book (2016 film)</final_answer>

Question: How many seasons are there of star wars the clone wars?
Answer: <final_answer>6</final_answer>

Question: Which us president appears as a character in the play annie?
Answer: <final_answer>Franklin D. Roosevelt</final_answer>

Question: Are Calochone and Adlumia both plants?
Answer: <final_answer>yes</final_answer>

Question: Yukio Mishima and Roberto Bolaño, are Chilean?
Answer: <final_answer>no</final_answer>
'''

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import threading

class BaseLLMClient:
    TRACE_DIR = Path("save_path")
    _trace_lock = threading.Lock()  # Class-level lock, shared by all instances

    def __init__(
        self,
        model_name: str = "gpt4o",
        system_jinja2_path: Optional[str] = None,
        user_jinja2_path: Optional[str] = None,
        default_temperature: float = 0.5,
        max_retries: int = 3,
        retry_interval: float = 1.0,
    ):
        self.model_name = model_name
        self.system_jinja2_path = system_jinja2_path or "/mnt/default/system_prompt.jinja2"
        self.user_jinja2_path = user_jinja2_path or "/mnt/default/user_prompt.jinja2"
        self.default_temperature = default_temperature
        self.max_retries = max_retries
        self.retry_interval = retry_interval
        self.enable_trace = True  # Save switch

        logging.info(f"初始化 LLM_Client: model={model_name}")
        self.llm_client = LLM_Client(
            model_name=self.model_name,
            system_jinja2_path=self.system_jinja2_path,
            user_jinja2_path=self.user_jinja2_path
        )

        self.EXAMPLE_PROMPT = EXAMPLE_PROMPT

    def get_response(self, messages: List[Dict[str, str]], temperature: Optional[float] = None) -> str:
        temp = temperature if temperature is not None else self.default_temperature
        for attempt in range(1, self.max_retries + 1):
            try:
                res, _ = self.llm_client.query_model(messages=messages, temperature=temp)
                return res[0]['message']['content']
            except Exception as e:
                logging.error(f"Failed to invoke LLM：{e}")
                if attempt < self.max_retries:
                    time.sleep(self.retry_interval)
        return "[ERROR] Failed to invoke LLM"

    async def async_get_response(
        self, 
        messages: List[Dict[str, str]], 
        agent_name: str,  
        temperature: Optional[float] = None
    ) -> str:
        """Asynchronous invocation and recording of trace data (optional)"""
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, self.get_response, messages, temperature)

        # Only save if enable_trace is True
        if self.enable_trace:
            self._record_trace(messages, response, temperature, agent_name)

        return response

    def _record_trace(
        self, 
        messages: List[Dict[str, str]], 
        response: str, 
        temperature: Optional[float],
        agent_name: str
    ):
        """Append the called data and write it to the JSONL file (thread-safe, including agent_name)"""
        self.TRACE_DIR.mkdir(parents=True, exist_ok=True)
        date_str = datetime.now().strftime("%Y%m%d")
        trace_file = self.TRACE_DIR / f"{date_str}.jsonl"

        trace_entry = {
            "timestamp": datetime.now().isoformat(),
            "model_name": self.model_name,
            "agent_name": agent_name,
            "temperature": temperature if temperature is not None else self.default_temperature,
            "messages": messages,
            "response": response
        }

        with BaseLLMClient._trace_lock:
            with open(trace_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(trace_entry, ensure_ascii=False) + "\n")

    async def async_batch_get_response(
        self,
        batch_messages: List[List[Dict[str, str]]],
        agent_name: str,
        temperature: Optional[float] = None
    ) -> List[str]:
        "Batch asynchronous invocation, batch recording of trace (controlled by enable_trace)"
        tasks = [
            self.async_get_response(msg, agent_name, temperature) 
            for msg in batch_messages
        ]
        results = await asyncio.gather(*tasks)
        return results


class PlanningAgent(BaseLLMClient):
    """
    Input a question, and output the dissection思路 of LLM for that question：
    Multiple sub-questions and their corresponding sub-answers.
    The output format is as instructed by LLM： <qN>... </qN> <aN>... </aN>
    If the sub-answer is unknown, write 'unkown'.
    """

    def __init__(
        self,
        model_name: str = "gpt4o",
        system_jinja2_path: Optional[str] = None,
        user_jinja2_path: Optional[str] = None,
        default_temperature: float = 0.0
    ):
        super().__init__(
            model_name=model_name,
            system_jinja2_path=system_jinja2_path,
            user_jinja2_path=user_jinja2_path,
            default_temperature=default_temperature
        )

    def _build_messages(self, query: str) -> List[dict]:
        """
        Construct a prompt to make the LLM output the disassembly results in the specified label format.
        """
        return [
            {
                "role": "system",
                "content": (
                    "You are a knowledgeable reasoning assistant. "
                    "Your task is to decompose a given question into multiple sub-questions "
                    "based purely on your own knowledge (do not use external tools), "
                    "and answer each sub-question based on your own knowledge. "
                    "If you do not know the answer, write 'unkown' exactly. "
                    "Output all results strictly in the following format for easy parsing:\n"
                    "<q1>sub-question text</q1>\n<a1>answer text</a1>\n<q2>...</q2>\n<a2>...</a2>\n"
                    "Continue numbering sequentially until done."
                    "At the end, output your final conclusion answer to the original question "
                    "in the tag <final_answer>...</final_answer>, and inside this tag follow the style of:\n"
                    f"{self.EXAMPLE_PROMPT}\n"
                    "Do not add explanations inside <final_answer>, only the direct QA pair in the above style."
                )
            },
            {
                "role": "user",
                "content": f"Question: {query}"
            }
        ]

    @staticmethod
    def parse_response(response_text: str, original_question: str) -> Dict[str, Any]:
        """
        Parse the output of LLM into a structured dictionary and perform strict format checking. 
        Rules:
            1. Only the structure <qN>...</qN>, <aN>...</aN>, and <final_answer>...</final_answer> is allowed in the output. Any other content is considered illegal.
            2. The output must strictly follow the sequence q1 a1 q2 a2 ... qN aN.
            3. The number of questions <= 5.
            4. There must be no duplicate question contents.
        """

        def _safe_strip(x):
            """Remove the leading and trailing spaces from the string. This rule only applies when x is a string."""
            return x.strip() if isinstance(x, str) else x

        is_legal_format = True  # Default legality

        # When there is no content, return a default response.
        if not response_text or not isinstance(response_text, str):
            return {
                "question": original_question,
                "thinking_trajectory": {
                    "t1": {"question": original_question, "answer": "unkown"}
                },
                "final_answer": "unkown",
                "is_legal_format": False
            }

        trajectory = {}

        # Match the legal <qN> and <aN> contents
        q_matches = re.findall(r"<q(\d+)>(.*?)</q\1>", response_text, re.DOTALL | re.IGNORECASE)
        a_matches = re.findall(r"<a(\d+)>(.*?)</a\1>", response_text, re.DOTALL | re.IGNORECASE)

        # Convert to dictionary to facilitate subsequent assembly by serial number
        q_dict = {int(num): _safe_strip(text) for num, text in q_matches}
        a_dict = {int(num): _safe_strip(text) for num, text in a_matches}

        # If no valid tags are matched, then perform fallback parsing
        if not q_dict and not a_dict:
            is_legal_format = False  # Without legal labels, it is directly illegal
            lines = [l.strip() for l in response_text.strip().split("\n") if l.strip()]
            for idx, line in enumerate(lines, start=1):
                if ":" in line:
                    parts = line.split(":", 1)
                    trajectory[f"t{idx}"] = {
                        "question": parts[0],
                        "answer": parts[1] or "unkown"
                    }
            if not trajectory:
                trajectory["t1"] = {"question": original_question, "answer": "unkown"}
        else:
            # Normal assembly sequence numbers: t1, t2...
            for idx in sorted(set(q_dict.keys()) | set(a_dict.keys())):
                trajectory[f"t{idx}"] = {
                    "question": q_dict.get(idx, ""),
                    "answer": a_dict.get(idx, "unkown")
                }

        # final_answer match
        final_match = re.search(r"<final_answer>(.*?)</final_answer>", response_text, re.DOTALL | re.IGNORECASE)
        if final_match:
            final_answer = _safe_strip(final_match.group(1))
        else:
            # When there is no "final_answer", the fallback option is to take the last row.
            candidates = [l.strip() for l in response_text.strip().split("\n") if l.strip()]
            final_answer = candidates[-1] if candidates else "unkown"
            if len(final_answer) > 500:
                final_answer = "unkown"

        # ===== Rule Detection =====

        # 1) Check whether the content is entirely composed of legal labels.
        allowed_pattern = r'(?:<q\d+>.*?</q\d+>|<a\d+>.*?</a\d+>|<final_answer>.*?</final_answer>)'
        allowed_segments = re.findall(allowed_pattern, response_text, re.DOTALL | re.IGNORECASE)
        reconstructed = ''.join(allowed_segments)
        # Remove the spaces/line breaks, then compare the original string with the valid fragment string. If they are not consistent, it indicates the presence of additional illegal content.
        if re.sub(r'\s+', '', response_text) != re.sub(r'\s+', '', reconstructed):
            is_legal_format = False

        # 2) Is the detection sequence q1 a1 q2 a2...? qN aN
        tags_in_order = re.findall(r"<\s*(q|a)(\d+)\s*>", response_text, re.IGNORECASE)
        if tags_in_order:
            max_num = max(int(n) for _, n in tags_in_order)
            expected_order = []
            for i in range(1, max_num + 1):
                expected_order.append(('q', str(i)))
                expected_order.append(('a', str(i)))
            if tags_in_order != expected_order[:len(tags_in_order)]:
                is_legal_format = False
        else:
            is_legal_format = False

        # 3) Number of tests and repetition
        questions_list = [v["question"] for v in trajectory.values()]
        if len(questions_list) > 5:
            is_legal_format = False
        elif len(set(questions_list)) < len(questions_list):
            is_legal_format = False

        return {
            "question": original_question,
            "thinking_trajectory": trajectory,
            "final_answer": final_answer or "unkown",
            "is_legal_format": is_legal_format
        }

    def get_results(self, query: str, temperature: Optional[float] = None) -> Dict[str, Any]:
        """
        Break down a single question and return a structured dictionary
        """
        messages = self._build_messages(query)
        response_text = self.get_response(messages, temperature)
        return self.parse_response(response_text, query)

    async def async_get_results(self, query: str, temperature: Optional[float] = None) -> Dict[str, Any]:
        """
        Asynchronous processing of a single issue
        """
        messages = self._build_messages(query)
        response_text = await self.async_get_response(messages, 'PlanningAgent', temperature)
        return self.parse_response(response_text, query)

    async def get_results_parallel(
        self,
        queries: List[str],
        temperature: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Process multiple issues in parallel
        """
        batch_messages = [self._build_messages(q) for q in queries]
        raw_results = await self.async_batch_get_response(batch_messages, 'PlanningAgent', temperature)
        return [self.parse_response(r, q) for r, q in zip(raw_results, queries)]

class SearchAgent(BaseLLMClient):
    """
    输入一个包含原始问题和已知思考轨迹（thinking_trajectory）的 dict，
    输出一个可搜索的 query 或者 <end>。
    可搜索的 query 指具有明确的实体或事实方向，可以被检索引擎搜索。
    模型需遵循如下逻辑：
      - 如果 thinking_trajectory 中有 sub-question 尚未得到值得信赖的 sub-answer（'unkown'或不明确），
        则针对该 sub-question 提出一个可搜索的 query。
      - 如果思考轨迹中存在需要补充的新 sub-question（通过原始问题判断），
        则提出新的可搜索 query。
      - 如果不需要进行进一步搜索，则输出 <end>。
    输出严格限制为：
      <search>xxx</search> 或 <end>
    不能出现其他内容。
    """

    def __init__(
        self,
        model_name: str = "gpt4o",
        system_jinja2_path: Optional[str] = None,
        user_jinja2_path: Optional[str] = None,
        default_temperature: float = 0.0  # 保持稳定输出格式
    ):
        super().__init__(
            model_name=model_name,
            system_jinja2_path=system_jinja2_path,
            user_jinja2_path=user_jinja2_path,
            default_temperature=default_temperature
        )

    def _build_messages(self, data: Dict[str, Any], search_history: List[dict]) -> List[dict]:
        """
        构造 prompt，让 LLM 根据已有的轨迹数据输出下一个可搜索 query 或结束标志。
        """
        question = data.get("question", "")
        trajectory = data.get("thinking_trajectory", {})

        # 格式化思考轨迹
        trajectory_text_lines = []
        for k, qa in trajectory.items():
            trajectory_text_lines.append(f"{k}: Q: {qa.get('question', '')} | A: {qa.get('answer', '')}")
        trajectory_text = "\n".join(trajectory_text_lines)

        # 格式化搜索历史
        if search_history:
            history_lines = []
            for idx, h in enumerate(search_history, start=1):
                history_lines.append(f"{idx}. Query: {h.get('search_query', '')} | Summary: {h.get('summarization', '')}")
            history_text = "\n".join(history_lines)
        else:
            history_text = "(No previous searches)"

        return [
            {
                "role": "system",
                "content": (
                    "You are a search query generation assistant.\n"
                    "Your input contains:\n"
                    " - The original question\n"
                    " - A thinking_trajectory (list of sub-questions and their sub-answers)\n"
                    " - A search_history (list of past search queries and their summaries)\n\n"
                    "Note:\n"
                    " - The search_history represents queries YOU have already searched in previous turns.\n"
                    " - You MUST avoid generating queries that are identical to or semantically similar to any query in search_history.\n"
                    " - Only generate a new query if it provides new, distinct information that has not yet been searched.\n\n"
                    "Your task:\n"
                    "1. If any sub-question in thinking_trajectory has an answer 'unkown' or unclear,\n"
                    "   generate a specific, searchable query to find the missing information.\n"
                    "2. If you believe additional sub-questions are needed to answer the original question,\n"
                    "   generate a specific searchable query for that.\n"
                    "3. If no further search is needed, output <end>.\n\n"
                    "Output only one of the following two formats:\n"
                    "<search>Your query here</search>\n"
                    "<end>\n"
                    "Do not include any extra text, explanations, or other tags."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Original question: {question}\n\n"
                    f"Thinking trajectory:\n{trajectory_text}\n\n"
                    f"Search history:\n{history_text}"
                    "Output only one of the following two formats:\n"
                    "<search>Your query here</search>\n"
                    "<end>\n"
                    "Do not include any extra text, explanations, or other tags."
                )
            }
        ]

    @staticmethod
    def parse_response(response_text: str) -> Dict[str, Any]:
        """
        解析 LLM 输出为结构化 dict
        输出 dict 格式:
        {
            "action": "search" or "end",
            "query": 可搜索 query（当 action=search 时有）
            "is_legal_format": True/False  # 标记是否符合预期格式
        }
        新增规则：
        1. 当存在重复的 <search> 时，is_legal_format=False
        2. 当标签外有任何内容时，is_legal_format=False
        """
        def _safe_strip(x):
            return x.strip() if isinstance(x, str) else x

        # 默认认为格式合法
        is_legal_format = True

        # 兜底第一步: 空或类型异常 → 默认结束
        if not response_text or not isinstance(response_text, str):
            return {"action": "end", "query": None, "is_legal_format": False}

        # ===== 新增重复标签检查 =====
        search_tags = re.findall(r"<search>.*?</search>", response_text, re.DOTALL | re.IGNORECASE)
        if len(search_tags) > 1:
            is_legal_format = False

        # ===== 新增标签外多余内容检查 =====
        cleaned = re.sub(r"<search>.*?</search>", "", response_text, flags=re.DOTALL | re.IGNORECASE)
        # 对于 <end> 也算合法标签，不当作多余内容
        cleaned = re.sub(r"<end>", "", cleaned, flags=re.IGNORECASE)
        if cleaned.strip():
            # 标签之外确实有多余的非空内容
            is_legal_format = False

        # 正常匹配 <search>（取第一个匹配）
        search_match = re.search(r"<search>(.*?)</search>", response_text, re.DOTALL | re.IGNORECASE)
        if search_match:
            return {
                "action": "search",
                "query": _safe_strip(search_match.group(1)),
                "is_legal_format": is_legal_format
            }

        # 正常匹配 <end>
        end_match = re.search(r"<end>", response_text, re.IGNORECASE)
        if end_match:
            return {"action": "end", "query": None, "is_legal_format": is_legal_format}

        # === 智能兜底逻辑 ===
        is_legal_format = False  # 进入兜底解析逻辑时标记为 False

        # 如果模型没按格式输出，我们尝试自动提取一个搜索关键词
        cleaned_text = _safe_strip(response_text)
        # 截取前 200 字左右避免噪音
        snippet = cleaned_text.split("\n")[0][:200]
        # 简单去掉多余标点
        snippet = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fa5\s]", " ", snippet)
        snippet = re.sub(r"\s+", " ", snippet).strip()

        if snippet:
            # 兜底时给出一个搜索
            return {"action": "search", "query": snippet, "is_legal_format": False}

        # 如果提取不到有效片段 → 结束
        return {"action": "end", "query": None, "is_legal_format": False}


    def get_results(self, data: Dict[str, Any], temperature: Optional[float] = None) -> Dict[str, Any]:
        """
        单次输入 dict，返回结构化 action 输出
        """
        messages = self._build_messages(data)
        response_text = self.get_response(messages, temperature)
        return self.parse_response(response_text)

    async def async_get_results(self, data: Dict[str, Any], search_history: List[dict], temperature: Optional[float] = None) -> Dict[str, Any]:
        """
        单次输入 dict，返回结构化 action 输出
        """
        messages = self._build_messages(data, search_history)
        response_text = await self.async_get_response(messages, 'SearchAgent', temperature)
        return self.parse_response(response_text)

    async def get_results_parallel(
        self,
        dataset: List[Dict[str, Any]],
        temperature: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        并行处理多个输入 dict
        """
        batch_messages = [self._build_messages(d) for d in dataset]
        raw_results = await self.async_batch_get_response(batch_messages, 'SearchAgent', temperature)
        return [self.parse_response(r) for r in raw_results]

class RetrievalTool:
    """
    RetrievalTool 调用远程搜索引擎 API，
    输入 question 和 N，返回 N 个最相关的文档。
    支持多个 API URL 池，每次随机选择，并在失败时重试。
    """

    def __init__(self, api_urls: List[str]):
        """
        初始化 RetrievalTool
        
        Args:
            api_urls (List[str]): 搜索引擎 API 地址池
        """
        if not api_urls:
            raise ValueError("api_urls 列表不能为空")
        self.api_urls = api_urls

    def query(self, question: str, N: int, max_attempts: int = 5) -> List[Dict[str, Any]]:
        """
        查询相关文档（支持随机 URL 及重试策略）
        
        Args:
            question (str): 查询问题
            N (int): 返回文档数量
            max_attempts (int): 失败重试次数上限
        
        Returns:
            List[Dict[str, Any]]: 文档列表
        """
        headers = {'Content-Type': 'application/json'}
        payload = {
            'questions': [question],
            'N': N
        }

        attempts = 0
        tried_urls = set()

        while attempts < max_attempts:
            # 从未尝试过的 URL 中随机选一个
            available_urls = [url for url in self.api_urls if url not in tried_urls]
            if not available_urls:
                # 所有 URL 都尝试过，重新允许重复尝试
                available_urls = self.api_urls

            api_url = random.choice(available_urls)
            tried_urls.add(api_url)

            try:
                response = requests.post(api_url, headers=headers, data=json.dumps(payload), timeout=5)
                response.raise_for_status()  # HTTP 状态错误抛异常

                data = response.json()
                if not data or "top_k_docs" not in data[0]:
                    raise ValueError("Unexpected API response format")

                return data[0]["top_k_docs"]

            except (requests.exceptions.RequestException, ValueError) as e:
                attempts += 1
                print(f"⚠️ 第 {attempts} 次尝试失败（URL: {api_url}）：{e}")

        # 如果所有尝试都失败
        raise RuntimeError(f"❌ 请求失败次数超过 {max_attempts} 次，无法获取文档")


class SummaryAgent(BaseLLMClient):
    """
    SummaryAgent

    核心功能：
      - 输入 query，以及相关候选文档列表（list, 多个 doc）
      - 从这些文档中提取回答 query 的关键 evidence
      - 如果没有关键 evidence，则返回 <evidence>No useful information</evidence>

    输出严格限制为：
      <evidence>关键证据</evidence>
      或
      <evidence>No useful information</evidence>
    """

    def __init__(
        self,
        model_name: str = "gpt4o",
        system_jinja2_path: Optional[str] = None,
        user_jinja2_path: Optional[str] = None,
        default_temperature: float = 0.0
    ):
        super().__init__(
            model_name=model_name,
            system_jinja2_path=system_jinja2_path,
            user_jinja2_path=user_jinja2_path,
            default_temperature=default_temperature
        )

    def _build_messages(self, data: Dict[str, Any]) -> List[dict]:
        """
        data 示例:
        {
            "query": "用户问题",
            "docs": [
                "doc1 内容 ...",
                "doc2 内容 ..."
            ]
        }
        """
        query = data.get("query", "")
        docs = data.get("docs", [])

        docs_text_lines = []
        for idx, doc in enumerate(docs, start=1):
            docs_text_lines.append(f"Doc{idx}: {doc}")
        docs_text = "\n".join(docs_text_lines)

        return [
            {
                "role": "system",
                "content": (
                    "You are an evidence extraction assistant.\n"
                    "You will be given:\n"
                    " - A search query\n"
                    " - Several candidate documents related to that query\n\n"
                    "Your task:\n"
                    "1. From the provided documents, extract the key evidence that directly answers the query.\n"
                    "2. If no key evidence exists in the documents, output 'No useful information'.\n\n"
                    "Output exactly in one of the two formats:\n"
                    "<evidence>your key evidence here</evidence>\n"
                    "<evidence>No useful information</evidence>\n"
                    "Do not include explanations, reasoning steps, or any text outside the <evidence> tags."
                )
            },
            {
                "role": "user",
                "content": f"Query: {query}\nCandidate Documents:\n{docs_text}"
            }
        ]

    @staticmethod
    def parse_response(
        response_text: str,
        query: Optional[str] = None,
        docs: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        解析 LLM 输出为结构化 dict。
        
        返回:
        {
            "query": "原始查询",
            "evidence": "关键证据 或 No useful information",
            "is_legal_format": True/False  # 是否符合预期格式
        }

        兜底策略：
        - 如果有 <evidence> 标签，直接取标签内容
        - 如果没有 <evidence> 标签：
            * 尝试从响应文本首句提取核心片段
            * 如果无有效片段，则从 docs 中匹配 query 的关键词找最相关句子
            * 否则 No useful information

        新增规则：
        1. 重复 <evidence> 标签 → is_legal_format=False
        2. 标签之外有内容 → is_legal_format=False
        3. 解析出的 evidence 单词数超过 75 → is_legal_format=False
        """
        def _safe_strip(x):
            return x.strip() if isinstance(x, str) else x

        NOINFO = "No useful information"

        # 默认合法
        is_legal_format = True

        if not response_text or not isinstance(response_text, str):
            return {"query": query, "evidence": NOINFO, "is_legal_format": False}

        # === 规则1: 重复标签检查 ===
        evidence_tags = re.findall(r"<evidence>.*?</evidence>", response_text, re.DOTALL | re.IGNORECASE)
        if len(evidence_tags) > 1:
            is_legal_format = False

        # === 规则2: 标签外内容检查 ===
        cleaned = re.sub(r"<evidence>.*?</evidence>", "", response_text, flags=re.DOTALL | re.IGNORECASE)
        if cleaned.strip():
            is_legal_format = False

        # === 取标签内容 ===
        evidence_match = re.search(r"<evidence>(.*?)</evidence>", response_text, re.DOTALL | re.IGNORECASE)
        if evidence_match:
            evidence_content = _safe_strip(evidence_match.group(1))
            # === 规则3: 单词数超过75检查 ===
            word_count = len(evidence_content.split())
            if word_count > 75:
                is_legal_format = False
            return {
                "query": query,
                "evidence": evidence_content,
                "is_legal_format": is_legal_format
            }

        # === 兜底逻辑 ===
        is_legal_format = False

        # 第一兜底: 从首句提取
        cleaned_text = _safe_strip(response_text)
        snippet = cleaned_text.split("\n")[0][:200]
        snippet = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fa5\s]", " ", snippet)
        snippet = re.sub(r"\s+", " ", snippet).strip()

        if snippet:
            if any(kw.lower() in snippet.lower() for kw in ["no useful", "unknown", "none", "unavailable"]):
                return {"query": query, "evidence": NOINFO, "is_legal_format": False}
            return {"query": query, "evidence": snippet, "is_legal_format": False}

        # 第二兜底: docs 搜索
        if query and docs:
            query_terms = [t for t in re.split(r"\s+", query.strip()) if t]
            best_sentence = None
            best_score = 0
            for doc in docs:
                sentences = re.split(r"[。.!?]", doc)
                for sent in sentences:
                    score = sum(1 for term in query_terms if term.lower() in sent.lower())
                    if score > best_score:
                        best_score = score
                        best_sentence = sent.strip()
            if best_sentence and best_score > 0:
                return {"query": query, "evidence": best_sentence, "is_legal_format": False}

        return {"query": query, "evidence": NOINFO, "is_legal_format": False}

    def get_results(self, data: Dict[str, Any], temperature: Optional[float] = None) -> Dict[str, Any]:
        """
        单条测试
        """
        messages = self._build_messages(data)
        response_text = self.get_response(messages, temperature)
        return self.parse_response(
            response_text,
            query=data.get("query"),
            docs=data.get("docs")
        )

    async def async_get_results(self, data: Dict[str, Any], temperature: Optional[float] = None) -> Dict[str, Any]:
        """
        单条测试
        """
        messages = self._build_messages(data)
        response_text = await self.async_get_response(messages, 'SummaryAgent', temperature)
        return self.parse_response(
            response_text,
            query=data.get("query"),
            docs=data.get("docs")
        )

    async def get_results_parallel(
        self,
        dataset: List[Dict[str, Any]],
        temperature: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        批量处理多个输入
        """
        batch_messages = [self._build_messages(d) for d in dataset]
        raw_results = await self.async_batch_get_response(batch_messages, 'SummaryAgent', temperature)
        return [
            self.parse_response(r, query=d.get("query"), docs=d.get("docs"))
            for r, d in zip(raw_results, dataset)
        ]

class UpdateAgent(BaseLLMClient):
    """
    UpdateAgent

    核心功能：
    - 输入：
        1. 原始context_dict: { 'question': ..., 'thinking_trajectory': {...}, 'final_answer': ... }
        2. 新的信息 new_info: { 'query': ..., 'evidence': ... }
    - 输出：
        <Update>ti</Update>
        或 <Add>t(n+1)</Add>
    - 严格不输出其他内容
    - 额外本地函数负责根据模型输出更新原始context_dict
    """

    def __init__(
        self,
        model_name: str = "gpt4o",
        system_jinja2_path: Optional[str] = None,
        user_jinja2_path: Optional[str] = None,
        default_temperature: float = 0.0
    ):
        super().__init__(
            model_name=model_name,
            system_jinja2_path=system_jinja2_path,
            user_jinja2_path=user_jinja2_path,
            default_temperature=default_temperature
        )

    def _build_messages(self, context_dict: Dict[str, Any], new_info: Dict[str, Any]) -> List[dict]:
        """
        构造 prompt。
        context_dict 示例:
        {
            'question': 'Which city has the largest population, Beijing, Shanghai or Shenzhen?',
            'thinking_trajectory': {
                't1': {'question': 'What is the population of Beijing?', 'answer': '...'},
                't2': {'question': 'What is the population of Shanghai?', 'answer': '...'},
                't3': {'question': 'What is the population of Shenzhen?', 'answer': '...'},
            },
            'final_answer': 'Shanghai'
        }

        new_info 示例:
        {
            'query': 'Who was the first President of the United States?',
            'evidence': 'George Washington ...'
        }
        """
        # 格式化原来的思考轨迹
        trajectory_text_lines = []
        for k, qa in context_dict.get("thinking_trajectory", {}).items():
            trajectory_text_lines.append(f"{k}: Q: {qa.get('question', '')} | A: {qa.get('answer', '')}")
        trajectory_text = "\n".join(trajectory_text_lines)

        # 原始问题与最终答案
        orig_question = context_dict.get("question", "")
        final_answer = context_dict.get("final_answer", "")

        # 新的信息
        new_query = new_info.get("query", "")
        new_evidence = new_info.get("evidence", "")

        return [
            {
                "role": "system",
                "content": (
                    "You are an update detection assistant.\n"
                    "You are given:\n"
                    " - An original question and its thinking_trajectory (sub-questions and their answers)\n"
                    " - The final_answer derived from thinking_trajectory\n"
                    " - A new question and its evidence\n\n"
                    "Your task:\n"
                    "1. If the new question and evidence can replace one existing sub-question-answer pair in thinking_trajectory,\n"
                    "   output <Update>ti</Update> where ti is the key.\n"
                    "2. If the new question and evidence contains information not present in thinking_trajectory,\n"
                    "   output <Add>t(n+1)</Add> where n is current number of t's.\n"
                    "3. Output strictly one of these formats and no other text."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Original question: {orig_question}\n"
                    f"Final answer: {final_answer}\n"
                    f"Thinking trajectory:\n{trajectory_text}\n\n"
                    f"New query: {new_query}\n"
                    f"Evidence: {new_evidence}"
                )
            }
        ]

    @staticmethod
    def parse_response(
        response_text: str,
        context_dict: Dict[str, Any],
        new_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        解析 LLM 输出为结构化动作 dict。

        返回：
        {
            "action": "update" 或 "add",
            "target": "ti" 或 "t(n+1)",
            "updated_context": 更新后的context_dict,
            "is_legal_format": True/False
        }

        严格合法性检查：
        1. <Update>ti</Update> 必须且 ti 在轨迹中
        2. <Add>t(n+1)</Add> 必须且标签内仅为纯编号格式 t正整数（不允许前导零）
        3. 存在重复的 Update 或 Add → is_legal_format=False
        4. 标签外有多余内容 → is_legal_format=False

        额外规则：
        - 一旦 is_legal_format=False，无论原因，统一兜底为 action='add', target='t(n+1)'
        """

        import re

        def _safe_strip(x):
            """安全去除字符串首尾空格"""
            return x.strip() if isinstance(x, str) else x

        # 当前轨迹的 key 列表，例如 ["t1", "t2", ...]
        trajectory_keys = list(context_dict.get("thinking_trajectory", {}).keys())
        n = len(trajectory_keys)  # 已有轨迹数量

        # 初始化值
        action = None
        target = None
        is_legal_format = True

        # === 规则 3：检查重复标签 ===
        update_tags = re.findall(r"<Update>.*?</Update>", response_text, re.DOTALL | re.IGNORECASE)
        add_tags = re.findall(r"<Add>.*?</Add>", response_text, re.DOTALL | re.IGNORECASE)
        if len(update_tags) > 1 or len(add_tags) > 1:
            is_legal_format = False

        # === 规则 4：标签外多余内容检查 ===
        cleaned = re.sub(r"<Update>.*?</Update>", "", response_text, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r"<Add>.*?</Add>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
        if cleaned.strip():  # 有额外内容则非法
            is_legal_format = False

        # === 规则 1：匹配并校验 Update ===
        update_match = re.search(r"<Update>(.*?)</Update>", response_text, re.DOTALL | re.IGNORECASE)
        if update_match:
            cand_target = _safe_strip(update_match.group(1))
            if cand_target in trajectory_keys:
                action = "update"
                target = cand_target
            else:
                action = "update"
                target = cand_target
                is_legal_format = False

        # === 规则 2：严格匹配 Add（必须 t 正整数且不带前导零） ===
        add_match = re.search(r"<Add>(.*?)</Add>", response_text, re.DOTALL | re.IGNORECASE)
        if add_match:
            cand_target = _safe_strip(add_match.group(1))
            # 仅允许 t + 非零开头数字，例如 t1、t2、t10，不允许 t01、t02
            num_match = re.fullmatch(r"t([1-9]\d*)", cand_target)
            if num_match:
                num_val = int(num_match.group(1))
                if num_val == n + 1:
                    action = "add"
                    target = cand_target
                else:
                    # 序号错误
                    action = "add"
                    target = cand_target
                    is_legal_format = False
            else:
                # 格式不符合（不是纯 t正整数 或有前导零）
                action = "add"
                target = cand_target
                is_legal_format = False

        # === 特殊情况：没匹配到任何标签 ===
        if not action:
            action = "add"
            target = f"t{n+1}"
            is_legal_format = False

        # === 统一非法兜底逻辑 ===
        if not is_legal_format:
            action = "add"
            target = f"t{n+1}"

        # === 应用更新动作 ===
        updated_context = UpdateAgent.apply_update_action(context_dict, new_info, action, target)

        return {
            "action": action,
            "target": target,
            "updated_context": updated_context,
            "is_legal_format": is_legal_format
        }


    @staticmethod
    def apply_update_action(context_dict: Dict[str, Any], new_info: Dict[str, Any],
                            action: str, target: str) -> Dict[str, Any]:
        """
        根据动作更新 context_dict 的 thinking_trajectory。
        """
        # # 浅拷贝
        # updated_context = {k: v for k, v in context_dict.items()}
        # 深拷贝
        import copy
        updated_context = copy.deepcopy(context_dict)

        thinking_traj = dict(updated_context.get("thinking_trajectory", {}))

        if action == "update" and target in thinking_traj:
            thinking_traj[target] = {
                "question": new_info.get("query", ""),
                "answer": new_info.get("evidence", "")
            }
        elif action == "add":
            thinking_traj[target] = {
                "question": new_info.get("query", ""),
                "answer": new_info.get("evidence", "")
            }

        updated_context["thinking_trajectory"] = thinking_traj
        return updated_context

    def get_results(self, context_dict: Dict[str, Any], new_info: Dict[str, Any],
                    temperature: Optional[float] = None) -> Dict[str, Any]:
        """
        单次输入 （原上下文 + 新信息），返回更新动作以及新的 context_dict。
        """
        messages = self._build_messages(context_dict, new_info)
        response_text = self.get_response(messages, temperature)
        return self.parse_response(response_text, context_dict, new_info)

    async def async_get_results(self, context_dict: Dict[str, Any], new_info: Dict[str, Any],
                    temperature: Optional[float] = None) -> Dict[str, Any]:
        """
        单次输入 （原上下文 + 新信息），返回更新动作以及新的 context_dict。
        """
        messages = self._build_messages(context_dict, new_info)
        response_text = await self.async_get_response(messages, 'UpdateAgent', temperature)
        return self.parse_response(response_text, context_dict, new_info)

    async def get_results_model_nameparallel(self,
                                   dataset: List[Dict[str, Any]],
                                   temperature: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        并行处理多个输入，dataset 内每个元素是:
        {
            "context_dict": {...},
            "new_info": {...}
        }
        """
        batch_messages = [
            self._build_messages(d["context_dict"], d["new_info"])
            for d in dataset
        ]
        raw_results = await self.async_batch_get_response(batch_messages, 'UpdateAgent', temperature)
        return [
            self.parse_response(r, d["context_dict"], d["new_info"])
            for r, d in zip(raw_results, dataset)
        ]

class AnswerAgent(BaseLLMClient):
    """
    AnswerAgent

    核心功能：
      - 输入 question 和 thinking_trajectory
      - 根据轨迹生成最终 final_answer
      - 输出格式：<final_answer>...</final_answer>
      - 返回的结果 dict 必须包含 question + thinking_trajectory + final_answer
    """

    def __init__(
        self,
        model_name: str = "gpt4o",
        system_jinja2_path: Optional[str] = None,
        user_jinja2_path: Optional[str] = None,
        default_temperature: float = 0.0
    ):
        super().__init__(
            model_name=model_name,
            system_jinja2_path=system_jinja2_path,
            user_jinja2_path=user_jinja2_path,
            default_temperature=default_temperature
        )

    def _build_messages(self, data: Dict[str, Any]) -> List[dict]:
        """
        构造 LLM Prompt
        """
        question = data.get("question", "")
        trajectory = data.get("thinking_trajectory", {})

        trajectory_text_lines = [
            f"{k}: Q: {qa.get('question', '')} | A: {qa.get('answer', '')}"
            for k, qa in trajectory.items()
        ]
        trajectory_text = "\n".join(trajectory_text_lines)

        return [
            {
                "role": "system",
                "content": (
                    "You are a final answer generation assistant.\n"
                    "You are given:\n"
                    " - An original question\n"
                    " - A thinking_trajectory (list of sub-questions and their answers)\n\n"
                    "Your task:\n"
                    "1. Use the provided thinking_trajectory's answers to determine the final answer to the original question.\n"
                    "2. You must only output in the following strict format as shown in EXAMPLE_PROMPT:\n"
                    f"{self.EXAMPLE_PROMPT}\n"
                    "IMPORTANT: Output must be:\n"
                    "<final_answer>your answer here</final_answer>\n"
                    "No extra text, no reasoning."
                )
            },
            {
                "role": "user",
                "content": f"Question: {question}\nThinking trajectory:\n{trajectory_text}"
            }
        ]

    @staticmethod
    def parse_response(response_text: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        从模型输出中提取 final_answer
        返回完整 dict：包含 question + thinking_trajectory + final_answer + is_legal_format
        带兜底策略：
        1. 优先取 <final_answer> tag 内容
        2. 否则取首行
        3. 若为空，尝试根据思考轨迹中的信息自动推断

        新增规则：
        - 当存在重复的 <final_answer> 时，is_legal_format = False
        - 当标签外有额外内容时，is_legal_format = False
        """
        def _safe_strip(x):
            return x.strip() if isinstance(x, str) else x

        final_answer_val = ""
        is_legal_format = True  # 默认合法

        # ---------------- 新增部分（规则 1 & 2 检查） ----------------
        if response_text and isinstance(response_text, str):
            tags = re.findall(r"<final_answer>.*?</final_answer>", response_text, re.DOTALL | re.IGNORECASE)
            if len(tags) > 1:  # 重复标签
                is_legal_format = False

            # 检查标签外是否有内容
            cleaned = re.sub(r"<final_answer>.*?</final_answer>", "", response_text, flags=re.DOTALL | re.IGNORECASE)
            if cleaned.strip():
                is_legal_format = False
        # -------------------------------------------------------------

        # 1) 优先匹配标签（取第一个标签内容）
        if response_text and isinstance(response_text, str):
            match = re.search(r"<final_answer>(.*?)</final_answer>", response_text, re.DOTALL | re.IGNORECASE)
            if match:
                final_answer_val = _safe_strip(match.group(1))
                # ⚠ 注意：不要在这里强制 is_legal_format=True，因为上面规则可能已经令其为 False
            else:
                # 没标签 → 尝试首行
                if not final_answer_val:
                    is_legal_format = False
                    first_line = _safe_strip(response_text.split("\n")[0])
                    if first_line:
                        final_answer_val = first_line
        else:
            is_legal_format = False

        # 3) 如果还是空 → 根据轨迹自动推断
        if not final_answer_val:
            is_legal_format = False  # 推断也不属于严格格式
            trajectory = data.get("thinking_trajectory", {})
            max_val = -float("inf")
            best_entity = ""
            for qa in trajectory.values():
                ans_text = qa.get("answer", "")
                numbers = re.findall(r"[\d\.]+", ans_text.replace(",", ""))
                if numbers:
                    try:
                        num_val = float(numbers[0])
                        if num_val > max_val:
                            max_val = num_val
                            best_entity = re.sub(
                                r"^What is the population of\s+",
                                "",
                                qa.get("question", ""),
                                flags=re.IGNORECASE
                            )
                    except ValueError:
                        pass
            if best_entity:
                final_answer_val = best_entity

            # 无法数字推断 → fallback 首条答案
            if not final_answer_val and trajectory:
                first_key = list(trajectory.keys())[0]
                final_answer_val = trajectory[first_key].get("answer", "")

        return {
            "question": data.get("question", ""),
            "thinking_trajectory": data.get("thinking_trajectory", {}),
            "final_answer": final_answer_val,
            "is_legal_format": is_legal_format
        }


    def get_results(self, data: Dict[str, Any], temperature: Optional[float] = None) -> Dict[str, Any]:
        """
        单条生成最终答案
        """
        messages = self._build_messages(data)
        response_text = self.get_response(messages, temperature)
        return self.parse_response(response_text, data)

    async def async_get_results(self, data: Dict[str, Any], temperature: Optional[float] = None) -> Dict[str, Any]:
        """
        单条生成最终答案
        """
        messages = self._build_messages(data)
        response_text = await self.async_get_response(messages, 'AnswerAgent', temperature)
        return self.parse_response(response_text, data)

    async def get_results_parallel(self, dataset: List[Dict[str, Any]], temperature: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        批量生成最终答案，每个返回值也是完整的 dict
        """
        batch_messages = [self._build_messages(d) for d in dataset]
        raw_results = await self.async_batch_get_response(batch_messages, 'AnswerAgent', temperature)
        return [
            self.parse_response(r, d)
            for r, d in zip(raw_results, dataset)
        ]

class DirectAnswer(BaseLLMClient):
    """
    DirectAnswer

    核心功能：
      - 输入 query（原始问题）
      - 直接调用 LLM 输出最终答案（严格使用 <final_answer>...</final_answer> 格式）
      - 风格：按照 self.EXAMPLE_PROMPT
      - 后处理函数带兜底策略
      - 输出 dict：包含 query + answer
    """

    def __init__(
        self,
        model_name: str = "gpt4o",
        system_jinja2_path: Optional[str] = None,
        user_jinja2_path: Optional[str] = None,
        default_temperature: float = 0.0
    ):
        super().__init__(
            model_name=model_name,
            system_jinja2_path=system_jinja2_path,
            user_jinja2_path=user_jinja2_path,
            default_temperature=default_temperature
        )

    def _build_messages(self, data: Dict[str, Any]) -> List[dict]:
        """
        构造 LLM Prompt
        """
        query = data.get("query", "")

        return [
            {
                "role": "system",
                "content": (
                    "You are a direct answer generation assistant.\n"
                    "You are given:\n"
                    " - A query (question)\n\n"
                    "Your task:\n"
                    "1. Produce the final answer directly without reasoning.\n"
                    "2. Only output in the strict format shown in EXAMPLE_PROMPT:\n"
                    f"{self.EXAMPLE_PROMPT}\n"
                    "IMPORTANT: Output must be:\n"
                    "<final_answer>your answer here</final_answer>\n"
                    "No extra text, no reasoning."
                )
            },
            {
                "role": "user",
                "content": f"Query: {query}"
            }
        ]

    @staticmethod
    def parse_response(response_text: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        从模型输出中提取 answer
        带兜底策略：
          1. 优先取 <final_answer> tag 内容
          2. 否则取首行
          3. 若为空，尝试使用简易推断或原query
        """
        def _safe_strip(x):
            return x.strip() if isinstance(x, str) else x

        answer_val = ""

        if response_text and isinstance(response_text, str):
            # 1) 标签匹配
            match = re.search(r"<final_answer>(.*?)</final_answer>", response_text, re.DOTALL | re.IGNORECASE)
            if match:
                answer_val = _safe_strip(match.group(1))
            else:
                # 2) 首行兜底
                first_line = _safe_strip(response_text.split("\n")[0])
                if first_line:
                    answer_val = first_line

        # 3) 空值兜底
        if not answer_val:
            query = data.get("query", "")
            # 简单推断：如果 query 是问数量、人口等，直接返回 "Unknown"
            if re.search(r"\b(population|number of|count|amount)\b", query, re.IGNORECASE):
                answer_val = "Unknown"
            else:
                answer_val = "No Answer"

        return {
            "query": data.get("query", ""),
            "answer": answer_val
        }

    def get_results(self, data: Dict[str, Any], temperature: Optional[float] = None) -> Dict[str, Any]:
        """
        单条生成最终答案
        """
        messages = self._build_messages(data)
        response_text = self.get_response(messages, temperature)
        return self.parse_response(response_text, data)

    async def async_get_results(self, data: Dict[str, Any], temperature: Optional[float] = None) -> Dict[str, Any]:
        """
        单条生成最终答案（异步）
        """
        messages = self._build_messages(data)
        response_text = await self.async_get_response(messages, 'DirectAnswer', temperature)
        return self.parse_response(response_text, data)

    async def get_results_parallel(self, dataset: List[Dict[str, Any]], temperature: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        批量生成最终答案，每个返回值也是完整的 dict
        """
        batch_messages = [self._build_messages(d) for d in dataset]
        raw_results = await self.async_batch_get_response(batch_messages, 'DirectAnswer', temperature)
        return [
            self.parse_response(r, d)
            for r, d in zip(raw_results, dataset)
        ]

class RAGAnswer(BaseLLMClient):
    """
    RAGAnswer

    核心功能：
      - 输入 query（原始问题）和 docs（相关文档列表）
      - 调用 LLM，根据 docs 回答 query
      - 严格使用 <final_answer>...</final_answer> 格式
      - 风格和后处理逻辑与 DirectAnswer 相同
    """

    def __init__(
        self,
        model_name: str = "gpt4o",
        system_jinja2_path: Optional[str] = None,
        user_jinja2_path: Optional[str] = None,
        default_temperature: float = 0.0
    ):
        super().__init__(
            model_name=model_name,
            system_jinja2_path=system_jinja2_path,
            user_jinja2_path=user_jinja2_path,
            default_temperature=default_temperature
        )

    def _build_messages(self, data: Dict[str, Any]) -> List[dict]:
        """
        构造 LLM Prompt，包含 docs 作为参考
        """
        query = data.get("query", "")
        docs = data.get("docs", [])

        # 将 docs 转成字符串拼接
        docs_str = "\n\n".join(
            f"Document {i+1}:\n{doc}" for i, doc in enumerate(docs)
        )

        return [
            {
                "role": "system",
                "content": (
                    "You are a retrieval-augmented generation (RAG) assistant.\n"
                    "You are given:\n"
                    " - A query (question)\n"
                    " - Several relevant documents (context)\n\n"
                    "Your task:\n"
                    "1. Use ONLY the information from the given documents to answer the query.\n"
                    "2. If the answer cannot be found in the provided docs, say 'Unknown'.\n"
                    "3. Only output in the strict format shown in EXAMPLE_PROMPT:\n"
                    f"{self.EXAMPLE_PROMPT}\n"
                    "IMPORTANT: Output must be:\n"
                    "<final_answer>your answer here</final_answer>\n"
                    "No extra text, no reasoning."
                )
            },
            {
                "role": "user",
                "content": f"Query: {query}\n\nRelevant Documents:\n{docs_str}"
            }
        ]

    @staticmethod
    def parse_response(response_text: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        从模型输出中提取 answer
        带兜底策略：
          1. 优先取 <final_answer> tag 内容
          2. 否则取首行
          3. 若为空，返回 Unknown 或 No Answer
        """
        def _safe_strip(x):
            return x.strip() if isinstance(x, str) else x

        answer_val = ""

        if response_text and isinstance(response_text, str):
            # 1) 标签匹配
            match = re.search(r"<final_answer>(.*?)</final_answer>", response_text, re.DOTALL | re.IGNORECASE)
            if match:
                answer_val = _safe_strip(match.group(1))
            else:
                # 2) 首行兜底
                first_line = _safe_strip(response_text.split("\n")[0])
                if first_line:
                    answer_val = first_line

        # 3) 空值兜底
        if not answer_val:
            query = data.get("query", "")
            if re.search(r"\b(population|number of|count|amount)\b", query, re.IGNORECASE):
                answer_val = "Unknown"
            else:
                answer_val = "No Answer"

        return {
            "query": data.get("query", ""),
            "answer": answer_val
        }

    def get_results(self, data: Dict[str, Any], temperature: Optional[float] = None) -> Dict[str, Any]:
        """
        单条生成最终答案
        """
        messages = self._build_messages(data)
        response_text = self.get_response(messages, temperature)
        return self.parse_response(response_text, data)

    async def async_get_results(self, data: Dict[str, Any], temperature: Optional[float] = None) -> Dict[str, Any]:
        """
        单条生成最终答案（异步）
        """
        messages = self._build_messages(data)
        response_text = await self.async_get_response(messages, 'RAGAnswer', temperature)
        return self.parse_response(response_text, data)

    async def get_results_parallel(self, dataset: List[Dict[str, Any]], temperature: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        批量生成最终答案，每个返回值也是完整的 dict
        """
        batch_messages = [self._build_messages(d) for d in dataset]
        raw_results = await self.async_batch_get_response(batch_messages, 'RAGAnswer', temperature)
        return [
            self.parse_response(r, d)
            for r, d in zip(raw_results, dataset)
        ]


if __name__ == "__main__":

    # 初始化各 Agent
    planning_agent = PlanningAgent(model_name="gpt4o")
    search_agent = SearchAgent(model_name="gpt4o")
    retrieval_tool = RetrievalTool(api_url="http://localhost:8000/search")
    summary_agent = SummaryAgent(model_name="gpt4o")
    update_agent = UpdateAgent(model_name="gpt4o")
    answer_agent = AnswerAgent(model_name="gpt4o")

    # 实例化 workflow
    qa_workflow = QAWorkflow(
        planning_agent,
        search_agent,
        retrieval_tool,
        summary_agent,
        update_agent,
        answer_agent,
        max_loops=5,
        top_k_docs=5
    )

    # 测试 query
    test_query = "The battle in which Giuseppe Arimondi lost his life secured what for Ethiopia?"
    final_answer = qa_workflow.run(test_query)
    print(f"\n最终输出: {final_answer}")
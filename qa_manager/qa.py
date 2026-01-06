import numpy as np
from typing import Dict, List, Any
import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask
from verl import DataProto

# from qa_manager.base_agent import PlanningAgent, SearchAgent, RetrievalTool, SummaryAgent, UpdateAgent, AnswerAgent
from qa_manager.base_agent import (
    PlanningAgent,
    SearchAgent,
    RetrievalTool,
    SummaryAgent,
    UpdateAgent,
    AnswerAgent,
)

def remove_trailing_marker(text):
    # Check if the text ends with the marker and remove it
    marker = "<|im_end|>"
    if text.endswith(marker):
        return text[:-len(marker)]
    return text

class Agentic_RAG_Manager:
    def __init__(self, tokenizer, config):
        self.tokenizer = tokenizer

        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.return_raw_chat = config.get("return_raw_chat", True)
        self.return_full_prompt = config.get("return_full_prompt", False)
        self.truncation = config.get("truncation", "right")

        # agent name
        self.planning_agent = PlanningAgent()
        self.search_agent = SearchAgent()
        ips = ['10.148.30.244']  # IP for retrieval model
        ips_ports = []
        for ip in ips:
            temp_list = ["http://{}:800{}/search".format(ip, i) for i in range(8)]
            ips_ports.extend(temp_list)
        self.retrieval_tool = RetrievalTool(api_urls=ips_ports)
        self.summary_agent = SummaryAgent()
        self.update_agent = UpdateAgent()
        self.answer_agent = AnswerAgent()

        # Configure the parameter rules for each agent_name
        self.AGENT_PARAM_RULES = {
            "PlanningAgent": [(str,)],                       # 1 str
            "SearchAgent": [(dict,)],                         # 1 dict
            "RetrievalTool": [(str, int)],                    # str + int
            "SummaryAgent": [(str,)],                         # 1 str
            "UpdateAgent": [(str,)],                          # 1 str
            "AnswerAgent": [(dict,)],                          # 1 dict
        }
    
    def trans_rawprompt_to_ids(self, agent_name: str, *args):
        """
        Execute with different parameter combinations based on the value of agent_name.
        agent_name must be one of the following: 
        PlanningAgent, SearchAgent, RetrievalTool, SummaryAgent, UpdateAgent, AnswerAgent
        """
        if agent_name == "PlanningAgent":
            # receive query: str
            if len(args) == 1 and isinstance(args[0], str):
                # print(f"[{agent_name}] query = {args[0]}")
                query = args[0]
                messages = self.planning_agent._build_messages(query)
            else:
                raise ValueError("The PlanningAgent requires a string named "query".")

        elif agent_name == "SearchAgent":
            # receive memory: dict
            if len(args) == 2 and isinstance(args[0], dict) and isinstance(args[1], list):
                # print(f"[{agent_name}] memory = {args[0]}")
                memory, search_history = args[0], args[1]
                messages = self.search_agent._build_messages(memory, search_history)
            else:
                raise ValueError("SearchAgent requires a dict memory")

        elif agent_name == "RetrievalTool":
            # receive query: str + top_k: int
            if len(args) == 2 and isinstance(args[0], str) and isinstance(args[1], int):
                # print(f"[{agent_name}] query = {args[0]}, top_k = {args[1]}")
                query, top_k = args[0], args[1]
            else:
                raise ValueError("RetrievalTool need (query: str, top_k: int)")

        elif agent_name == "SummaryAgent":
            # receive summary_input: str
            if len(args) == 1 and isinstance(args[0], dict):
                # print(f"[{agent_name}] summary_input = {args[0]}")
                summary_input = args[0]
                messages = self.summary_agent._build_messages(summary_input)
            else:
                raise ValueError("The SummaryAgent requires a string named summary_input")

        elif agent_name == "UpdateAgent":
            # receive summary_output: str
            if len(args) == 2 and isinstance(args[0], dict) and isinstance(args[1], dict):
                # print(f"[{agent_name}] summary_output = {args[0]}")
                summary_output = args[0]
                context_dict, new_info = args[0], args[1]
                messages = self.update_agent._build_messages(context_dict, new_info)
            else:
                raise ValueError("UpdateAgent requires a string named summary_output")

        elif agent_name == "AnswerAgent":
            # receive memory: dict
            if len(args) == 1 and isinstance(args[0], dict):
                # print(f"[{agent_name}] memory = {args[0]}")
                memory = args[0]
                messages = self.answer_agent._build_messages(memory)
            else:
                raise ValueError("AnswerAgent need a dict memory")

        else:
            raise ValueError(f"unknown agent_name: {agent_name}")

        single_dict = self.get_single_ids(messages)

        return single_dict
        
    
    def get_single_ids(self, messages):

        update_dict = {}

        raw_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        position_ids = compute_position_id_with_mask(attention_mask)

        update_dict["input_ids"] = input_ids[0]
        update_dict["attention_mask"] = attention_mask[0]
        update_dict["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        update_dict["raw_prompt_ids"] = raw_prompt_ids
        # encode prompts without chat template
        if self.return_raw_chat:
            update_dict["raw_prompt"] = messages

        return update_dict

    def get_answers_text(self, data: DataProto):
        
        predicted_answers_list = []

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            # valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            # valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            # sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences = valid_response_ids
            sequences_str = self.tokenizer.decode(sequences)
            sequences_str = remove_trailing_marker(sequences_str)
            predicted_answers_list.append(sequences_str)

        return predicted_answers_list
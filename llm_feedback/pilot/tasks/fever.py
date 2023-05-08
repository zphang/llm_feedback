from typing import List, Dict, Optional

import datasets
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    SystemMessage
)

from ...utils.io import read_json
from ...utils.contriever import Contriever
from .base import BaseTask

MAX_NUM_QUERIES = 3


class FEVERTask(BaseTask):
    """FEVER task"""

    def __init__(self, task_args_str):
        fever_config = read_json(task_args_str)
        self.contriever = Contriever.setup(
            passage_path=fever_config["passage_path"],
            index_path=fever_config["index_path"],
        )

    def get_dataset(self, phase: str):
        return datasets.load_dataset("fever", "v1.0")[phase]

    def get_chain(self, generation_llm: str, feedback_llm: str, refinement_llm: str,
                  chain_name: Optional[str] = None):
        # 0. Setup
        assert chain_name is None
        initial_llm = ChatOpenAI(model_name=generation_llm)
        feedback_llm = ChatOpenAI(model_name=feedback_llm)
        refinement_llm = ChatOpenAI(model_name=refinement_llm)

        # === 1a. Initial search === #
        initial_search_terms_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are a fact-checking assistant."),
            HumanMessagePromptTemplate.from_template("""
The following is a claim that we want to verify if it is true, false, or there is insufficient evidence to make a conclusion.

Claim: {claim}

To help verify this claim, output a list of up to 3 search queries that we want to search Google or Wikipedia for, in the following format:
<search>...</search>
<search>...</search>
<search>...</search>
            """.strip(), input_variables=["claim"])
        ])
        initial_search_terms_chain = LLMChain(llm=initial_llm, prompt=initial_search_terms_prompt, output_key="initial_search_terms")

        # === 1b. Initial answer === #
        initial_answer_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are a fact-checking assistant."),
            HumanMessagePromptTemplate.from_template("""
The following is a claim that we want to verify if it is true, false, or there is insufficient evidence to make a conclusion.

Claim: {claim}

To help verify this claim, we ran a quick Google/Wikipedia search and obtained the following excerpts:

{formatted_search_result}

Based on the search results, determine whether the evident supports, refutes, or is insufficient to make a conclusion on the above claim. Output one of the following exactly:
<answer>SUPPORTS</answer>
<answer>REFUTES</answer>
<answer>NOT ENOUGH INFO</answer>
            """.strip(), input_variables=["claim", "formatted_search_result"])
        ])
        initial_answer_chain = LLMChain(llm=initial_llm, prompt=initial_answer_prompt, output_key="initial_answer")

        # === 2. Feedback === #
        ilf_feedback_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are a fact-checking assistant."),
            HumanMessagePromptTemplate.from_template("""
The following is a claim that we want to verify if it is true, false, or there is insufficient evidence to make a conclusion.

Claim: {claim}

To help verify this claim, a student ran a quick Google/Wikipedia search and obtained the following excerpts:

{formatted_search_result}

The student then read the above search results and provided the following answer:
"{initial_answer}"

How would you improve the above search and answer? Please provide feedback on both the choice of search terms as well as the final answers.
            """.strip(), input_variables=["claim", "formatted_search_result", "initial_answer"])
        ])
        feedback_chain = LLMChain(llm=feedback_llm, prompt=ilf_feedback_prompt, output_key="feedback")

        # === 3a. Refinement Search === #
        refinement_search_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are a fact-checking assistant."),
            HumanMessagePromptTemplate.from_template("""
The following is a claim that we want to verify if it is true, false, or there is insufficient evidence to make a conclusion.

Claim: {claim}

A previous student performed a search on the following search terms:
{formatted_search_terms}

Then they provided the following answer:
"{initial_answer}"

A teacher then provided the following feedback:
"{feedback}"

Based on the above, output a list of up to 3 search queries that we want to search Google or Wikipedia for, in the following format:
<search>...</search>
<search>...</search>
<search>...</search>
            """.strip(), input_variables=["claim", "formatted_search_terms", "initial_answer", "feedback"])
        ])
        refinement_search_chain = LLMChain(llm=refinement_llm, prompt=refinement_search_prompt,
                                           output_key="refinement_search_terms")

        # === 3b. Refinement answer === #
        refinement_answer_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are a fact-checking assistant."),
            HumanMessagePromptTemplate.from_template("""
The following is a claim that we want to verify if it is true, false, or there is insufficient evidence to make a conclusion.

Claim: {claim}

A previous student performed a search on the following search terms:
{formatted_search_terms}

Then they provided the following answer:
"{initial_answer}"

A teacher then provided the following feedback:
"{feedback}"

We took the above into account and ran a Google/Wikipedia search and obtained the following excerpts:

{formatted_refinement_search_result}

Based on the search results, determine whether the evident supports, refutes, or is insufficient to make a conclusion on the above claim. Output one of the following exactly:
<answer>SUPPORTS</answer>
<answer>REFUTES</answer>
<answer>NOT ENOUGH INFO</answer>
            """.strip(), input_variables=["claim", "formatted_search_terms", "initial_answer", "feedback",
                                          "formatted_refinement_search_result"])
        ])
        refinement_answer_chain = LLMChain(llm=refinement_llm, prompt=refinement_answer_prompt,
                                           output_key="refinement_answer")
        return {
            "initial_search_chain": initial_search_terms_chain,
            "initial_answer_chain": initial_answer_chain,
            "feedback_chain": feedback_chain,
            "refinement_search_chain": refinement_search_chain,
            "refinement_answer_chain": refinement_answer_chain,
        }

    def process(self, chain, example):
        out1 = chain["initial_search_chain"]({"claim": example["claim"]})

        search_terms = parse_search_terms(out1["initial_search_terms"])[:MAX_NUM_QUERIES]
        search_result = self.contriever.get_multi_passages(search_terms, top_k=2)
        formatted_search_result = format_search_results(search_result)

        out2 = chain["initial_answer_chain"]({
            "claim": example["claim"],
            "formatted_search_result": formatted_search_result,
        })
        out3 = chain["feedback_chain"]({
            "claim": example["claim"],
            "formatted_search_result": formatted_search_result,
            "initial_answer": out2["initial_answer"],
        })
        formatted_search_terms = "\n".join(f"- {search_term}" for search_term in search_terms)
        out4 = chain["refinement_search_chain"]({
            "claim": example["claim"],
            "formatted_search_terms": formatted_search_terms,
            "initial_answer": out2["initial_answer"],
            "feedback": out3["feedback"],
        })

        refinement_search_terms = parse_search_terms(out4["refinement_search_terms"])[:MAX_NUM_QUERIES]
        refinement_search_result = self.contriever.get_multi_passages(refinement_search_terms, top_k=2)
        formatted_refinement_search_result = format_search_results(refinement_search_result)
        out5 = chain["refinement_answer_chain"]({
            "claim": example["claim"],
            "formatted_search_terms": formatted_search_terms,
            "initial_answer": out2["initial_answer"],
            "feedback": out3["feedback"],
            "formatted_refinement_search_result": formatted_refinement_search_result,
        })
        return {
            "claim": example["claim"],
            "initial_search_terms": out1["initial_search_terms"],
            "search_results": search_result,
            "initial_answer": out2["initial_answer"],
            "feedback": out3["feedback"],
            "refinement_search_terms": out4["refinement_search_terms"],
            "refinement_search_result": refinement_search_result,
            "refinement_answer": out5["refinement_answer"],
        }

    def evaluate(self, phase: str, outputs: List[Dict]):
        dataset = self.get_dataset(phase=phase)
        scores = {"initial_score": [], "refined_score": []}
        for row, example in zip(outputs, dataset):
            initial_solution = get_fever_answer(row["initial_answer"])
            refined_solution = get_fever_answer(row["refinement_answer"])
            scores["initial_score"].append(example["label"] == initial_solution)
            scores["refined_score"].append(example["label"] == refined_solution)
        return {
            "initial_score": float(pd.Series(scores["initial_score"]).mean()),
            "refined_score": float(pd.Series(scores["refined_score"]).mean()),
        }


def get_fever_answer(string):
    if "<answer>SUPPORTS</answer>" in string:
        return "SUPPORTS"
    elif "<answer>REFUTES</answer>" in string:
        return "REFUTES"
    elif "<answer>NOT ENOUGH INFO</answer>" in string:
        return "NOT ENOUGH INFO"
    else:
        return "other"


def parse_search_terms(string):
    out_list = []
    parts = string.split("<search>")
    for part in parts:
        if not part.strip():
            continue
        out_list.append(part.split("</search>")[0])
    return out_list


def format_search_results(search_result):
    lines = []
    for i, (query, passages) in enumerate(search_result.items()):
        lines.append(f"Search {i+1}: \"{query}\"")
        for passage in passages:
            lines.append(f"    Article: {passage['title']}")
            lines.append(f"    Excerpt: {passage['text']}")
            lines.append("")
        lines.append("")
        lines.append("===")
        lines.append("")
    return "\n".join(lines)

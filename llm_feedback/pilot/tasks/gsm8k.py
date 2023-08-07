from typing import List, Dict, Optional
import pandas as pd
import os
import openai
from datasets import load_dataset
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    SystemMessage
)

from .base import BaseTask

import logging
import re

class GSM8KTask(BaseTask):
    """GSM8K task"""

    def get_dataset(self, phase: str):
        ds = load_dataset("gsm8k", "main")[phase]
        return ds

    def get_chain(self, generation_llm: str, feedback_llm: str, refinement_llm: str,
                  chain_name: Optional[str] = None):
        # 0. Setup
        assert chain_name is None
        initial_llm = self.get_llm(model_name=generation_llm)
        feedback_llm = self.get_llm(model_name=feedback_llm)
        refinement_llm = self.get_llm(model_name=refinement_llm)

        # === 1. Initial solution === #
        initial_solution_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are a math question-answering assistant."),
            HumanMessagePromptTemplate.from_template("""
        The following is a math problem. Reason through the problem step-by-step, putting each separate reasoning step on a new numbered line (e.g. "Step 1. ") and finally respond with the right answer. Put the final answer on a single line with the format 'The answer is (answer)'. The answer should be a number without units.

        Question:
        {text}
            """.strip(), input_variables=["text"])
        ])
        initial_solution_chain = LLMChain(llm=initial_llm, prompt=initial_solution_prompt, output_key="initial_solution")

        # === 2. Feedback === #
        ilf_feedback_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are a math question-answering assistant."),
            HumanMessagePromptTemplate.from_template("""
        The following is a proposed solution to a math question. There may be an error with the solution, or it may be correct. Go through each line and indicate if that line has an error (and explain what the error is) or no error ("OK."). After that, print "REFINE" one a single line if there are errors identified, or if there are no errors, print "CORRECT".

        The output should look like:

            Step X: (Description of error)

            or 

            Step X: OK.

        for each line.

        Question:
        {text}

        Proposed solution:
        {initial_solution}
            """.strip(), input_variables=["text", "initial_solution"])
        ])
        feedback_chain = LLMChain(llm=feedback_llm, prompt=ilf_feedback_prompt, output_key="feedback")

        
        # === 3. Refinement === #
        ilf_refinement_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are a math question-answering assistant."),
            HumanMessagePromptTemplate.from_template("""
        You will be given a math problem, and a proposed answer from a student. You will also be provided feedback a teacher provided on that initial solution. Based on the feedback, reason through the problem step-by-step, and finally respond with the right answer. Put the final answer on a single line with the format 'The answer is (answer)'. The answer should be a number without units.

        Instruction:
        {text}
        Student's answer:
        {initial_solution}
        Teacher's feedback:
        {feedback}
            """.strip(), input_variables=["text", "initial_solution", "feedback"])
        ])
        refinement_chain = LLMChain(llm=refinement_llm, prompt=ilf_refinement_prompt, output_key="refinement")

        ilf_chain = SequentialChain(
            chains=[initial_solution_chain, feedback_chain, refinement_chain],
            input_variables=["text"],
            output_variables=["initial_solution", "feedback", "refinement"],
        )
        return ilf_chain

    def process(self, chain, example, q_key="question", a_key="answer"):
        return chain({"text": example[q_key], "answer": example[a_key]})

    def evaluate(self, phase: str, outputs: List[Dict]):
        metric_dict = {}
        dataset = self.get_dataset(phase=phase)
        scores = {"initial_score": [], "refined_score": [], "initial_results" : [],  "refined_results": [], "answers": []}
        for row, example in zip(outputs, dataset):
            initial_solution = get_gsm8k_answer(row["initial_solution"])
            refined_solution = get_gsm8k_answer(row["refinement"])
            parsed_answer = get_gsm8k_dataset_answer(example["answer"])

            initial_solution = int(initial_solution) if initial_solution.isdigit() else initial_solution
            refined_solution = int(refined_solution) if refined_solution.isdigit() else refined_solution
            parsed_answer = int(parsed_answer) if parsed_answer.isdigit() else parsed_answer
            if "X" in str(parsed_answer) or "Y" in str(initial_solution) or "Y" in str(refined_solution):
                # print(row["answer"], row["initial_solution"], row["refinement"], sep="\n")
                print(example["question"], example["answer"], sep="\n")
                print(parsed_answer, initial_solution, refined_solution)
                print("="*20)
            scores["initial_score"].append(parsed_answer == initial_solution)
            scores["refined_score"].append(parsed_answer == refined_solution)
            scores["initial_results"].append(initial_solution)
            scores["refined_results"].append(refined_solution)
            scores["answers"].append(parsed_answer)
        metric_dict["initial_results"] = scores["initial_results"]
        metric_dict["refined_results"] = scores["refined_results"]
        metric_dict["answers"] = scores["answers"]
        print("initial_results", scores["initial_results"])
        print("refined_results", scores["refined_results"])
        # output different index
        diff = [i for i in range(len(scores["initial_results"])) if scores["initial_results"][i] != scores["refined_results"][i]]
        print("diff", diff)
        print("answers", scores["answers"])
        metric_dict["initial_score"] = float(pd.Series(scores["initial_score"]).mean())
        metric_dict["refined_score"] = float(pd.Series(scores["refined_score"]).mean())
        return metric_dict

def get_gsm8k_dataset_answer(solution):
    #### \d+
    regex = re.compile(r"(#### \d+)")
    matches = regex.findall(solution)
    if len(matches) == 0:
        return "X"
    else:
        return matches[-1].replace("#### ", "")


def get_gsm8k_answer(solution):
    # print(solution)
    # The answer is 48
    # The answer is $48.
    # there could be a $ sign
    regex = re.compile(r"(The answer is \$?\d+)")
    matches = regex.findall(solution)
    if len(matches) == 0:
        return "Y"
    else:
        return matches[-1].replace("The answer is ", "").replace("$", "")
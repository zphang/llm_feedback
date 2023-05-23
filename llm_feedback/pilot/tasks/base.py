import os
from langchain.chat_models import ChatOpenAI
import dataclasses
from typing import List, Dict, Optional


@dataclasses.dataclass
class BaseTask:
    def get_dataset(self, phase: str):
        """Get dataset for a phase

        :return: something that is iterable and yields examples. Examples should be directly consumed by the chain
        """
        raise NotImplementedError()

    def get_chain(
        self,
        generation_llm: str,
        feedback_llm: str,
        refinement_llm: str,
        chain_name: Optional[str] = None,
    ):
        """Return LLM langchain

        Note: we only pass in the model names, not the actual models.
        It is up to the chain implementation to decide what type of model to use (e.g. LLM, Chat)

        :param generation_llm: LLM for generation
        :param feedback_llm: LLM for feedback
        :param refinement_llm: LLM for refinement
        :param chain_name: If there are different langchains (e.g. different prompt formats), otherwise ignore this.
        :return: Langchain that takes an example (from .get_dataset) as input and outputs a dictionary of
                 outputs (and all intermediate inputs)
        """
        raise NotImplementedError()

    def process(self, chain, example):
        """Process an example. Override in cases where we need to preprocess the example (e.g. rename keys)

        :param langchain: langchain from get_chain
        :param example: example from get_dataset
        :return: dictionary of outputs (and all intermediate inputs)
        """
        return chain(example)

    def evaluate(self, phase: str, outputs: List[Dict]):
        """Takes a list of outputs (from get_chain) and evaluates them

        Re-load the dataset if necessary.
        Num examples should be len(outputs) corresponding to the first N examples, though this may change in the
         future if we start shuffling.

        :param phase: e.g. "train", "validation", "test"
        :param outputs: list of outputs from get_chain
        :return: dictionary of metrics (make sure it's JSON serializable)
        """
        raise NotImplementedError()
    
    def get_llm(self, model_name: str):
        """Return LLM model

        :param model_name: model name
        :return: LLM model
        """
        print(model_name)
        if "vicuna" in model_name or "llama" in model_name:
            import openai
            openai.api_key = "EMPTY" # Not support yet
            openai.api_base = "http://localhost:8000/v1"
        else:
            # reset to default
            os.environ["OPENAI_API_BASE"] = ""
        return ChatOpenAI(model_name=model_name)

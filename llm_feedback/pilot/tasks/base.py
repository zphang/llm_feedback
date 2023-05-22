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

        :param chain: langchain from get_chain
        :param example: example from get_dataset
        :return: dictionary of outputs (and all intermediate inputs)
        """
        return chain(example)

    def batch_process(self, chain, example_list):
        """Process a batch of examples.

        :param chain: langchain from get_chain
        :param example_list: example from get_dataset
        :return: dictionary of outputs (and all intermediate inputs)
        """
        return [self.process(chain=chain, example=example) for example in example_list]

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

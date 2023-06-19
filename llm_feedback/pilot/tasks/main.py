from typing import Optional
from . import base
from . import alfworld
from . import beerqa
from . import fever
from . import hotpotqa
from . import example
from . import mathqa
from . import mbpp
from . import slf5k


def get_task(task_name: str, task_args_str: Optional[str] = None) -> base.BaseTask:
    """Get task by name"""
    if task_name == "example":
        return example.ExampleTask()
    elif task_name == "alfworld":
        return alfworld.AlfworldTask()
    elif task_name == "beerqa":
        return beerqa.BeerQATask(task_args_str=task_args_str)
    elif task_name == "fever":
        return fever.FEVERTask(task_args_str=task_args_str)
    elif task_name == "hotpotqa":
        return hotpotqa.HotPotQATask(task_args_str=task_args_str)
    elif task_name == "mathqa":
        return mathqa.MathQATask()
    elif task_name == "mbpp":
        return mbpp.MBPPTask()
    elif task_name == "slf5k":
        return slf5k.SLF5KTask()
    else:
        raise ValueError("Unknown task {}".format(task_name))

from . import base
from . import example
from . import mathqa
from . import mbpp


def get_task(task_name: str) -> base.BaseTask:
    """Get task by name"""
    if task_name == "example":
        return example.ExampleTask()
    elif task_name == "mathqa":
        return mathqa.MathQATask()
    elif task_name == "mbpp":
        return mbpp.MBPPTask()
    else:
        raise ValueError("Unknown task {}".format(task_name))

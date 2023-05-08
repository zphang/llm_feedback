from . import base
from . import example
from . import mathqa
from . import mbpp
from . import slf5k


def get_task(task_name: str) -> base.BaseTask:
    """Get task by name"""
    if task_name == "example":
        return example.ExampleTask()
    elif task_name == "mathqa":
        return mathqa.MathQATask()
    elif task_name == "mbpp":
        return mbpp.MBPPTask()
    elif task_name == "slf5k":
        return slf5k.SLF5KTask()
    else:
        raise ValueError("Unknown task {}".format(task_name))

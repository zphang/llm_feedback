from langchain.chat_models import ChatOpenAI, ChatAnthropic

OPENAI_MODEL_NAMES = [
    "gpt-3.5-turbo-0301",
    "gpt-4-0314",
]
ANTHROPIC_MODEL_NAMES = [
    "claude-v1",
    "claude-v1-100k",
    "claude-instant-v1",
    "claude-v1.0",
    "claude-v1.2",
    "claude-v1.3",
    "claude-instant-v1.0",
    "claude-instant-v1.1",
]


def get_chat_model(model_name: str, **kwargs):
    if is_openai_model(model_name):
        return ChatOpenAI(model_name=model_name, **kwargs)
    elif is_anthropic_model(model_name):
        return ChatAnthropic(model=model_name, **kwargs)
    else:
        raise KeyError(model_name)


def is_openai_model(model_name):
    return model_name in OPENAI_MODEL_NAMES


def is_anthropic_model(model_name):
    return model_name in ANTHROPIC_MODEL_NAMES

import os
import dotenv


def get_env_path():
    return os.path.abspath(os.path.join(os.path.split(__file__)[0], "..", "..", ".env"))


def load_dotenv():
    dotenv.load_dotenv(get_env_path(), override=True)

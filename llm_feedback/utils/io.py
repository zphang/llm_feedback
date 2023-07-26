import re
import json


def read_file(path, mode="r", **kwargs):
    with open(path, mode=mode, **kwargs) as f:
        return f.read()


def write_file(data, path, mode="w", **kwargs):
    with open(path, mode=mode, **kwargs) as f:
        f.write(data)


def read_json(path):
    return json.loads(read_file(path))


def write_json(data, path):
    json_str = json.dumps(data, indent=2)
    json_str = re.sub(r'": \[\s+', '": [', json_str)
    json_str = re.sub(r'",\s+', '", ', json_str)
    json_str = re.sub(r'"\s+\]', '"]', json_str)
    return write_file(json_str, path)


def read_jsonl(path):
    # Manually open because .splitlines is different from iterating over lines
    ls = []
    with open(path, "r") as f:
        for line in f:
            ls.append(json.loads(line))
    return ls


def write_jsonl(data, path):
    assert isinstance(data, list)
    lines = [
        to_jsonl(elem)
        for elem in data
    ]
    write_file("\n".join(lines), path)


def to_jsonl(data):
    json_str = json.dumps(data).replace("\n", "")
    json_str = re.sub(r'": \[\s+', '": [', json_str)
    json_str = re.sub(r'",\s+', '", ', json_str)
    json_str = re.sub(r'"\s+\]', '"]', json_str)
    return json_str

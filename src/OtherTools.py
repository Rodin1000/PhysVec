import os
import sys
import asyncio
import json


def load_task_parameters(topic: str, task_name: str) -> dict:
    task_json_path = os.path.join("..", "Paper_dataset", topic, "tasks", task_name)
    if os.path.exists(task_json_path):
        try:
            with open(task_json_path, 'r', encoding='utf-8') as _f:
                task_data = json.load(_f)
            if "LimitingCases" not in task_data:
                task_data["LimitingCases"] = None
            if "ScientificResults" not in task_data:
                task_data["ScientificResults"] = None
            return task_data
        except Exception as _e:
            raise RuntimeError(f"Failed to load task JSON '{task_json_path}': {_e}")
    else:
        raise FileNotFoundError(f"Task file '{task_json_path}' not found.")

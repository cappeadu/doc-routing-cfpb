import json
from pathlib import Path


def save_dict(
    directory: str, dict_to_save: dict, file_name: str = "evaluation_results"
) -> dict:
    """Save dict eg. results from computing metrics.

    Args:
        directory (str): directory to save dict.
        dict_to_save (dict): dict object to save.
        file_name (str): file name. Defaults to "evaluation_results"

    """
    dir_path = Path(__file__).parent.parent / directory
    if not dir_path.exists():
        dir_path.mkdir(exist_ok=True)
    with open(f"{dir_path}/{file_name}.json", "w") as fp:
        json.dump(dict_to_save, fp=fp, indent=2)
        fp.write("\n")

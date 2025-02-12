import json
import os
from datetime import datetime


def save_data(
    results: list,
    file_name: str
) -> None:
    """Save extracted data in a JSON file with proper encoding and timestamp."""

    date_stamp = datetime.now().strftime("%d%m%Y")
    path = f"../data/data_extracted_{file_name}_{date_stamp}.json"
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f_json:
        json.dump(results, f_json, ensure_ascii=False, indent=2)

    print(f"-----> Data successfully saved in: {path}")
    
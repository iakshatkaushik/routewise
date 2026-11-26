"""Step 1 - download, clean and split the data (Data Pre-processing)."""
from _common import save_json
from llm_router import data

if __name__ == "__main__":
    report = data.build_processed()
    print(report)
    save_json("01_data_report.json", report)

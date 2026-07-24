import json


def process_data(input_file_path):
    with open(input_file_path, "r") as file:
        original_data = json.load(file)

    formatted_data = []

    for entry in original_data:
        merged_entry = []
        instruction = entry["instruction"] + entry["input"]
        history = entry["history"]

        merged_entry.append(instruction)
        merged_entry.append(entry["output"])

        for pair in history[1:]:
            for item in pair:
                merged_entry.append(item)

        boolean_flags = [True, False] * len(history)
        formatted_entry = [merged_entry, boolean_flags]
        formatted_data.append(formatted_entry)

    with open(input_file_path, "w") as file:
        json.dump(formatted_data, file, indent=4)

    print(f"{input_file_path}")


train_file_path = "./dbgpt_hub/data/spider_codes/example_text2sql_train.json"
dev_file_path = "./dbgpt_hub/data/spider_codes/example_text2sql_dev.json"
process_data(train_file_path)
process_data(dev_file_path)

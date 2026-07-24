import os
import json
import sys
from multiprocessing import Process, Manager

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_PATH)

from tqdm import tqdm
from typing import List, Dict, Optional, Any

from dbgpt_hub.data_process.data_utils import extract_sql_prompt_dataset
from dbgpt_hub.llm_base.chat_model import ChatModel


def prepare_dataset(
    predict_file_path: Optional[str] = None,
    split_part: Optional[str] = None
) -> List[Dict]:
    with open(predict_file_path, "r") as fp:
        data = json.load(fp)
    if split_part is not None:
        part,_,split=split_part.partition(":")
        start=int(int(part)/int(split)*len(data))
        end=int((int(part)+1)/int(split)*len(data))
        data=data[start:end]
    predict_data = [extract_sql_prompt_dataset(item) for item in data]
    return predict_data


def inference(model: ChatModel, predict_data: List[Dict], predicted_out_filename, **input_kwargs):
    res = []
    # test
    # for item in predict_data[:20]:
    for item in tqdm(predict_data, desc="Inference Progress", unit="item"):
        # print(f"item[input] \n{item['input']}")
        response, _ = model.chat(query=item["input"], history=[], **input_kwargs)
        res.append(response)

        if len(res)%16==0:
            with open(predicted_out_filename, "w") as f:
                for p in res:
                    try:
                        f.write(p.replace("\n", " ") + "\n")
                    except:
                        f.write("Invalid Output!\n")

    return res


def predict(model: ChatModel):
    args = model.data_args
    ## predict file can be give by param --predicted_input_filename ,output_file can be gived by param predicted_out_filename
    predict_data = prepare_dataset(args.predicted_input_filename, args.split_part)
    result = inference(model, predict_data, args.predicted_out_filename)

    with open(args.predicted_out_filename, "w") as f:
        for p in result:
            try:
                f.write(p.replace("\n", " ") + "\n")
            except:
                f.write("Invalid Output!\n")


if __name__ == "__main__":
    model = ChatModel()
    predict(model)

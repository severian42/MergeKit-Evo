import re
import argparse

from transformers import AutoTokenizer, AutoModelForCausalLM
import datasets

def extract_regex(text):
    num_list = re.findall(r'\d+', text)
    if num_list is not None:
        return num_list
    else:
        return None

def score_mgsm(pred, answer_number) -> bool:
    if "." in pred:
        pred = pred.rstrip("0").rstrip(".")
    pred = pred.replace(",", "")
    if str(answer_number) == pred:
        return 1
    else:
        return 0

def main(args):
    df = datasets.load_dataset("juletxara/mgsm", "ja")
    df = df["test"]

    model = AutoModelForCausalLM.from_pretrained(args.model_id, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    cnt = 0
    for i in range(len(df)):
        question = df["question"][i]
        prompt = f"Please solve the following math problem. Describe the reasoning process of the solution before giving the final answer. In the answer, include only the integer answer without any additional text.\nProblem:{question}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        tokens = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
        )
        output = tokenizer.decode(tokens[0], skip_special_tokens=True)
        try:
            pred = extract_regex(output)[-1]
            score = score_mgsm(pred, df["answer_number"][i])
        except IndexError:
            score = 0
        cnt += score
        print(i)
    print(cnt/len(df))        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id")
    args = parser.parse_args()
    main(args)

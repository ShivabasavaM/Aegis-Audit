import os
import json
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from langchain_groq import ChatGroq
from ragas.metrics import Faithfulness

load_dotenv()

def load_jsonl_dataset(file_path):
    data_dict = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }
    
    print(f"Loading data from {file_path}...")
    with open(file_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            data_dict["question"].append(item.get("question", ""))
            data_dict["answer"].append(item.get("answer", ""))
            data_dict["contexts"].append(item.get("contexts", []))
            data_dict["ground_truth"].append(item.get("ground_truth", ""))
            
    MAX_SAMPLES = 30
    dataset_size = len(data_dict["question"])
    
    if dataset_size > MAX_SAMPLES:
        print(f"Truncating dataset from {dataset_size} down to {MAX_SAMPLES} to respect Groq API limits...")
        data_dict["question"] = data_dict["question"][:MAX_SAMPLES]
        data_dict["answer"] = data_dict["answer"][:MAX_SAMPLES]
        data_dict["contexts"] = data_dict["contexts"][:MAX_SAMPLES]
        data_dict["ground_truth"] = data_dict["ground_truth"][:MAX_SAMPLES]
        dataset_size = MAX_SAMPLES

    print(f"✅ Successfully loaded {dataset_size} evaluation samples!")
    return Dataset.from_dict(data_dict), dataset_size

def run_offline_evaluation():
    print("Initializing the Groq Judge")
    
    judge_llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.0 
    )

    dataset_path = "ragas_eval_dataset.jsonl"
    dataset, size = load_jsonl_dataset(dataset_path)

    if size == 0:
        print("⚠️ JSONL file is empty")
        return

    print(f"Firing Ragas Evaluation Pipeline across {size} documents...")
    
    faithfulness_metric = Faithfulness(llm=judge_llm)

    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness_metric]
    )

    print("\n Final Evaluation Results:")
    print(results)
    print("\n Metric!")

if __name__ == "__main__":
    run_offline_evaluation()
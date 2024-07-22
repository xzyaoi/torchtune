import json
import numpy as np

def sample_training_data(args):
    print(args)
    with open(args.input_jsonl, "r") as f:
        data = [json.loads(line) for line in f]
    print(f"Loaded {len(data)} samples")
    sample_size = int(len(data) * args.sample_ratio)
    print(f"Sampling {sample_size} samples")
    # set seed
    np.random.seed(args.seed)
    sampled_data = np.random.choice(data, sample_size, replace=False)
    with open(args.output_jsonl, "w+") as f:
        for d in sampled_data:
            f.write(json.dumps(d) + "\n")


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=str, help="input jsonl file", required=True)
    parser.add_argument("--sample-ratio", type=float, help="sample ratio", required=True)
    parser.add_argument("--output-jsonl", type=str, help="output jsonl file", required=True)
    parser.add_argument("--seed", type=int, help="random seed", default=42)
    args = parser.parse_args()
    sample_training_data(args)
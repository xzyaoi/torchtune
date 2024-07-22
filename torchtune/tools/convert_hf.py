import torch

def convert_hf(args):
    print(args)
    with open(args.finetuned_weights, "rb") as f:
        weights = torch.load(f)
        print(weights.keys())

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", type=str, required=True)
    parser.add_argument("--finetuned-weights", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    convert_hf(parser.parse_args())
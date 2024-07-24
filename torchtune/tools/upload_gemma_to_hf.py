import os
import torch
from accelerate import init_empty_weights
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from tqdm import tqdm

def upload_ft_gemma_to_hf(args):
    print(args)
    config = AutoConfig.from_pretrained(args.ckpt_path)
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    checkpoints = [x.replace("hf_model_", "") for x in os.listdir(args.ckpt_path) if x.endswith(".pt") and x.startswith("hf_model_")]
    shards = set()
    epochs = set()
    global_steps = {}
    has_final_ckpts = {}
    for ckpt in checkpoints:
        ckpt = ckpt.replace(".pt", "")
        shard = int(ckpt.split("_")[0])
        shards.add(shard)
        epoch = int(ckpt.split("_")[1])
        epochs.add(epoch)
        if len(ckpt.split("_")) > 2:
            if epoch not in global_steps:
                global_steps[epoch] = set()
            global_steps[epoch].add(int(ckpt.split("_")[2]))
        else:
            has_final_ckpts[epoch] = True
    print(f"Shards: {shards}, Epochs: {epochs}, Global Steps: {global_steps}, Has final ckpts: {has_final_ckpts}")
    
    for epoch in epochs:
        print(f"Uploading epoch {epoch}")
        for global_step in global_steps[epoch]:
            print(f"Uploading global step {global_step}")
            loaded = {}
            with init_empty_weights():
                model = AutoModelForCausalLM.from_config(config)
                # https://github.com/vllm-project/vllm/issues/3323
                del model.lm_head
            for shard in shards:
                with open(f"{args.ckpt_path}/hf_model_{shard:04d}_{epoch}_{global_step}.pt", "rb") as f:
                    data = torch.load(f)
                    loaded.update(data)
            model.load_state_dict(loaded, strict=True, assign=True)
            config.push_to_hub(
                repo_id=args.hf_repo, 
                commit_message=f"config: epoch={epoch}; step={global_step}",
                revision=f"epoch_{epoch}-step_{global_step}",
                private=True,
            )
            tokenizer.push_to_hub(
                repo_id=args.hf_repo, 
                commit_message=f"tokenizer: epoch={epoch}; step={global_step}",
                revision=f"epoch_{epoch}-step_{global_step}",
                private=True,
            )
            model.push_to_hub(
                repo_id=args.hf_repo, 
                commit_message=f"ckpt: epoch={epoch}; step={global_step}",
                revision=f"epoch_{epoch}-step_{global_step}",
                private=True,
            )
        if has_final_ckpts[epoch]:
            print(f"Uploading final step in epoch {epoch}")
            loaded = {}
            with init_empty_weights():
                model = AutoModelForCausalLM.from_config(config)
                del model.lm_head
            for shard in shards:
                with open(f"{args.ckpt_path}/hf_model_{shard:04d}_{epoch}.pt", "rb") as f:
                    data = torch.load(f)
                    loaded.update(data)
            model.load_state_dict(loaded, strict=True, assign=True)
            config.push_to_hub(
                repo_id=args.hf_repo, 
                commit_message=f"config: epoch={epoch};",
                revision=f"epoch_{epoch}-final",
                private=True,
            )
            tokenizer.push_to_hub(
                repo_id=args.hf_repo, 
                commit_message=f"tokenizer: epoch={epoch};",
                revision=f"epoch_{epoch}-final",
                private=True,
            )
            model.push_to_hub(
                repo_id=args.hf_repo, 
                commit_message=f"ckpt: epoch={epoch};",
                revision=f"epoch_{epoch}-final",
                private=True,
            )
        print(f"Uploaded epoch {epoch}")
    
if __name__=="__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--hf-repo", type=str, required=True)
    parser.add_argument('--temp-dir', type=str, default=None)
    parser.add_argument('--tokenizer', type=str, default=None)
    upload_ft_gemma_to_hf(parser.parse_args())
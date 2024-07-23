# adapted from https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py
import os
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

def convert_hf(args):
    print(args)
    with open(args.finetuned_weights, "rb") as f:
        loaded = torch.load(f)
    config = AutoConfig.from_pretrained(args.pretrained)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    n_shards = 1
    n_layers = config.num_hidden_layers
    n_heads = config.num_attention_heads
    n_heads_per_shard = n_heads // n_shards
    dim = config.hidden_size
    dims_per_head = dim // n_heads
    n_kv_heads = config.num_key_value_heads
    n_local_kv_heads = n_kv_heads // n_shards
    kv_dim = dims_per_head * n_kv_heads

    state_dict = {}
    print(f"num_hidden_layers: {n_layers}, num_attention_heads: {n_heads}, hidden_size: {dim}, num_key_value_heads: {n_kv_heads}, num_local_kv_heads: {n_local_kv_heads}, kv_dim: {kv_dim}, n_heads_per_shard: {n_heads_per_shard}")

    def permute(w, n_heads, dim1=dim, dim2=dim):
        return w.view(n_heads, dim1 // n_heads // 2, 2, dim2).transpose(1, 2).reshape(dim1, dim2)

    for layer_idx in tqdm(range(n_layers)):
        state_dict[f'model.layers.{layer_idx}.self_attn.q_proj.weight'] = permute(
            loaded[f'layers.{layer_idx}.attention.wq.weight'], n_heads=n_heads)
        state_dict[f'model.layers.{layer_idx}.self_attn.k_proj.weight'] = permute(
            loaded[f'layers.{layer_idx}.attention.wk.weight'], n_heads=n_kv_heads, dim1=kv_dim)
        state_dict[f'model.layers.{layer_idx}.self_attn.v_proj.weight'] = loaded[f'layers.{layer_idx}.attention.wv.weight']
        state_dict[f'model.layers.{layer_idx}.self_attn.o_proj.weight'] = loaded[f'layers.{layer_idx}.attention.wo.weight']
        state_dict[f'model.layers.{layer_idx}.mlp.gate_proj.weight'] = loaded[f'layers.{layer_idx}.feed_forward.w1.weight']
        state_dict[f'model.layers.{layer_idx}.mlp.down_proj.weight'] = loaded[f'layers.{layer_idx}.feed_forward.w2.weight']
        state_dict[f'model.layers.{layer_idx}.mlp.up_proj.weight'] = loaded[f'layers.{layer_idx}.feed_forward.w3.weight']
        state_dict[f'model.layers.{layer_idx}.input_layernorm.weight'] = loaded[f'layers.{layer_idx}.attention_norm.weight']
        state_dict[f'model.layers.{layer_idx}.post_attention_layernorm.weight'] = loaded[f'layers.{layer_idx}.ffn_norm.weight']
        
    state_dict['model.embed_tokens.weight'] = loaded['tok_embeddings.weight']
    state_dict['model.norm.weight'] = loaded['norm.weight']
    state_dict['lm_head.weight'] = loaded["output.weight"]
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    tokenizer.save_pretrained(args.output)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)
    model.load_state_dict(state_dict, assign=True, strict=True)
    model.save_pretrained(args.output, safe_serialization=True)
    
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", type=str, required=True)
    parser.add_argument("--finetuned-weights", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    convert_hf(parser.parse_args())

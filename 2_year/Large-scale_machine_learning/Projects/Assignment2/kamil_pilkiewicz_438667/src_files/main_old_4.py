from functools import partial
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from types import SimpleNamespace
from torch.optim import AdamW
import torch.nn.functional as F
from torch.nn.attention import SDPBackend
from collections import OrderedDict
from datasets import load_dataset, load_from_disk
from transformers import GPT2TokenizerFast

# CHANGE BEGIN
import argparse
import wandb
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import os
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    enable_wrap,
    wrap,
)
from torch.optim.lr_scheduler import LambdaLR

import threading
import GPUtil
import psutil
import time
# CHANGE END

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_len):
        super(EmbeddingLayer, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_len, embed_dim)

    def forward(self, x):
        # x: (batch_size, seq_len)
        seq_len = x.size(1)
        positions = (
            torch.arange(seq_len, dtype=torch.long, device=x.device)
            .unsqueeze(0)
            .expand_as(x)
        )
        token_embeddings = self.token_embedding(x)
        position_embeddings = self.position_embedding(positions)
        embeddings = token_embeddings + position_embeddings
        return embeddings


class AttentionLayer(nn.Module):
    def __init__(
        self,
        dmodel,
        heads,
    ):
        super(AttentionLayer, self).__init__()

        self.ln = nn.LayerNorm(dmodel)

        self.heads = heads

        self.input_projection = nn.Linear(dmodel, 3 * dmodel, bias=False)

        self.output_projection = nn.Linear(dmodel, dmodel, bias=False)

    def forward(self, x, attention_mask):
        x = self.ln(x)

        projected = self.input_projection(x)

        batch, seq_len = x.shape[:-1]
        q_chunk, k_chunk, v_chunk = torch.chunk(projected, chunks=3, dim=-1)
        query = q_chunk.view(batch, seq_len, self.heads, -1).transpose(1, 2)
        key = k_chunk.view(batch, seq_len, self.heads, -1).transpose(1, 2)
        value = v_chunk.view(batch, seq_len, self.heads, -1).transpose(1, 2)

        with torch.nn.attention.sdpa_kernel(
            [
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.MATH,
            ]
        ):
            attention_output = F.scaled_dot_product_attention(
                query=query,
                key=key,
                value=value,
                attn_mask=attention_mask,
                is_causal=True,
            )

        output = self.output_projection(attention_output.transpose(1, 2).flatten(-2))

        return output


def FeedForward(
    dmodel,
):
    return nn.Sequential(
        OrderedDict(
            [
                (
                    "ff_layernorm",
                    nn.LayerNorm(dmodel)
                ),
                (
                    "pre_relu",
                    nn.Linear(
                        dmodel,
                        4 * dmodel,
                        bias=True,
                    ),
                ),
                ("relu", nn.ReLU()),
                (
                    "post_relu",
                    nn.Linear(
                        4 * dmodel,
                        dmodel,
                        bias=True,
                    ),
                ),
            ]
        )
    )

# CHANGE BEGIN: defince SwiGLU
class SwiGLU(nn.Module):
    def __init__(
        self,
        dmodel,
        beta=1.,
        scale=4,
        bias=True
    ):
        super(SwiGLU, self).__init__()

        self.beta = beta
        self.ln = nn.LayerNorm(dmodel)
        self.swish_projection1 = nn.Linear(
            dmodel,
            scale * dmodel,
            bias=bias,
        )
        self.swish_projection2 = nn.Linear(
            dmodel,
            scale * dmodel,
            bias=bias,
        )
        self.output_projection = nn.Linear(scale*dmodel, dmodel, bias=bias)
        # Note, that in paper they propose omit bias, but it should not harm, maybe even help during
        # training.
    
    def swish(self, x):
        y = torch.sigmoid(self.beta * x)
        return x * y

    def forward(self, x):
        x = self.ln(x)
        x1 = self.swish(self.swish_projection1(x))
        x2 = self.swish_projection2(x)
        x = x1 * x2
        return self.output_projection(x)
# CHANGE END

class Block(nn.Module):

    def __init__(
        self,
        dmodel,
        heads,
    ):
        super().__init__()
        self.attention_layer = AttentionLayer(dmodel, heads)
        # self.feed_forward_layer = FeedForward(dmodel) # DELETED
        self.swiglu_layer = SwiGLU(dmodel) # ADDED

    def forward(self, x, attention_mask):
        out_attention = self.attention_layer(x, attention_mask)
        x = x + out_attention

        # CHANGE BEGIN
        # out_feed_forward = self.feed_forward_layer(x)
        # x = x + out_feed_forward
        out_swiglu = self.swiglu_layer(x) 
        x = x + out_swiglu
        # CHANGE END
        return x


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding_layer = EmbeddingLayer(
            config.vocab_size, config.d_model, config.max_len
        )
        self.blocks = nn.ModuleList(
            [Block(config.d_model, config.num_heads) for _ in range(config.num_layers)]
        )

        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None):
        output = self.embedding_layer(input_ids)

        for block in self.blocks:
            output = block(output, attention_mask)

        output = self.head(output)
        return output


def collate_tokenize(tokenizer, sequence_length, data):
    text_batch = [element["text"] for element in data]
    tokenized = tokenizer(
        text_batch,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=sequence_length + 1,
    )
    input_ids = tokenized["input_ids"]
    tokenized["input_ids"] = input_ids[:, :-1]
    tokenized["target_ids"] = input_ids[:, 1:]
    tokenized["attention_mask"] = tokenized["attention_mask"][:, :-1]
    return tokenized


def get_dataloader(
    batch_size,
    sequence_length,
    split="train",
    buffer_size=10000,
    seed=42,
    num_workers=2,
):
    if split == "train":
        hf_dataset = load_from_disk("/net/tscratch/people/plgjkrajewski/datasets/c4/train")
    else:
        hf_dataset = load_from_disk("/net/tscratch/people/plgjkrajewski/datasets/c4/validation")
    hf_dataset = hf_dataset.to_iterable_dataset(num_shards=64)
    hf_dataset = hf_dataset.shuffle(buffer_size=buffer_size, seed=seed)
    # BEGIN CHANGE
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    # if split == "train":
    #     sampler = DistributedSampler(hf_dataset, rank=rank, num_replicas=world_size, shuffle=True)
    # else:
    #     sampler = DistributedSampler(hf_dataset, rank=rank, num_replicas=world_size)
    hf_dataset = hf_dataset.shard(num_shards=world_size, index=rank)
    # END CHANGE
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    dataloader = DataLoader(
        hf_dataset,
        batch_size=batch_size,
        collate_fn=partial(collate_tokenize, tokenizer, sequence_length),
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        # sampler=sampler, # ADDED
    )
    return dataloader


def calculate_valid_loss(model, valid_dataloader, device, validation_steps):
    valid_losses = []
    for _, batch in zip(range(validation_steps), valid_dataloader):
        with torch.no_grad():
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            attention_mask = batch["attention_mask"]
            outputs = model(input_ids)
            mask_loss = F.cross_entropy(
                outputs.flatten(0, -2),
                target_ids.reshape(-1).long(),
                reduction="none",
            )
            mask_loss = mask_loss[attention_mask.reshape(-1) == 1]
            loss = mask_loss.mean().item()
            valid_losses.append(loss)
            mean_valid_loss = sum(valid_losses) / validation_steps
    return mean_valid_loss

# CHANGE BEGIN: add GPU monitoring
def start_monitor(wandb_logger=None, interval=2):
    stop_event = threading.Event()
    
    def monitor():
        while not stop_event.is_set():
            gpus = GPUtil.getGPUs()
            log_dict = {}
            for i, gpu in enumerate(gpus):
                log_dict[f"gpu{i}_mem_MB"] = gpu.memoryUsed
                log_dict[f"gpu{i}_util"] = gpu.load*100
            log_dict["cpu_percent"] = psutil.cpu_percent()
            log_dict["ram_used_GB"] = psutil.virtual_memory().used / 1e9
            if wandb_logger is not None:
                wandb_logger.log(log_dict)
            time.sleep(interval)
    
    t = threading.Thread(target=monitor, daemon=True)
    t.start()
    return stop_event, t
# CHANGE END


def train_model(config, device, wandb_logger):
    # CHANGE BEGIN - add monitoring
    if wandb_logger is not None:
        stop_event, monitor_thread = start_monitor(wandb_logger)
    # CHANGE END
    dataloader = get_dataloader(config.batch_size, config.seq_length)
    valid_dataloader = get_dataloader(config.batch_size, config.seq_length, split="validation")
    validation_steps = int(1e06 // (config.batch_size * config.seq_length)) # we want to evaluate on 1M tokens
    model = Transformer(config)
    # CHANGE BEGIN - FSDP
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD #SHARD_GRAD_OP
    mp_policy = MixedPrecision(
      param_dtype=torch.bfloat16,
      # Gradient communication precision.
      reduce_dtype=torch.bfloat16,
      # Buffer precision.
      buffer_dtype=torch.bfloat16,
    )
    custom_auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={Block}
    )
    model = FSDP(model,
        auto_wrap_policy=custom_auto_wrap_policy,
        mixed_precision=mp_policy,
        sharding_strategy=sharding_strategy,
        device_id=torch.cuda.current_device())
    # CHANGE END

    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    
    # CHANGE BEGIN - WSD learning rate
    total_steps = config.train_steps
    warmup_steps = total_steps // 100 # so 1%
    decay_steps = total_steps // 10 # 10%
    stable_steps = total_steps - warmup_steps - decay_steps
    def wsd_lr_lambda(steps):
        steps += 1 # We should start counting from 1, not 0
        if steps < warmup_steps:
            return steps / max(1, warmup_steps)
        elif steps < warmup_steps + stable_steps:
            return 1.
        else:
            return max(0., 1. - (steps - warmup_steps - stable_steps) / max(1, decay_steps))
    
    scheduler = LambdaLR(optimizer, lr_lambda=wsd_lr_lambda)
    # CHANGE END

    model.train()

    for i, batch in zip(range(config.train_steps), dataloader):
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device) # CHANGE: to(device)

        optimizer.zero_grad()
        outputs = model(input_ids)

        mask_loss = F.cross_entropy(
            outputs.flatten(0, -2),
            target_ids.reshape(-1).long(),
            reduction="none",
        )
        mask_loss = mask_loss[attention_mask.reshape(-1) == 1]
        loss = mask_loss.mean()
        
        # CHANGE BEGIN (wandb logger + valid freq in proper place + usage of WSD lr)
        if i % config.log_train_loss_freq == 0:
            print(f"Step:{i}, Train individual Loss:{loss}")
            loss_for_log = loss.detach().clone()
            dist.all_reduce(loss_for_log, op=dist.ReduceOp.AVG)
            print(f"Step:{i}, Train Loss:{loss_for_log}")
            if wandb_logger is not None:
                wandb_logger.log({"step": i, "train_loss": loss_for_log})

        if i % config.log_valid_loss_freq == 0:
            val_loss = calculate_valid_loss(model, valid_dataloader, device, validation_steps)
            print(f"Valid individual loss:{val_loss}")
            val_loss = torch.tensor(val_loss, device=device)
            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG) 
            # Each node computes val_loss on 1e6 of tokens,
            # these tokens are different for different nodes. Then
            # we take avereage of these losses. 
            print(f"Valid loss:{val_loss}")
            if wandb_logger is not None:
                wandb_logger.log({"step": i, "val_loss": val_loss})

        loss.backward()
        optimizer.step()
        scheduler.step() # CHANGED
        if wandb_logger is not None:
            wandb_logger.log({"lr": optimizer.param_groups[0]["lr"]}) # CHANGED
    val_loss = calculate_valid_loss(model, valid_dataloader, device, validation_steps)
    val_loss = torch.tensor(val_loss, device=device) # CHANGED
    dist.all_reduce(val_loss, op=dist.ReduceOp.AVG) # CHANGED
    print(f"Final valid loss:{val_loss}")
    if wandb_logger is not None:
        wandb_logger.log({"step": config.train_steps,"val_loss": val_loss})
    # CHANGE END
    # CHANGE BEGIN: end GPU monitoring
    if wandb_logger is not None:
        stop_event.set()
        monitor_thread.join()
    # CHANGE END


def setup():
    dist.init_process_group("nccl")

def cleanup():
    dist.barrier()
    dist.destroy_process_group()


def main():
    # Arguments:
    # Default: n_layers=4, dmodel=256, n_heads=4, batch_size=64, n_training_steps=1000
    parser = argparse.ArgumentParser(description="Hyperparameters of transformer")
    parser.add_argument("n_layers", type=int, help="Number of layers")
    parser.add_argument("dmodel", type=int, help="Dimension of model embedding")
    parser.add_argument("n_heads", type=int, help="Number of heads")
    parser.add_argument("batch_size", type=int, help="Batch size")
    parser.add_argument("n_training_steps", type=int, help="Number of training steps")
    args = parser.parse_args()

    local_rank = int(os.environ['LOCAL_RANK'])
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    # wandb init:
    run = wandb.init(
        entity="ka-pilkiewic-university-of-warsaw",
        project="lml-assignment2-project",
        config=vars(args),
    ) if rank == 0 else None

    config = SimpleNamespace(
        train_steps=args.n_training_steps,
        vocab_size=50257,
        max_len=256,
        d_model=args.dmodel,
        num_heads=args.n_heads,
        num_layers=args.n_layers,
        learning_rate=1e-4,
        dropout=0.0,
        seq_length=256,
        batch_size=args.batch_size,
        log_train_loss_freq=100,
        log_valid_loss_freq=100
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print(f"Device type is: {device}. Remember to train on GPU.")

    setup()
    
    train_model(config, device, run)

    cleanup()
    if run is not None:
        run.finish()

if __name__ == "__main__":
    main()

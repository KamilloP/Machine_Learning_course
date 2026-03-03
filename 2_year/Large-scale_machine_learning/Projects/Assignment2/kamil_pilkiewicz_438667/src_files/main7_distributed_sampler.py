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
import os
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.optim.lr_scheduler import LambdaLR
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

# CHANGE BEGIN
class SwiGLU(nn.Module):
    """
    Arguments:
    1) dmodel: Dimension of model embedding.
    2) beta:   Parameter of function Swish_beta. Default=1.
    3) scale:  Scale of hidden dimension in regard to dmodel. In ff_layer it was 4, but
               in paper they propose to change it into 2/3 of that value, so approximately 2.66666. 
               The reason is that we want to keep similar number of parameters.
    4) bias:   Whether we want add learnable bias in linear projections or not. In paper both options
               were considered, but eventually they have ommited bias. However bias should not harm
               training effectiveness, so we keep it (same as in ff_layer). 
    """
    def __init__(
        self,
        dmodel,
        beta=1.,
        scale=2.66666,
        bias=True
    ):
        super(SwiGLU, self).__init__()

        self.beta = beta
        hidden_dim = int(scale * dmodel)
        self.ln = nn.LayerNorm(dmodel)
        self.swish_projection1 = nn.Linear(
            dmodel,
            hidden_dim,
            bias=bias,
        )
        self.swish_projection2 = nn.Linear(
            dmodel,
            hidden_dim,
            bias=bias,
        )
        self.output_projection = nn.Linear(hidden_dim, dmodel, bias=bias)
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

        out_swiglu = self.swiglu_layer(x) # SMALL CHANGE
        x = x + out_swiglu
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
   
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    distributed_sampler = torch.utils.data.distributed.DistributedSampler(
        hf_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=seed,
        drop_last=True
    )
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    dataloader = DataLoader(
        hf_dataset,
        batch_size=batch_size,
        collate_fn=partial(collate_tokenize, tokenizer, sequence_length),
        shuffle=False,  
        pin_memory=True,
        num_workers=num_workers,
        sampler=distributed_sampler,  
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


def count_fsdp_params(model):
    local_params = 0
    sharded_params = 0
    for p in model.parameters():
        if hasattr(p, "_local_shard"):
            sharded_params += p._local_shard.numel()
        else:
            local_params += p.numel()
    return local_params, sharded_params

def wsd_lr_scheduler(optimizer, total_steps):
    """
    Arguments:
    1) optimizer:   Optimizer where we want to apply our scheduler to.
    2) total_steps: Total number of training steps.
    
    Return: WSD learning rate scheduler, already applied to optimizer.
    """
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
    return LambdaLR(optimizer, lr_lambda=wsd_lr_lambda) 

def train_model(config, device, wandb_logger):
    # CHANGE BEGIN
    local_rank = int(os.environ['LOCAL_RANK'])
    rank = int(os.environ["RANK"])
    # CHANGE END
    dataloader = get_dataloader(config.batch_size, config.seq_length)
    valid_dataloader = get_dataloader(config.batch_size, config.seq_length, split="validation")
    validation_steps = int(1e06 // (config.batch_size * config.seq_length)) # we want to evaluate on 1M tokens
    model = Transformer(config)
    # CHANGE BEGIN
    local_param_count, sharded_param_count = count_fsdp_params(model)
    print(
         f"Rank {rank} | local_rank {local_rank} | "
         f"cuda device {torch.cuda.current_device()} | "
         f"local params {local_param_count} | sharded params {sharded_param_count}",
         flush=True
    )
    # CHANGE END
    # CHANGE BEGIN - FSDP
    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD #SHARD_GRAD_OP
    mp_policy = MixedPrecision(
      param_dtype=torch.bfloat16,
      reduce_dtype=torch.bfloat16,
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
        device_id=device
        # device_id=torch.cuda.current_device()
    )
    # CHANGE END

    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    # CHANGE BEGIN - WSD learning rate
    scheduler = wsd_lr_scheduler(optimizer, config.train_steps)
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
        if i == 0:
            # CHANGE: whole if-statement, check if model has been sharded
            full, shard = count_fsdp_params(model)
            print(
                f"Rank {rank} | full params {full} | local shard {shard}",
                flush=True
            )
            print(
                f"Rank {rank} | sharding strategy:",
                model.sharding_strategy,
                flush=True
            )

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

def setup():
    dist.init_process_group("nccl")

def cleanup():
    dist.barrier()
    dist.destroy_process_group()


def main():
    # Arguments:
    # Default: n_layers=4, dmodel=256, n_heads=4, batch_size=64, n_training_steps=1000, 
    # learning_rate=1e-4, dropout=0.0, sequence_length=256,
    class Range(object):
        def __init__(self, start, end):
            self.start = start
            self.end = end
        def __eq__(self, other):
            return self.start <= other <= self.end
    parser = argparse.ArgumentParser(description="Hyperparameters of transformer")
    parser.add_argument("--n_layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--dmodel", type=int, default=256, help="Dimension of model embedding")
    parser.add_argument("--n_heads", type=int, default=4, help="Number of heads")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--n_training_steps", type=int, default=1000, help="Number of training steps")
    parser.add_argument("--dropout", type=float, default=0.0, choices=[Range(0.,1)], help="Dropout")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--sequence_length", type=int, default=256, help="Number of tokens in sequence")
    args = parser.parse_args()

    config = SimpleNamespace(
        train_steps=args.n_training_steps,
        vocab_size=50257,
        max_len=256,
        d_model=args.dmodel,
        num_heads=args.n_heads,
        num_layers=args.n_layers,
        learning_rate=args.learning_rate,
        dropout=args.dropout,
        seq_length=args.sequence_length,
        batch_size=args.batch_size,
        log_train_loss_freq=100,
        log_valid_loss_freq=100
    )

    local_rank = int(os.environ['LOCAL_RANK'])
    rank = int(os.environ['RANK'])
    # world_size = int(os.environ['WORLD_SIZE'])

    # wandb init, only for node with rank=0:
    run = wandb.init(
        entity="ka-pilkiewic-university-of-warsaw",
        project="lml-assignment2-project",
        config=vars(args),
    ) if rank == 0 else None

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
        print(f"Device type is: {device}. Remember to train on GPU.")

    setup()
    
    train_model(config, device, run)

    cleanup()
    if run is not None:
        run.finish()

if __name__ == "__main__":
    main()

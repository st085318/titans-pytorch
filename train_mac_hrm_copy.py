import random
import tqdm
import gzip
import numpy as np

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from adam_atan2_pytorch import AdoptAtan2

from titans_pytorch import (
    MemoryAsContextTransformer,
    MemoryMLP,
    MemoryAttention,
    HRMMemoryBlock,
    MambaBlock
)

from transformers import AutoTokenizer
from datasets import load_dataset

PATH = '/home/sirius3085/exps/titans/mamba_'

# constants

NUM_BATCHES = int(1e5)
BATCH_SIZE = 4
GRADIENT_ACCUMULATE_EVERY = 4
LEARNING_RATE = 5e-5
VALIDATE_EVERY  = 50
GENERATE_EVERY  = 500
PRIME_LENGTH = 100
GENERATE_LENGTH = 512
SHOULD_GENERATE = True
SEQ_LEN = 512

# neural memory related

NEURAL_MEMORY_DEPTH = 2
NUM_PERSIST_MEM = 4
NUM_LONGTERM_MEM = 4
NEURAL_MEM_LAYERS = (2, 4, 6)                   # layers 2, 4, 6 have neural memory, can add more
NEURAL_MEM_GATE_ATTN_OUTPUT = False
NEURAL_MEM_MOMENTUM = True
NEURAL_MEM_MOMENTUM_ORDER = 1
NEURAL_MEM_QK_NORM = True
NEURAL_MEM_MAX_LR = 1e-1
USE_MEM_ATTENTION_MODEL = False
WINDOW_SIZE = 32
NEURAL_MEM_SEGMENT_LEN = 4                      # set smaller for more granularity for learning rate / momentum etc
NEURAL_MEM_BATCH_SIZE = 128                     # set smaller to update the neural memory weights more often as it traverses the sequence
SLIDING_WINDOWS = True
STORE_ATTN_POOL_CHUNKS = True                   # whether to use attention pooling for chunk derived momentum, per-layer lr mod, decay
MEMORY_MODEL_PER_LAYER_LEARNED_LR = True
NEURAL_MEM_WEIGHT_RESIDUAL = True               # learning to accept contributions from the weights of the previous neural mem layer brings about significant improvements. this was improvised and not in the paper, but inspired by the value residual learning free lunch paper
NEURAL_MEM_QKV_RECEIVES_DIFF_VIEW = True        # will allow the neural memory to select what layers from which to derive queries / keys / values, effectively allowing it to graft itself to the transformer in any way to be beneficial. this is to address an issue from a phd student who noted that the mem network is learning nothing more than wk @ wv. this also generalizes all possible ways to connect the neural memory to a transformer, a sort of NAS
NEURAL_MEM_SPEC_NORM_SURPRISES = True           # applying lessons from Muon optimizer to surprise updates, by spectral norming the surprises

USE_MEM_HRM_MODEL = True

# experiment related

PROJECT_NAME = 'titans-mac-transformer'
RUN_NAME = f'mac - {NUM_LONGTERM_MEM} longterm mems with mamba block, layers {NEURAL_MEM_LAYERS}'
WANDB_ONLINE = False # turn this on to pipe experiment to cloud

# perf related

USE_ACCELERATED_SCAN = True
USE_FLEX_ATTN = True
USE_FAST_INFERENCE = False

# wandb experiment tracker

import wandb
wandb.init(project = PROJECT_NAME, mode = 'offline' if not WANDB_ONLINE else 'online')
wandb.run.name = RUN_NAME
wandb.run.save()

# helpers

def cycle(loader):
    while True:
        for data in loader:
            yield data

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))

torch.cuda.set_device(4)
# memory model

if USE_MEM_ATTENTION_MODEL:
    neural_memory_model = MemoryAttention(
        dim = 64
    )
elif USE_MEM_HRM_MODEL:
        neural_memory_model = MambaBlock(
        dim = 64
    )
else:
    neural_memory_model = MemoryMLP(
        dim = 64,
        depth = NEURAL_MEMORY_DEPTH
    )

print("Getting embeds..")
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
#/home/sirius3085/light_lm/titans-pytorch/

embeds = torch.nn.Embedding.from_pretrained(torch.load("checkpoints/gpt2_complete_embeddings.pt")['token_embeddings'])
# instantiate memory-as-context transformer

model = MemoryAsContextTransformer(
    num_tokens=embeds.num_embeddings,
    dim=embeds.embedding_dim,
    token_emb=embeds,
    depth = 8,
    segment_len = WINDOW_SIZE,
    num_persist_mem_tokens = NUM_PERSIST_MEM,
    num_longterm_mem_tokens = NUM_LONGTERM_MEM,
    neural_memory_layers = NEURAL_MEM_LAYERS,
    neural_memory_segment_len = NEURAL_MEM_SEGMENT_LEN,
    neural_memory_batch_size = NEURAL_MEM_BATCH_SIZE,
    neural_mem_gate_attn_output = NEURAL_MEM_GATE_ATTN_OUTPUT,
    neural_mem_weight_residual = NEURAL_MEM_WEIGHT_RESIDUAL,
    neural_memory_qkv_receives_diff_views = NEURAL_MEM_QKV_RECEIVES_DIFF_VIEW,
    use_flex_attn = USE_FLEX_ATTN,
    sliding_window_attn = SLIDING_WINDOWS,
    neural_memory_model = neural_memory_model,
    neural_memory_kwargs = dict(
        dim_head = 64,
        heads = 4,
        attn_pool_chunks = STORE_ATTN_POOL_CHUNKS,
        qk_rmsnorm = NEURAL_MEM_QK_NORM,
        momentum = NEURAL_MEM_MOMENTUM,
        momentum_order = NEURAL_MEM_MOMENTUM_ORDER,
        default_step_transform_max_lr = NEURAL_MEM_MAX_LR,
        use_accelerated_scan = USE_ACCELERATED_SCAN,
        per_parameter_lr_modulation = MEMORY_MODEL_PER_LAYER_LEARNED_LR,
        spectral_norm_surprises = NEURAL_MEM_SPEC_NORM_SURPRISES
    )
).cuda()

print("model -- got it!")
print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
# prepare fineweb data

print("Loading data...")
# download 10B tokens for sample pretrain
fw = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
fw_train = fw.skip(400)
fw_val = fw.take(400)


class FinewebLoader:
    def __init__(self, data_source, tokenizer, batch_size):
        self.data_iter = iter(data_source)
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.batch_size = batch_size

    def __iter__(self):
        return self
    
    def __next__(self):
        lines = [next(self.data_iter)['text'] for _ in range(self.batch_size)]
        tokens = self.tokenizer(lines, padding=True, truncation=True, max_length=512, return_tensors='pt')['input_ids']
        return tokens.cuda()


train_loader  = cycle(FinewebLoader(fw_train, tokenizer, batch_size=BATCH_SIZE))
val_loader    = FinewebLoader(fw_val, tokenizer, batch_size=BATCH_SIZE)

print("data -- no problem!")

# optimizer

optim = AdoptAtan2(model.parameters(), lr = LEARNING_RATE)

# training

#wandb.watch(model, log="all", log_freq=100)

best_val_loss = np.inf

model.train()
for i in tqdm.tqdm(range(NUM_BATCHES), mininterval = 10., desc = 'training'):
    train_loss = 0
    for __ in range(GRADIENT_ACCUMULATE_EVERY):
        tl = next(train_loader)
        #if i == 1:
        #    print(tokenizer.decode(tl))
        loss = model(tl, return_loss = True)
        loss.backward()
        train_loss += loss.item()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    train_loss /= GRADIENT_ACCUMULATE_EVERY

    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.3)
    optim.step()
    optim.zero_grad()
    wandb.log(dict(train_loss = train_loss))

    if i % VALIDATE_EVERY == 0:
        print(f'training loss: {train_loss}')

        val_loader = FinewebLoader(fw_val, tokenizer, batch_size=BATCH_SIZE)
        torch.save(model.state_dict(), PATH + "/mac_fw_mem3_last")

        model.eval()
        with torch.no_grad():
            val_loss, cnt = 0, 0
            for batch in tqdm.tqdm(val_loader):
                loss = model(batch, return_loss=True)
                val_loss += loss.item()
                cnt += 1
            val_loss /= cnt
            wandb.log(dict(validation_loss = val_loss))
            print(f'validation loss: {val_loss}')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), PATH + "/mac_fw_mem3_best")

    #if SHOULD_GENERATE and i % GENERATE_EVERY == 0:
    #    val_loader = FinewebLoader(fw_val, tokenizer, batch_size=BATCH_SIZE)
    #    model.eval()
    #    inp = next(iter(val_loader))[0, :PRIME_LENGTH]
    #    prime = tokenizer.decode(inp)
    #    print(f'%s \n\n %s', (prime, '*' * 100))

    #    sample = model.sample(inp[None, ...], GENERATE_LENGTH, use_cache = USE_FAST_INFERENCE)
    #    output_str = tokenizer.decode(sample[0])
    #    print(output_str)

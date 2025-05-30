"""
Sample from the trained model with PyTorch
"""
import os
import pickle
from contextlib import nullcontext
import torch
from threepart import ModelArgs, Transformer
from tokenizer import Tokenizer

from tinystories import get_tokenizer_model_path

# -----------------------------------------------------------------------------
checkpoint = 'out/relaxed2025_05_17_10_00_18.pt'
start = "" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 1024 # number of samples to draw
batch_size = 256
max_new_tokens = 196 # number of tokens generated in each sample
temperature = 1.0 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 300 # retain only the top_k most likely tokens, clamp others to have 0 probability
tokenizer = "data/tok4096.model" # override the tokenizer model path
seed = 1337
device = 'cuda' if torch.cuda.is_available() else 'cpu' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
#dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
dtype = "bfloat16"
compile = True # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file
    # -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# init from a model saved in a specific directory
checkpoint_dict = torch.load(checkpoint, map_location=device)
gptconf = ModelArgs(**checkpoint_dict['model_args'])
model = Transformer(gptconf)
state_dict = checkpoint_dict['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict, strict=False)

model.eval()
model.to(device)
if compile:
    print("Compiling the model...")
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# load the tokenizer
vocab_source = checkpoint_dict["config"].get("vocab_source", "llama2")
vocab_size = gptconf.vocab_size
if tokenizer:
    # a specific tokenizer is provided, use it
    tokenizer_model = tokenizer
else:
    # let's try to find the tokenizer model automatically. bit gross here...
    query_vocab_size = 0 if vocab_source == "llama2" else vocab_size
    tokenizer_model = get_tokenizer_model_path(vocab_size=query_vocab_size)
enc = Tokenizer(tokenizer_model=tokenizer_model)

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
for _ in range(num_samples//batch_size):
    start_ids = enc.encode(start, bos=True, eos=False)
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
    x_batch = x.repeat(batch_size, 1)

    # If you have different starting prompts (e.g., from a file):
    # You'd need to handle padding if prompts have different lengths.
    # This is more complex and not shown here for brevity.

    with torch.no_grad():
        with ctx:
            # Generate all sequences in a batch
            y_batch = model.generate(x_batch, max_new_tokens, temperature=temperature, top_k=top_k)

            # Process each generated sequence in the batch
            for k in range(batch_size):
                print(enc.decode(y_batch[k].tolist()))
                print('---------------')


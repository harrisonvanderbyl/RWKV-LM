########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import gc, math, os
from typing import List, Optional

import numpy as np
import torch
# torch._C._jit_set_profiling_executor(True)
# torch._C._jit_set_profiling_mode(True)
import torch.nn as nn
from torch.nn import functional as F


import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
import deepspeed.runtime.lr_schedules
import wandb

from torch.utils.cpp_extension import load

# Script dir for various files
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
CUDA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../cuda"))

########################################################################################################
# JIT / torch compile special handling
########################################################################################################

# Currently the features we need for torch compile, is avaliable only in
# 2.1 nightly build (and is expected to be in 2.1 official release)
#
# We only enable torch compile, if we either the user has explicitly
# set it, or we detect that we are running on a 2.1+ build automatically
from packaging import version
def is_torch_version_above(required_version):
    torch_version = version.parse(torch.__version__.split('+')[0])
    return torch_version >= version.parse(required_version)
IS_TORCH_2_1 = is_torch_version_above("2.0.9999")

# Get the JIT / torch compile option flags from the environment
RWKV_JIT_ON        = os.getenv("RWKV_JIT_ON", "1").lower() in ("1", "true", "yes")
RWKV_TORCH_COMPILE = os.getenv("RWKV_TORCH_COMPILE", f"{IS_TORCH_2_1}").lower() in ("1", "true", "yes")
RWKV_TORCH_RUN_MODE = None

# We enable JITMod*/Function when supporting torch.jit
# We use TorchCompile* when supporting torch compile

RWKV_TORCH_RUN_MODE = "torch-native"
JITModClass  = nn.Module
JITModMethod = lambda x: x
JITFunction  = lambda x: x

TCompileMax        = lambda x: x
TCompileBaseline   = lambda x: x
TCompileDisable    = lambda x: x

print(f"[RWKV.model] Running RWKV model using '{RWKV_TORCH_RUN_MODE}' with torch '{torch.__version__}'")


def wkv_mop(time_decay, time_first, k, v, wkv_state):
    u = time_first.double()
    w = time_decay.double().exp().neg()
    # print k hasnan
    kk = torch.exp(k.double())
    vv = v.double()
    wr1 = wkv_state[0] +  torch.exp(u+w+k) * vv
    wr2 = wkv_state[1]  +  torch.exp(u+w+k)
    # print("wr1[0]", wkv_state[0][0])
    # print("wr2[0]", wkv_state[1][0])
    y = wr1 / wr2
    wkv_state[0] = ((wkv_state[0] + kk*vv) * torch.exp(w)).float()
    wkv_state[1] = ((wkv_state[1]  + kk) * torch.exp(w)).float()
    return y.to(k.dtype), wkv_state

########################################################################################################
# RWKV: State Blocks
########################################################################################################

class TimeMixState:

    def __init__(self, shift_state: torch.Tensor, wkv_state: torch.Tensor):
        self.shift_state = shift_state
        self.wkv_state = wkv_state


class ChannelMixState:

    def __init__(self, shift_state: torch.Tensor):
        self.shift_state = shift_state


class BlockState:

    def __init__(self, time_mix_state: TimeMixState,
                 channel_mix_state: ChannelMixState):
        self.time_mix_state = time_mix_state
        self.channel_mix_state = channel_mix_state


class BlockStateList:

    def __init__(self, att_shift_states, ffn_shift_states, wkv_states):
        self.wkv_states = wkv_states
        self.att_shift_states = att_shift_states
        self.ffn_shift_states = ffn_shift_states
    

    def create(N, B, C, device, dtype):
        result = BlockStateList.empty(N, B, C, device, dtype)
        result.wkv_states[:] = 0
        # result.wkv_states[:, :, :, -1] = -1e38

        # result.att_shift_states[:] = 0
        result.ffn_shift_states[:] = 0

        return result

    def empty(N, B, C, device, dtype):
        wkv_states = torch.zeros((N, 3, C),
                                 device=device,
                                 dtype=torch.float)
        
        # HOT FIX 2**12, 12 should = layer count
        att_shift_states = [[] for _ in range(N)]
        ffn_shift_states = torch.zeros((N, 1, C), device=device, dtype=dtype)
        return BlockStateList(att_shift_states, ffn_shift_states, wkv_states)

    def __getitem__(self, layer: int):
        return BlockState(
            TimeMixState(self.att_shift_states[layer], self.wkv_states[layer]),
            ChannelMixState(self.ffn_shift_states[layer]))

    def __setitem__(self, layer: int, state: BlockState):
        # HOT FIX 2**12, 12 should = layer count
        # print("[__setitem__] layer", layer)
        # print("[__setitem__] state.time_mix_state.shift_state", state.time_mix_state.shift_state.shape)
        # print("[__setitem__] self.att_shift_states[layer]", self.att_shift_states[layer].shape)

        self.att_shift_states[layer] = state.time_mix_state.shift_state[-(2**layer):]
        self.wkv_states[layer] = state.time_mix_state.wkv_state
        self.ffn_shift_states[layer] = state.channel_mix_state.shift_state

########################################################################################################
# RWKV: RWKV Time-mix + RWKV Channel-mix
########################################################################################################

class RWKV_TimeMix(JITModClass):

    def __init__(self, layer_id, n_layer, n_embd, dim_att, load_model, float_mode, device):
        super().__init__()


     

        # self.key = nn.Linear(n_embd, dim_att, bias=False)
        # self.value = nn.Linear(n_embd, dim_att, bias=False)
        # self.receptance = nn.Linear(n_embd, dim_att, bias=False)
        # self.output = nn.Linear(dim_att, n_embd, bias=False)

        self.shiftamount = pow(2,layer_id)
        # self.time_shift = nn.ZeroPad2d((0, 0, shiftamount, -shiftamount))
        self.time_mix_k = load_model[f"blocks.{layer_id}.att.time_mix_k"].squeeze().to(dtype=float_mode, device=device)
        self.time_mix_v = load_model[f"blocks.{layer_id}.att.time_mix_v"].squeeze().to(dtype=float_mode, device=device)
        self.time_mix_r = load_model[f"blocks.{layer_id}.att.time_mix_r"].squeeze().to(dtype=float_mode, device=device)
        self.time_decay = load_model[f"blocks.{layer_id}.att.time_decay"].squeeze().to(device)
        self.time_first = load_model[f"blocks.{layer_id}.att.time_first"].squeeze().to(device)
        self.key = load_model[f"blocks.{layer_id}.att.key.weight"].to(dtype=float_mode, device=device)
        self.value = load_model[f"blocks.{layer_id}.att.value.weight"].to(dtype=float_mode, device=device)
        self.receptance = load_model[f"blocks.{layer_id}.att.receptance.weight"].to(dtype=float_mode, device=device)
        self.output = load_model[f"blocks.{layer_id}.att.output.weight"].to(dtype=float_mode, device=device)
        self.zz = torch.zeros_like(self.time_mix_k)

 
    def _forward_kvsr(self, x, last_state: TimeMixState):

        # Print the various shapes for debugging
        # print("")
        # print("x shape: ", x.shape) # eg. [1, 2909, 2560]
        # print("last_state.wkv_state: ", last_state.wkv_state.shape) # eg. [1, 2560, 3]
        # print("last_state.shift_state: ", last_state.shift_state.shape) # eg. [4096, 1, 2560]
        # print("last_state.shift_state.unsqueeze(0): ", last_state.shift_state.unsqueeze(0).shape) # eg. [1, 1, 2560]
        
        xxx = last_state.shift_state + [x.clone()]
        
        if(len(xxx) < self.shiftamount):
            xx = self.zz
        else:
            xx = xxx[-self.shiftamount]

        # print("xx[0]", xx[0])
        # print("x[0]", x[0])
        # xx = xxxx.view(x.shape[0], x.shape[1], x.shape[2])
        # print( "xxxx shape: ", xxxx.shape) # eg. [1, 2582, 2560]
        # print( "xxx shape: ", xxx.shape) # eg. [1, 2582, 2560]
        # print( "xx.shape: ", xx.shape) # eg. [1, 2582, 2560]
        # print( "x.shape: ", x.shape) # eg. [1, 2488, 2560]
        # print( "self.time_mix_k.shape: ", self.time_mix_k.shape)

        # last_state.shift_state
        # xx

        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)

        k = self.key @ (xk)
        v = self.value @ (xv)
        r = self.receptance @ (xr)


        sr = torch.sigmoid(r)

        # Enforce bf16 type for kv, as this can be mis init
        # when being called directly via inference
        

        return k, v, sr, xxx

    def _forward_out(self, sr, y, x_l, new_wkv_state):
        return self.output @ (sr * y), TimeMixState(x_l, new_wkv_state)

    def forward(self, x, last_state: TimeMixState):
        # Enforce bf16 for self.time_first
        # as this can be mis init when being called directly via inference
       
        # Perform the WKV op via cuda code
        k, v, sr, xxx = self._forward_kvsr(x, last_state)
        # print("k[0]", k[0])
        # print("v[0]", v[0])
        y, new_wkv_state = wkv_mop(self.time_decay, self.time_first,
                                  k, v, last_state.wkv_state)
        
        # print("y[0]", y[0])
        
        return self._forward_out(sr, y, xxx, new_wkv_state)


########################################################################################################


class RWKV_ChannelMix(JITModClass):

    def __init__(self, layer_id, n_layer, n_embd, dim_ffn, load_model, float_mode, device):
        super().__init__()


      
        self.time_mix_k = load_model[f"blocks.{layer_id}.ffn.time_mix_k"].squeeze().to(dtype=float_mode, device=device)
        self.time_mix_r = load_model[f"blocks.{layer_id}.ffn.time_mix_r"].squeeze().to(dtype=float_mode, device=device)
        self.key = load_model[f"blocks.{layer_id}.ffn.key.weight"].to(dtype=float_mode, device=device)
        self.receptance = load_model[f"blocks.{layer_id}.ffn.receptance.weight"].to(dtype=float_mode, device=device)
        self.value = load_model[f"blocks.{layer_id}.ffn.value.weight"].to(dtype=float_mode, device=device)
        

    def forward(self, x, last_state: ChannelMixState):
        xx = last_state.shift_state[-1]
        
        # print("[CM] x shape: ", x.shape)
        # print("[CM] last_state.shift_state shape: ", last_state.shift_state.shape)
        # print("[CM] xx shape: ", xx.shape)
        # print("[CM] self.time_mix_k shape: ", self.time_mix_k.shape)
        # print("[CM] self.time_mix_r shape: ", self.time_mix_r.shape)

        xk = (x * self.time_mix_k + xx * (1 - self.time_mix_k))
        xr = (x * self.time_mix_r + xx * (1 - self.time_mix_r))
        k =  self.key @ (xk)
        k = torch.square(torch.relu(k))
        kv = self.value @ k
        return (torch.sigmoid(self.receptance @ (xr)) * kv,
                ChannelMixState(x.unsqueeze(0)))


########################################################################################################
# The RWKV Model blocks
########################################################################################################

class Block(nn.Module):

    def __init__(self, layer_id, n_layer, n_embd, dim_att, dim_ffn, load_model, float_mode, device):
        super().__init__()
        self.layer_id = layer_id


        if self.layer_id == 0:
            self.ln0weight = (load_model[f"blocks.{layer_id}.ln0.weight"]).to(dtype=float_mode, device=device)
            self.ln0bias = (load_model[f"blocks.{layer_id}.ln0.bias"]).to(dtype=float_mode, device=device)

        self.att = RWKV_TimeMix(layer_id, n_layer, n_embd, dim_att,load_model, float_mode, device)
        self.ffn = RWKV_ChannelMix(layer_id, n_layer, n_embd, dim_ffn,load_model, float_mode, device)

        self.ln1weight = (load_model[f"blocks.{layer_id}.ln1.weight"]).to(dtype=float_mode, device=device)
        self.ln1bias = (load_model[f"blocks.{layer_id}.ln1.bias"]).to(dtype=float_mode, device=device)
        self.ln2weight = (load_model[f"blocks.{layer_id}.ln2.weight"]).to(dtype=float_mode, device=device)
        self.ln2bias = (load_model[f"blocks.{layer_id}.ln2.bias"]).to(dtype=float_mode, device=device)

    def ln0(self, x):
        return F.layer_norm(x,(x.shape[-1],), self.ln0weight, self.ln0bias).squeeze()
    
    def ln1(self, x):
        return F.layer_norm(x,(x.shape[-1],), self.ln1weight, self.ln1bias).squeeze()
    
    def ln2(self, x):
        return F.layer_norm(x,(x.shape[-1],), self.ln2weight, self.ln2bias).squeeze()

    def forward(self, x:torch.Tensor, last_state: BlockState):
        # print has nan
        # print("[Block] x has nan: ", torch.isnan(x).any())
        # print(x[0])
        

        if self.layer_id == 0:
            x = self.ln0(x)
            # print(x[0])

        rx = self.ln1(x)
        # print(rx[0])

        att_out, att_state = self.att(
            rx,
            last_state.time_mix_state,
        )
        x = x + att_out
        # print(x[0])
        ffn_out, ffn_state = self.ffn(
            self.ln2(x),
            last_state.channel_mix_state,
        )

        x = x + ffn_out
        # print(x[0])

        return x, BlockState(att_state, ffn_state)




########################################################################################################
# Static optimized functions
########################################################################################################

# @ TCompileMax (no speed improvement)
# def F_cross_entropy_reduction_none_optimized(logits, targets):
#     return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction="none")

########################################################################################################
# Core RWKV module
########################################################################################################
class RWKV(nn.Module):

    def __init__(self,
                 ctx_len: int,
                 n_embd: int,
                 n_layer: int,
                 vocab_size: int,
                 # Model file path to load from
                 load_model,
                 # Context length schedule
                 ctx_len_cutoffs: List[int] = [],
                 ctx_len_warmup_steps: List[int] = [],
                 # Alternative to lr_init / lr_final
                 # that is multiplied by the gradient_accumulation_steps
                 # to get the actual learning rate
                 target_lr_init: float = -1.0,
                 target_lr_final: float = -1.0,
                 # Learning rate schedule
                 # use only target_lr_init / lr_init
                 # to configure a constant learning rate
                 lr_init: float = -1.0,
                 lr_final: float = -1.0,
                 lr_period: int = -1,
                 lr_period_type: str = 'epoch',
                 # Adam optimizer settings
                 beta1: float = 0.9,
                 beta2: float = 0.99,
                 adam_eps: float = 1.0e-08,
                 weight_decay: float = 0.01,
                 warmup_steps: int = -1,
                 # Backprop settings
                 grad_cp: bool = True,
                 bptt_learning: bool = True,
                 bptt_learning_range: int = -1,
                 bptt_truncated_learning: bool = False,
                 layerwise_lr: bool = True,
                 dim_att: Optional[int] = None,
                 dim_ffn: Optional[int] = None,
                 substep_cuda_cache_clear: bool = False,
                 substep_logging: bool = False,
                 torch_set_float32_matmul_precision:str = 'high'
                 ):

        # Lets save everything in one shot
        # (this is used for wandb logging)
        self.setup_args = locals()
        del self.setup_args["self"]
        del self.setup_args["__class__"]

        # Setup the model
        super().__init__()

        # Save the various other params for later
        self.ctx_len = ctx_len
        self.ctx_len_cutoffs = ctx_len_cutoffs
        self.ctx_len_warmup_steps = ctx_len_warmup_steps
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.layerwise_lr = layerwise_lr
        self.grad_cp = grad_cp
        self.target_lr_init = target_lr_init
        self.target_lr_final = target_lr_final
        self.lr_init = lr_init
        self.lr_final = lr_final
        self.lr_period = lr_period
        self.lr_period_type = lr_period_type
        self.warmup_steps = warmup_steps
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay
        self.adam_eps = adam_eps
        self.bptt_learning = bptt_learning
        self.bptt_learning_range = bptt_learning_range
        self.bptt_truncated_learning = bptt_truncated_learning
        self.substep_cuda_cache_clear = substep_cuda_cache_clear
        self.substep_logging = substep_logging

        dim_att = dim_att or n_embd
        dim_ffn = dim_ffn or n_embd * 4

        if torch_set_float32_matmul_precision is not None:
            torch.set_float32_matmul_precision(torch_set_float32_matmul_precision)


        self.float_mode = torch.bfloat16
        self.device = torch.device("cuda")

        self.blocks = nn.ModuleList([
            Block(i, n_layer, n_embd, dim_att, dim_ffn, load_model, self.float_mode, self.device) for i in range(n_layer)
        ])


        self.ln_outbias = (load_model["ln_out.bias"]).to(dtype=self.float_mode,device= self.device)
        self.ln_outweight = (load_model["ln_out.weight"]).to(dtype=self.float_mode,device= self.device)

        self.head = load_model["head.weight"].to(dtype=self.float_mode,device= self.device)
        self.emb = load_model["emb.weight"].to(self.float_mode)

    def ln_out(self, x):
        return F.layer_norm(x,(x.shape[-1],), self.ln_outweight, self.ln_outbias).squeeze()
    


        

 

    def forward(self, idx: torch.Tensor, last_att_shift_states: torch.Tensor, last_ffn_shift_states: torch.Tensor,
                last_wkv_states: torch.Tensor):
  
        x = self.emb[idx].squeeze().to(self.device)

        new_states = BlockStateList.empty(self.n_layer, 1, self.n_embd,
                                          x.device, x.dtype)
        
        if last_att_shift_states is None:
            cur_bs_list = BlockStateList.empty(
                self.n_layer, 1,
                self.n_embd,
                x.device, x.dtype
            )
        else:
            cur_bs_list = BlockStateList(last_att_shift_states,last_ffn_shift_states, last_wkv_states)


        for i in range(len(self.blocks)):
            block = self.blocks[i]
            last_state = cur_bs_list[i]
            x, new_state = block(x, last_state)
            new_states[i] = new_state

        x = self.ln_out(x)

        x = self.head @ x

        return x, new_states.att_shift_states, new_states.ffn_shift_states, new_states.wkv_states

  


########################################################################################################
# SimpleRWKV, a wrapper for RWKV that allows for simple usage of the model
########################################################################################################

# SimpleRWKV specific imports
from transformers import PreTrainedTokenizerFast

# Current script dir
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
SCRIPT_PARENT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '../'))

# SimpleRWKV is a wrapper for RWKV that allows for simple usage of the model
#
# it is not meant to be highly performant, but rather a simple minimal way to run the RWKV trainer module
# in inference mode, and can be used to validate the model trainer code / its changes
class SimpleRWKV():

    def __init__(
            self,
            model_path: str,
            ctx_len:int = 1024,
            device:str = "cuda",
            dtype:str = "fp32",
            tokenizer = "pile",
        ):

        # Device type must be cuda, cpu type is not supported (yet?)
        if device != "cuda":
            raise NotImplementedError("Only cuda device is supported (for now)")

        # Setup the tokenizer
        if tokenizer == "pile":
            tokenizer_file = os.path.join(SCRIPT_PARENT_DIR,"20B_tokenizer.json")
            tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)
            vocab_size = 50277
        else:
            raise NotImplementedError("Only pile tokenizer is supported")
        self.fastTokenizer = tokenizer

        # Load the model, yes i know this is a double load 
        # but the goal of SimpleRWKV is not to optimize init, but to "just work"
        _torch_load = torch.load(model_path, map_location="cpu")

        # Get the model params
        keys = list(_torch_load.keys())

        # Get the maximum block id
        max_block_id = 0
        for x in keys:
            if 'blocks.' in x:
                block_id = int(x.split('.')[1])
                max_block_id = max(max_block_id, block_id)
        
        # Compute the layer count, embed sizes, and vocab size
        n_layer = max_block_id + 1
        n_embd = _torch_load['head.weight'].shape[1]
        vocab_size = max(_torch_load['head.weight'].shape[0], vocab_size)

      

        # Prepare the model config with the model path, and custom torch load
        model_config = {}
        model_config["load_model"] = _torch_load
        model_config["n_embd"] = n_embd 
        model_config["n_layer"] = n_layer 
        model_config["vocab_size"] = vocab_size 
        model_config["ctx_len"] = ctx_len

        # This feature depends on deepspeed
        model_config["grad_cp"] = False
        # model_config["_torch_load_state"] = loaded_state

        # Save the config settings
        self.ctx_len = ctx_len
        self.device = device

        # Lets actually load the model
        self.model = RWKV(**model_config)

        # Lets map it over to the respective device type
        # and set it to run as eval/inference mode
        self.model.to(device)
        self.device = device
        self.model.eval()

    # Encoding strings
    def encode(self, text: str):
        return self.fastTokenizer.encode(text)

    # Decoding strings
    def decode(self, tokens: list):
        return self.fastTokenizer.decode(tokens)

    # Forwarding logic, withoout torch._no_grad() context
    def _forward(
            self, tokens:list, 
            stateObj = None
        ):

        logits_arr = None
        token_len = len(tokens)

        # Get the shift/wkv state
        if stateObj is None:
            att_shift_states = None
            ffn_shift_states = None
            wkv_states = None
        else:
            att_shift_states = stateObj["att_shift_states"]
            ffn_shift_states = stateObj["ffn_shift_states"]
            wkv_states = stateObj["wkv_states"]
        
        # For each token, process the state, in batches up to ctx_len
        for i in range(0, token_len, self.ctx_len):
            # Get the tokens for this batch
            batch_tokens = torch.tensor(
                [tokens[i:i+self.ctx_len]], 
                dtype=torch.long, 
                device="cpu"
            )

            # Compute the logits and state
            logits_arr, att_shift_states, ffn_shift_states, wkv_states = self.model.forward(
                batch_tokens, att_shift_states, ffn_shift_states, wkv_states
            )

        # Return the logits and state
        return logits_arr, { "att_shift_states": att_shift_states, "ffn_shift_states": ffn_shift_states, "wkv_states": wkv_states }
    
    # Forwarding logic, with torch._no_grad() context
    def forward(
            self, tokens:list, 
            stateObj = None
        ):
        with torch.no_grad():
            return self._forward(tokens, stateObj)

    # Sampling logits
    def sample_logits(
            self, logits, 
            prv_tokens=[0], 
            temperature=1.0, top_p=0.9,
            token_ban: list = []
            ):
        # Apply token ban
        for x in token_ban:
            logits[x] = -float("Inf")

        # Handle sampling with temperature
        if temperature > 0.0:
            probs = F.softmax(logits, dim=-1)
            sorted_probs = torch.sort(probs, descending=True)[0]
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1).cpu().numpy()
            cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])
            probs[probs < cutoff] = 0
            if temperature != 1.0:
                probs = probs.pow(1.0 / temperature)
            out = torch.multinomial(probs, num_samples=1)[0]
            return out
        else: 
            # Since the tokenizer sample does not support temp==0
            # we handle this case ourself, by fining the top token
            return torch.argmax(logits, dim=-1).item()

    # Completion API
    def completion(self, 
            prompt, 
            max_tokens: int = 32,
            temperature: float = 1.0,
            top_p: float = 0.9,
            token_ban: list = [],
            start_state = None,
            stream_to_stdout: bool = False,
        ):
        # Encode the context, if its a string
        if isinstance(prompt, str):
            enc = self.encode(prompt)
        # Check if the prompt is a list of tokens
        elif isinstance(prompt, list):
            enc = prompt
        else:
            raise ValueError("Prompt must be a string or a list of tokens")

        # Keep track of the logits and state
        logits = None
        stateObj = start_state

        # For each token, process the state
        logits, stateObj = self.forward(enc, stateObj)

        # # Garbage collect
        # gc.collect()
        # torch.cuda.empty_cache()

        # Generate each token
        full_tokens = enc.copy()
        out_tokens = []
        for i in range(max_tokens):
            ttt = self.sample_logits(
                logits, prv_tokens=full_tokens,
                temperature=temperature, top_p=top_p,
                token_ban=token_ban
            )
            
            # Append the token
            out_tokens.append(ttt)
            full_tokens.append(ttt)
            if stream_to_stdout:
                print(self.decode([ttt]), end="", flush=True)

            # Perform the forward pass
            logits, stateObj = self.forward([ttt], stateObj)

        # Decode the tokens
        out_str = self.decode(out_tokens)

        # # Garbage collect
        # gc.collect()
        # torch.cuda.empty_cache()

        # Return the output string, and state
        return out_str, stateObj

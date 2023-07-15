import math, os, sys, types, time, gc, torch

# Just enable it, keep it simple, this class was not meant for perf
os.environ["RWKV_JIT_ON"] = "1"
os.environ["RWKV_RUN_DEVICE"] = "cpu"

# Import the RWKV model
from .model_run import RWKV_RNN
from .utils import TOKENIZER

#
# SimpleRWKV wrapper, used to provide a simple inference interfcate to the RWKV model
#
# It maybe lacking lots of the features you find in a more complete wrapper.
# but it goes the job done for simple testing / eval
#
class SimpleRWKV:

    def __init__(
            self,
            model_path: str,
            device:str = "cpu",
            dtype:str = "fp32",
            tokenizer = "pile",
        ):

        # Setup the tokenizer
        if tokenizer == "pile":
            tokenizer = TOKENIZER(
                ["20B_tokenizer.json","20B_tokenizer.json"],
                UNKNOWN_CHAR = None
            )
            vocab_size = 50277
        else:
            raise NotImplementedError("Only pile tokenizer is supported")
        self.tokenizerCtx = tokenizer

        # Configure the respective run device mode
        if device == "cuda":
            os.environ["RWKV_RUN_DEVICE"] = "cuda"

        # Load the model
        _torch_load = torch.load(model_path, map_location="cpu")

        # Get the model params
        keys = list(_torch_load.keys())

        # Get the maximum block id
        max_block_id = 0
        for x in keys:
            if 'blocks.' in x:
                block_id = int(x.split('.')[1])
                max_block_id = max(max_block_id, block_id)
        
        # Compute the layer count & embed sizes
        n_layer = max_block_id + 1
        n_embd = _torch_load['head.weight'].shape[1]

        # Setup the model
        args = types.SimpleNamespace()
        args.n_layer = n_layer
        args.n_embd = n_embd
        args.ctx_len = 1 # Since we are doing a dumb RNN ctx_len is 1
        args.vocab_size = vocab_size
        args.head_qk = 0
        args.pre_ffn = 0
        args.grad_cp = 0
        args.my_pos_emb = 0
        args.RUN_DEVICE = device
        args.FLOAT_MODE = dtype
        self.model = RWKV_RNN(args, _torch_load=_torch_load)

    # Completion API
    def completion(self, 
            prompt: str, 
            max_tokens: int = 32,
            temperature: float = 1.0,
            top_p: float = 0.9,
            start_state = None,
            stream_to_stdout: bool = False,
        ):
        # Encode the context
        enc = self.tokenizerCtx.tokenizer.encode(prompt)
        enc_len = len(enc)

        # Keep track of the logits and state
        logits = None
        state = start_state

        # For each token, process the state
        for i in range(enc_len):
            logits, state = self.model([enc[i]], state)    
            if stream_to_stdout:
                print(self.tokenizerCtx.tokenizer.decode([enc[i]]), end="", flush=True)    

        # Garbage collect
        gc.collect()
        torch.cuda.empty_cache()

        # Generate each token
        full_tokens = enc.copy()
        out_tokens = []
        for i in range(max_tokens):
            if temperature > 0.0:
                ttt = self.tokenizerCtx.sample_logits(
                    logits, full_tokens,
                    None, # this was ctx_len, but the var is not in use
                    temperature=temperature, top_p_usual=top_p, top_p_newline=top_p
                )
            else: 
                # Since the tokenizer sample does not support temp==0
                # we handle this case ourself, by fining the top token
                ttt = torch.argmax(logits, dim=-1).item()
            
            # Append the token
            out_tokens.append(ttt)
            full_tokens.append(ttt)
            if stream_to_stdout:
                print(self.tokenizerCtx.tokenizer.decode([ttt]), end="", flush=True)

            # Perform the forward pass
            logits, state = self.model([ttt], state)

        # Decode the tokens
        out_str = self.tokenizerCtx.tokenizer.decode(out_tokens)

        # Garbage collect
        gc.collect()
        torch.cuda.empty_cache()

        # Return the output string, and state
        return out_str, state

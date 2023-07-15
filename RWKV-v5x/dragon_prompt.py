#!/usr/bin/env python3
from src.SimpleRWKV import SimpleRWKV
import sys

#
# Check if this file was executed directly
#
if __name__ == "__main__":

    # Check for a model path
    if len(sys.argv) < 2:
        print("Usage: python SimpleRWKV.py <model_path>")
        exit()

    # Get model path
    model_path = sys.argv[1]

    # Create the model
    model = SimpleRWKV(model_path)

    # Run the model
    dragon_prompt = "\nIn a shocking finding, scientist discovered a herd of dragons living in a remote, previously unexplored valley, in Tibet. Even more surprising to the researchers was the fact that the dragons spoke perfect Chinese."
    print("Prompt: ") 
    print("", end="")
    out_str = model.completion(dragon_prompt, max_tokens=200, temperature=1.0, top_p=0.9, stream_to_stdout=True)
    # print(out_str)
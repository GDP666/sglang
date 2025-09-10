"""
This example demonstrates how to provide tokenized ids to LLM as input instead of text prompt, i.e. a token-in-token-out workflow.
"""
import sys
import os
import subprocess

# 在导入sglang之前设置Python路径
# sglang_python_path = "/home/guopeng29/sglang052/bin/python3"
# if sglang_python_path not in sys.path:
#     sys.path.insert(0, sglang_python_path)
# 检查当前是否在目标Python环境中
target_python = "/home/guopeng29/sglang052/bin/python3"
if sys.executable != target_python and os.path.exists(target_python):
    print(f"Switching to target Python: {target_python}")
    # 使用目标Python重新运行当前脚本
    result = subprocess.run([target_python] + sys.argv)
    sys.exit(result.returncode)

import sglang as sgl
from sglang.srt.hf_transformers_utils import get_tokenizer

# MODEL_PATH = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_PATH = "/mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/meta-llama/Meta-Llama-3.1-70B-Instruct/main"

def main():
    # Sample prompts.
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    # Create a sampling params object.
    sampling_params = {"temperature": 0.8, "top_p": 0.95}

    # Tokenize inputs
    tokenizer = get_tokenizer(MODEL_PATH)
    token_ids_list = [tokenizer.encode(prompt) for prompt in prompts]

    # Create an LLM.
    llm = sgl.Engine(model_path=MODEL_PATH,
                     skip_tokenizer_init=True,
                     tp_size=4,
                     disable_cuda_graph=True,
                     disable_radix_cache=True)

    outputs = llm.generate(input_ids=token_ids_list, sampling_params=sampling_params)
    # Print the outputs.
    for prompt, output in zip(prompts, outputs):
        decode_output = tokenizer.decode(output["output_ids"])
        print("===============================")
        print(
            f"Prompt: {prompt}\nGenerated token ids: {output['output_ids']}\nGenerated text: {decode_output}"
        )
        print()


# The __main__ condition is necessary here because we use "spawn" to create subprocesses
# Spawn starts a fresh program every time, if there is no __main__, it will run into infinite loop to keep spawning processes from sgl.Engine
if __name__ == "__main__":
    main()

import argparse
from vllm import LLM, SamplingParams
import json
import time

sumtime = 0
sumalltoken = 0
sumouttoken = 0


def benchmark_throughput(model_path, max_model_len, dataset_path, num_prompts, trust_remote_code=False):
    # Load the dataset
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    # Select the first `num_prompts` prompts from the dataset
    prompts = dataset[:num_prompts]

    # Create a sampling params object
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=max_model_len)

    # Create an LLM instance with max_model_len parameter
    llm = LLM(
        model=model_path,
        tokenizer="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        trust_remote_code=trust_remote_code,
        max_model_len=max_model_len  # Add this line
    )

    # Start the benchmark
    start_time = time.time()

    # Generate texts from the prompts
    outputs = []
    for prompt in prompts:
        conversation = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt}
        ]
        output = llm.chat(conversation, sampling_params)
        outputs.append(output)
    elapsed_time = time.time() - start_time

    # Calculate throughput (prompts per second)
    throughput = num_prompts / elapsed_time

    # Print the results
    print(f"Total prompts processed: {num_prompts}")
    print(f"Total time elapsed: {elapsed_time:.2f} seconds")
    print(f"Throughput: {throughput:.2f} prompts/second")

    # Optionally, print the generated texts
    for i, output in enumerate(outputs):
        print(f"Prompt {i + 1}: {prompts[i]!r}")
        print(f"Generated text: {output.outputs[0].text!r}")
        print("-" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark throughput for a given model and dataset.")
    parser.add_argument("--model", type=str, required=True, help="Path to the model file.")
    parser.add_argument("--max_model_len", type=int, required=True, help="Maximum model length.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset file.")
    parser.add_argument("--num_prompts", type=int, required=True, help="Number of prompts to process.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Trust remote code.")

    args = parser.parse_args()

    benchmark_throughput(
        model_path=args.model,
        max_model_len=args.max_model_len,
        dataset_path=args.dataset,
        num_prompts=args.num_prompts,
        trust_remote_code=args.trust_remote_code
    )

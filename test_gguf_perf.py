import argparse
from vllm import LLM, SamplingParams
import json
import time
import numpy as np

def benchmark_throughput(model_path, max_model_len, dataset_path, num_prompts, trust_remote_code=False):
    # Load dataset
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    prompts = dataset[:num_prompts]

    # Initialize model
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=max_model_len)
    llm = LLM(
        model=model_path,
        tokenizer="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        trust_remote_code=trust_remote_code,
        max_model_len=max_model_len
    )
    tokenizer = llm.get_tokenizer()

    # Build conversations
    conversations = [[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": prompt}
    ] for prompt in prompts]

    # Warmup
    print("Warming up...")
    llm.chat([conversations[0]], sampling_params)

    # Benchmark
    print("Running benchmark...")
    start_time = time.perf_counter()
    outputs = llm.chat(conversations, sampling_params)
    elapsed_time = time.perf_counter() - start_time

    # Calculate metrics
    total_prompts = len(prompts)
    throughput = total_prompts / elapsed_time
    
    # Token counting
    input_lengths = []
    output_lengths = []
    latencies = []
    
    for conv, output in zip(conversations, outputs):
        # Input tokens
        input_text = tokenizer.apply_chat_template(conv, tokenize=False)
        input_tokens = len(tokenizer.encode(input_text))
        input_lengths.append(input_tokens)
        
        # Output tokens
        output_text = output.outputs[0].text
        output_tokens = len(tokenizer.encode(output_text))
        output_lengths.append(output_tokens)
        
    total_input_tokens = sum(input_lengths)
    total_output_tokens = sum(output_lengths)
    total_tokens = total_input_tokens + total_output_tokens
    
    # Latency percentiles (approximate for batch processing)
    batch_latency = elapsed_time
    avg_latency = batch_latency / total_prompts

    # Print results
    print("\nBenchmark Results:")
    print(f"Total Prompts: {total_prompts}")
    print(f"Total Time: {elapsed_time:.4f} seconds")
    print(f"Throughput: {throughput:.2f} prompts/sec")
    print(f"\nToken Statistics:")
    print(f"Input Tokens: {total_input_tokens} ({np.mean(input_lengths):.1f} avg/prompt)")
    print(f"Output Tokens: {total_output_tokens} ({np.mean(output_lengths):.1f} avg/prompt)")
    print(f"Total Tokens: {total_tokens} ({total_tokens/elapsed_time:.2f} tokens/sec)")
    print(f"\nLatency Statistics:")
    print(f"Batch Latency: {batch_latency:.4f} sec")
    print(f"Average Latency: {avg_latency:.4f} sec/prompt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Benchmark Tool")
    parser.add_argument("--model", type=str, required=True, help="Model path")
    parser.add_argument("--max_model_len", type=int, required=True, help="Max context length")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset JSON")
    parser.add_argument("--num_prompts", type=int, default=100, help="Number of prompts to test")
    parser.add_argument("--trust-remote-code", action="store_true", help="Trust remote code")

    args = parser.parse_args()
    
    benchmark_throughput(
        model_path=args.model,
        max_model_len=args.max_model_len,
        dataset_path=args.dataset,
        num_prompts=args.num_prompts,
        trust_remote_code=args.trust_remote_code
    )

git clone https://github.com/ggerganov/llama.cpp.git
pip install --upgrade huggingface_hub

models=(
#"Gensyn/Qwen2.5-0.5B-Instruct"
#"Gensyn/Qwen2.5-1.5B-Instruct"
#"distilbert/distilgpt2"
#"Qwen/Qwen2.5-7B-Instruct"
"unsloth/Llama-3.2-1B-Instruct"
#"TinyLlama/TinyLlama-1.1B-Chat-v1.0"
#"Qwen/Qwen2.5-0.5B"
#"google/gemma-3-1b-it"
#"sarvamai/sarvam-m"
#"facebook/KernelLLM"
#"nvidia/AceReason-Nemotron-14B"
#"nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1"
#"Intelligent-Internet/II-Medical-8B"
#"Qwen/Qwen3-8B"
"GSAI-ML/LLaDA-8B-Instruct"
#"Qwen/Qwen3-0.6B"
)


convert_model() {
    local model_name=$1
    echo "开始处理模型: $model_name"
    echo "正在下载模型: $model_name"
    huggingface-cli download   --resume-download --local-dir-use-symlinks False --local-dir "./${model_name//\//_}" $model_name
    echo "正在查找模型文件..."
    model_files=$(find "./${model_name//\//_}" -type f \( -name "*.bin" -o -name "*.pt" -o -name "*.safetensors" \) 2>/dev/null)

    if [ -z "$model_files" ]; then
        echo "未找到模型文件，尝试查找GGUF文件..."
    else
        # 转换为GGUF fp16
        echo "找到模型文件，开始转换为GGUF fp16格式..."
        output_dir="/mnt/ai_toolpartition/tool/gguf/${model_name//\//_}_gguf"
        mkdir -p $output_dir
        echo "转换模型文件: $model_file"
        python3 ./llama.cpp/convert_hf_to_gguf.py \
  --outtype f16 \
  --verbose \
  --outfile "$output_dir/${model_name##*/}-f16.gguf" \
  ./${model_name//\//_}  # 仅保留一次模型路径，作为最后一个参数
        if [ $? -eq 0 ]; then
            echo "转换成功，输出文件: $output_dir/${model_name##*/}-fp16.gguf"
            # 删除原始模型文件
            echo "删除原始模型目录: ./${model_name//\//_}"
            rm -rf "./${model_name//\//_}"
        else
            echo "转换失败: $model_file"
        fi
    fi

    echo "模型 $model_name 处理完成"
    echo "------------------------"
}

for model in "${models[@]}"; do
    convert_model "$model"
done

echo "所有模型转换完成！"

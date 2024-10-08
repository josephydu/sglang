

for i in {0..7}
do
    CUDA_VISIBLE_DEVICES=i python3 -m sglang.launch_server --model-path Qwen/Qwen2-7B --host 0.0.0.0 --mem-fraction-static 0.7 & 
done
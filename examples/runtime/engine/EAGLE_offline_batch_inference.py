import sglang as sgl


def main():
    # Sample prompts.
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
        "I am a ....",
        "How is the weather today?",
        "Do you aggree with this?",
        "Can you write an essay for me?",
        "Please translate this sentence to Chinense!",
        "Write a bubble sort use Java for me",
    ]

    # Create a sampling params object.
    sampling_params = {"temperature": 0, "max_new_tokens": 512}

    # Create an LLM.
    llm = sgl.Engine(
        model_path="/workspace/Llama-2-7b-chat-hf",
        speculative_algorithm="EAGLE",
        speculative_draft_model_path="/workspace/sglang-EAGLE-llama2-chat-7B",
        speculative_num_steps=5,
        speculative_eagle_topk=8,
        speculative_num_draft_tokens=64,
    )

    outputs = llm.generate(prompts, sampling_params)

    # Print the outputs.
    # for prompt, output in zip(prompts, outputs):
    #     print("===============================")
    #     print(f"Prompt: {prompt}\nGenerated text: {output['text']}")


# The __main__ condition is necessary here because we use "spawn" to create subprocesses
# Spawn starts a fresh program every time, if there is no __main__, it will run into infinite loop to keep spawning processes from sgl.Engine
if __name__ == "__main__":
    main()

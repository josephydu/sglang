import random
import string

random.seed(42)


def gen_prompt(tokenizer, token_num):
    cha_set = string.ascii_letters + string.digits
    ret = "".join(random.choices(cha_set, k=token_num))
    while len(tokenizer(ret).input_ids) < token_num:
        ret += random.choice(cha_set)
    return ret


import json


def gen_arguments(args, tokenizer):
    try:
        with open(
            f"generated_qas_{args.min_len_q}_{args.max_len_q}_{args.min_len_a}_{args.max_len_a}_{args.turns}_{args.num_qa}.json",
            "r",
        ) as json_file:
            return json.load(json_file)
    except Exception as e:
        pass
    multi_qas = [{"qas": []} for _ in range(args.num_qa)]
    for i in range(args.num_qa):
        qas = multi_qas[i]["qas"]
        turn_range = random.randint(1, args.turns)
        for _ in range(turn_range):
            prompt_len = random.randint(args.min_len_q, args.max_len_q)
            new_tokens = random.randint(args.min_len_a, args.max_len_a)
            qas.append(
                {
                    "prompt": gen_prompt(tokenizer, prompt_len),
                    "new_tokens": new_tokens,
                    "id": i,
                }
            )

    with open(
        f"generated_qas_{args.min_len_q}_{args.max_len_q}_{args.min_len_a}_{args.max_len_a}_{args.turns}_{args.num_qa}.json",
        "w",
    ) as json_file:
        json.dump(multi_qas, json_file, indent=4)
    return multi_qas

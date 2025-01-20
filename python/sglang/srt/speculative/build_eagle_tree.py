import cutex
import torch

# parent_table [bs,topk*depth+)]
# selected_index [bs,draft_token_num-1)]
# verified_seq_len [bs]
# tree_mask [draft_token*(seq_len[0]+draft_token) | draft_token*(seq_len[1]+draft_token) | ..] = [sum(verified_seq_len)*draft_token+bs*draft_token*draft_token]
# positions [bs*draft_token]
# retrive_index [b, draft_token, depth+2]
kernels = cutex.SourceModule(
    """
//cuda
__global__ void build_tree(Tensor<long, 2> parent_list, Tensor<long, 2> selected_index, Tensor<int, 1> verified_seq_len,
        Tensor<bool, 1> tree_mask, Tensor<long, 1> positions, Tensor<long, 3> retrive_index, int topk, int depth, int draft_token_num) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;

    if (tid >= draft_token_num) {
        return;
    }

    int seq_tree_idx = draft_token_num * draft_token_num * bid;
    for (int i = 0; i < bid; i++) {
        seq_tree_idx += verified_seq_len[i] * draft_token_num;
    }
    int seq_len = verified_seq_len[bid];
    int token_tree_idx = seq_tree_idx + (seq_len + draft_token_num) * tid + seq_len + 1;

    for (int i = 0; i < draft_token_num - 1; i++) {
        tree_mask[token_tree_idx + i] = false;
    }

    int position = 0;
    if (tid == 0) {
        positions[bid * draft_token_num] = seq_len;
        retrive_index[bid][0][0] = bid * draft_token_num;
        return;
    }

    int depends_order[10];

    int cur_position = tid - 1;
    while (true) {
        if (cur_position == 64) {
            printf("hahahahha");
            selected_index[bid][cur_position];
        }

        depends_order[position] = cur_position + 1;
        position += 1;

        tree_mask[token_tree_idx + cur_position] = true;

        int parent_tb_idx = selected_index[bid][cur_position] / topk;
        if (parent_tb_idx == 0) {
            break;
        }

        int token_idx = parent_list[bid][parent_tb_idx];
        for (cur_position = 0; cur_position < draft_token_num; cur_position++) {
            if (selected_index[bid][cur_position] == token_idx) {
                break;
            }
        }
    }

    positions[bid * draft_token_num + tid] = position + seq_len;

    int is_leaf = 0;
    for (int i = 1; i < draft_token_num; i++) {
        if (tree_mask[seq_tree_idx + i * (draft_token_num + seq_len) + seq_len + tid]) {
            is_leaf++;
        }
    }

    if (is_leaf == 1) {
        for (int i = 0; i < position; i++) {
            retrive_index[bid][tid][position - i] = depends_order[i] + bid * draft_token_num;
        }
        retrive_index[bid][tid][0] = bid * draft_token_num;
    }
}
//!cuda
""",
    float_bits=16,  # change to 16 to use half precision as `float` type in the above source code.
    boundscheck=True,  # turning on for debug and off for performance (to use full threads of a block), default is on.
)


def build_tree_kernel(parent_list, top_score_index, seq_lens, topk, depth, draft_token):
    bs = seq_lens.numel()
    device = parent_list.device
    tree_mask = torch.full(
        (torch.sum(seq_lens).item() * draft_token + draft_token * draft_token * bs,),
        True,
        device=device,
    )
    retrive_index = torch.full(
        (bs, draft_token, depth + 2), -1, device=device, dtype=torch.long
    )
    positions = torch.empty((bs * draft_token,), device=device, dtype=torch.long)

    print("====================================")
    print("parent_list shape:", parent_list.shape)
    print("top_score_index shape:", top_score_index.shape)
    print("seq_lens shape:", seq_lens.shape)
    print("tree_mask shape:", tree_mask.shape)
    print("positions shape:", positions.shape)
    print("retrive_index shape:", retrive_index.shape)
    print("====================================")

    # 创建保存目录
    save_dir = "error_inputs"
    os.makedirs(save_dir, exist_ok=True)

    # 保存输入数据到文件
    input_data = {
        "parent_list": parent_list.clone(),  # 使用 clone() 确保保存的是原始数据
        "top_score_index": top_score_index.clone(),
        "seq_lens": seq_lens.clone(),
        "topk": topk,
        "depth": depth,
        "draft_token": draft_token,
        "tree_mask": tree_mask.clone(),  # 也可以选择保存 tree_mask
        "positions": positions.clone(),  # 也可以选择保存 positions
        "retrive_index": retrive_index.clone(),  # 也可以选择保存 retrive_index
    }

    input_file = os.path.join(save_dir, "input_data.pt")
    torch.save(input_data, input_file)

    try:
        kernels.build_tree(
            parent_list,
            top_score_index,
            seq_lens.to(torch.int32),
            tree_mask,
            positions,
            retrive_index,
            topk,
            depth,
            draft_token,
            grid=(bs, 1, 1),
            block=(64, 1, 1),
        )
    except Exception as e:
        print(f"Error occurred: {e}. Input data saved to {input_file}.")
        raise  # 重新抛出异常以便后续处理
    index = retrive_index.sum(dim=-1) != -depth - 2
    cum_len = torch.cumsum(torch.sum(index, dim=-1), dim=-1)
    retrive_cum_len = torch.zeros(
        (cum_len.numel() + 1,), dtype=torch.int32, device="cuda"
    )
    retrive_cum_len[1:] = cum_len
    retrive_index = retrive_index[index]
    return tree_mask, positions, retrive_index, retrive_cum_len


import os

if __name__ == "__main__":

    def load_and_call_build_tree(input_file):

        # 加载输入数据
        input_data = torch.load(input_file)

        # 从加载的数据中提取参数
        parent_list = input_data["parent_list"]
        top_score_index = input_data["top_score_index"]
        seq_lens = input_data["seq_lens"]
        topk = input_data["topk"]
        depth = input_data["depth"]
        draft_token = input_data["draft_token"]
        tree_mask = input_data["tree_mask"]
        positions = input_data["positions"]
        retrive_index = input_data["retrive_index"]

        print("====================================")
        print("parent_list shape:", parent_list.shape)
        print("top_score_index shape:", top_score_index.shape)
        print("seq_lens shape:", seq_lens.shape)
        print("tree_mask shape:", tree_mask.shape)
        print("positions shape:", positions.shape)
        print("retrive_index shape:", retrive_index.shape)
        print("====================================")
        # 调用 build_tree 函数
        try:
            kernels.build_tree(
                parent_list,
                top_score_index,
                seq_lens.to(torch.int32),
                tree_mask,
                positions,
                retrive_index,
                topk,
                depth,
                draft_token,
                grid=(parent_list.size(0), 1, 1),  # 根据实际情况设置 grid
                block=(64, 1, 1),  # 根据实际情况设置 block
            )
            print("build_tree executed successfully.")
        except Exception as e:
            print(f"Error occurred during build_tree execution: {e}")

    # 使用示例
    input_file = "error_inputs/input_data.pt"  # 指定保存的输入数据文件路径
    if os.path.exists(input_file):
        load_and_call_build_tree(input_file)
    else:
        print(f"Input file {input_file} does not exist.")

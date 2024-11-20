from multiprocessing import Process, Manager

def serialize_tree_to_list(node, tree_list, parent_index=None, key=None):
    # 当前节点的索引
    current_index = len(tree_list)
    # 将当前节点存储为一个字典
    tree_list.append({
        'key': node.key,
        'value': node.value,
        'children': [],
        'parent': parent_index,
        'char': key
    })

    # 递归地处理所有子节点
    for char, child in node.children.items():
        child_index = serialize_tree_to_list(child, tree_list, current_index, char)
        tree_list[current_index]['children'].append((char, child_index))

    return current_index

def build_sample_tree():
    root = TreeNode()
    root.key = ''
    root.children['a'] = TreeNode()
    root.children['a'].key = 'a'
    root.children['a'].children['b'] = TreeNode()
    root.children['a'].children['b'].key = 'ab'
    root.children['a'].children['b'].value = 'Value for ab'
    return root
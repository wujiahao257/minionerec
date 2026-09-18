import fire
import pandas as pd
import json
from tqdm import tqdm

def merge(input_path, output_path, cuda_list):
    """按给定设备编号顺序合并预测 JSON 列表。

    Args:
        input_path (str): 输入文件或分片所在目录路径。
        output_path (str): 当前读写操作使用的文件或目录路径。
        cuda_list (int | list[int] | tuple[int, ...]): 设备编号或编号序列；用于分片命名和合并顺序，不在函数内启动 GPU 计算。

    Returns:
        None: 写入合并后的预测 JSON。
    """
    if type(cuda_list) == int:
        cuda_list = [cuda_list]
    cuda_list = list(cuda_list)
    data = []
    for i in tqdm(cuda_list):
        with open(f'{input_path}/{i}.json', 'r') as f:
            data.extend(json.load(f))
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == '__main__':
    fire.Fire(merge)

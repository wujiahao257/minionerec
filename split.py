import fire
import os
import pandas as pd

def split(input_path, output_path, cuda_list):
    """按行数将测试 CSV 连续切成多个设备编号命名的分片。

    Args:
        input_path (str): 输入文件或分片所在目录路径。
        output_path (str): 当前读写操作使用的文件或目录路径。
        cuda_list (int | list[int] | tuple[int, ...]): 设备编号或编号序列；用于分片命名和合并顺序，不在函数内启动 GPU 计算。

    Returns:
        None: 在输出目录写出编号 CSV。
    """
    if type(cuda_list) == int:
        cuda_list = [cuda_list]
    df = pd.read_csv(input_path)
    # df = df.sample(frac=1).reset_index(drop=True)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    df_len = len(df)
    cuda_list = list(cuda_list)
    cuda_num = len(cuda_list)
    for i in range(cuda_num):
        start = i * df_len // cuda_num
        end = (i+1) * df_len // cuda_num
        df[start:end].to_csv(f'{output_path}/{cuda_list[i]}.csv', index=True)
        
if __name__ == '__main__':
    fire.Fire(split)

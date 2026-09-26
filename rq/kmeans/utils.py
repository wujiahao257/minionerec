
import datetime
import os


def ensure_dir(dir_path):

    """在目录不存在时创建目录，供后续数据或 checkpoint 写入。

    Args:
        dir_path (str): 需要创建或写入输出文件的目录路径。

    Returns:
        None: 在文件系统中创建目录。
    """
    os.makedirs(dir_path, exist_ok=True)

def set_color(log, color, highlight=True):
    """在文本两端添加 ANSI 颜色与复位控制字符。

    Args:
        log (str): 要添加终端 ANSI 颜色的日志文本。
        color (str): 颜色名，如 green、pink；未知值回退到白色。
        highlight (bool): 是否使用高亮 ANSI 样式。

    Returns:
        str: 带终端颜色的日志文本。
    """
    color_set = ["black", "red", "green", "yellow", "blue", "pink", "cyan", "white"]
    try:
        index = color_set.index(color)
    except:
        index = len(color_set) - 1
    prev_log = "\033["
    if highlight:
        prev_log += "1;3"
    else:
        prev_log += "0;3"
    prev_log += str(index) + "m"
    return prev_log + log + "\033[0m"

def get_local_time():
    """生成用于 checkpoint 子目录命名的本地时间字符串。

    Args:
        无显式参数。

    Returns:
        str: 格式为 Mon-DD-YYYY_HH-MM-SS 的时间。
    """
    cur = datetime.datetime.now()
    cur = cur.strftime("%b-%d-%Y_%H-%M-%S")

    return cur

def delete_file(filename):
    """若指定文件存在则删除，用于清理无需保留的 checkpoint。

    Args:
        filename (str): 当前读写操作使用的文件或目录路径。

    Returns:
        None: 可能删除一个文件。
    """
    if os.path.exists(filename):
        os.remove(filename)

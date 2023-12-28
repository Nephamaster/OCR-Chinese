import os
import io
import sys

# 字符编码转换, GB18030是GBK的父集，所以能兼容GBK不能编码的字符
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='gb18030')

# 以二进制形式读取汉字字母表
with open('char_std_5990.txt', 'rb') as fd:
    cvt_lines = fd.readlines()

cvt_dict = {}
for i, line in enumerate(cvt_lines):
    """ key 为行数，value 为对应汉字的二进制码"""
    key = i+1
    value = line.strip()
    cvt_dict[key] = value

if __name__ == "__main__":
    """
    在终端执行:
    cd data
    python convert_text_list.py path/to/your/SyntheticChineseStringDataset/test.txt > test_list.txt
    python convert_text_list.py path/to/your/SyntheticChineseStringDataset/train.txt > train_list.txt
    即可分别创建 train 和 test 的 labels
    """
    cvt_fpath = sys.argv[1] # 原始label的路径

    with open(cvt_fpath) as fd:
        lines = fd.readlines()

    for line in lines:
        line_split = line.strip().split()
        img_path = line_split[0]
        label = ''
        for i in line_split[1:]:
            label += cvt_dict[int(i)].decode()
        print(img_path, ' ', label)
import os
import shutil

# 1. 这里设置你要整理的文件夹路径 ('.' 表示当前文件夹)
path = '.'

# 2. 只有这些后缀的文件会被移动
extensions = {
    'image': ['.jpg', '.png', '.jpeg', '.gif'],
    '文档': ['.doc', '.docx', '.pdf', '.txt'],
    '音乐': ['.mp3', '.wav']
}

# 3. 开始干活
for filename in os.listdir(path):
    # 跳过代码自己，别把自己也移动了
    if filename == 'run.py':
        continue
        
    # 获取文件后缀名 (比如 .jpg)
    ext = os.path.splitext(filename)[1].lower()
    
    # 看看这个后缀属于哪一类
    for folder_name, ext_list in extensions.items():
        if ext in ext_list:
            # 如果没有这个分类文件夹，就新建一个
            if not os.path.exists(folder_name):
                os.mkdir(folder_name)
            
            # 移动文件
            shutil.move(filename, folder_name + '/' + filename)
            print(f"把 {filename} 移动到了 {folder_name} 文件夹")
print("主人，任务全部完成，请检阅！")

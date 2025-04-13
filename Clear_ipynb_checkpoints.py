import os
import shutil

# 获取当前目录
current_directory = os.getcwd()

# 遍历当前目录及其子目录
for root, dirs, files in os.walk(current_directory, topdown=False):
    # 检查是否有名为 '.ipynb_checkpoints' 的文件夹
    if '.ipynb_checkpoints' in dirs:
        folder_path = os.path.join(root, '.ipynb_checkpoints')
        # 删除文件夹及其内容
        shutil.rmtree(folder_path)
        print(f"已删除文件夹: {folder_path}")

print("清理完成！")

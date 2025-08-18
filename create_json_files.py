import json
import os
import shutil

# 确保目录存在
os.makedirs('data', exist_ok=True)
os.makedirs('data/papers', exist_ok=True)

# 清空papers目录
print("Clearing papers directory...")
for file in os.listdir('data/papers'):
    file_path = os.path.join('data/papers', file)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
    except Exception as e:
        print(f"Error deleting {file_path}: {e}")

# 创建空的知识库文件
print("Creating empty knowledge_base.json...")
with open('data/knowledge_base.json', 'w', encoding='utf-8') as f:
    json.dump([], f)

# 创建空的论文元数据文件
print("Creating empty papers_metadata.json...")
with open('data/papers_metadata.json', 'w', encoding='utf-8') as f:
    json.dump([], f)

print("Successfully created empty JSON files with correct UTF-8 encoding and cleared papers directory.") 
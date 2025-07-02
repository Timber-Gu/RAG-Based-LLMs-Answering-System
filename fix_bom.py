import json

# Read the file with BOM handling
with open('data/knowledge_base.json', 'r', encoding='utf-8-sig') as f:
    data = json.load(f)

# Write it back without BOM
with open('data/knowledge_base.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print('✅ Fixed UTF-8 BOM issue in knowledge_base.json') 
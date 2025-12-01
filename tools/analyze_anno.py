from pathlib import Path
import json

root_data = Path('/volume/datasets/mp100')
anno_dir = root_data / 'annotations'

categories = []
anno_categories = {}

for anno_file in sorted(anno_dir.glob('*.json')):
    with open(anno_file, 'r') as f:
        data = json.load(f)

    # print(data['annotations'][0])
    # print(data['categories'][0])

    categories_id = {}
    for cate in data['categories']:
        categories_id[cate['id']] = {
            "name": f"{cate['supercategory']}/{cate['name']}",
            "num": 0
        }

    for anno in data['annotations']:
        cate_id = anno['category_id']
        categories_id[cate_id]['num'] += 1

    print(f'File: {anno_file.name}')
    # print(data['categories'])
    cate_name = [{
        "name": f"{cate['supercategory']}/{cate['name']}",
        "num": categories_id[cate['id']]['num'],
        "id": cate['id']
    } for cate in data['categories']]
    for cate in data['categories']:
        full_name = f"{cate['supercategory']}/{cate['name']}"
        if full_name not in anno_categories:
            anno_categories[full_name] = set()
        anno_categories[full_name].add(anno_file.name)

    categories.append({
        'file': anno_file.name,
        'num_categories': len(cate_name),
        'categories': cate_name
    })

with open('category_summary.json', 'w') as f:
    json.dump(categories, f, indent=4)

# Convert sets to lists for JSON serialization
for key in anno_categories:
    anno_categories[key] = sorted(list(anno_categories[key]))
with open('category_occurrences.json', 'w') as f:
    json.dump(anno_categories, f, indent=4)
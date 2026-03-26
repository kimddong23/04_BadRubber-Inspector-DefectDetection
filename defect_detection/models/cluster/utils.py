import re
import os
import shutil
import sys
import numpy as np
from tqdm import tqdm
import pickle
import torch
from collections import defaultdict
import random
from collections import Counter
random.seed(42)

def load_embeddings(path):

    with open(path, "rb") as f:
        data = pickle.load(f)

    img_names = data["data"]["img_names"]
    img_emb = data["data"]["img_emb"]

    if torch.is_tensor(img_emb):
        img_emb = img_emb.cpu().numpy()

    return img_names, img_emb

def load_embeddings_list(paths):
    img_names = []
    img_emb = []
    for path in tqdm(paths):
        n, e = load_embeddings(path)
        img_names.extend(n)
        img_emb.append(e)
    img_emb = np.concatenate(img_emb, axis=0)
    return img_names, img_emb

def reset_dir(path):
    if os.path.exists(path):
        print(f"Cleaning old results: {path}")
        shutil.rmtree(path)

def collect_images(folder):
    imgs = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                if "rep" in f:
                    continue
                imgs.append(os.path.join(root, f))
    return imgs

def collect_images_list(folders):
    imgs = []
    for folder in tqdm(folders):
        imgs.extend(collect_images(folder))
    return imgs

def extract_label(filename):
    import os
    name = os.path.basename(filename)
    name = os.path.splitext(name)[0]
    parts = name.split("_")
    label_parts = []
    for p in parts:
        if p.replace(".", "").isdigit():
            if label_parts:
                break
            else:
                continue
        label_parts.append(p)
    return "_".join(label_parts)

def collect_images_with_label(folder, conditions):
    imgs = {}
    condition_cnt = 0
    total_cnt = 0
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                if "rep" in f:
                    continue
                # label = os.path.basename(root)
                # label = label.rsplit("_", 1)[0]
                total_cnt += 1
                for condition in conditions:
                    if condition in f:
                        condition_cnt += 1
                        break
                else:
                    continue
                label = extract_label(f)
                if label not in imgs:
                    imgs[label] = []
                imgs[label].append(os.path.join(root, f))
    print(f"{folder} Condition count (conditions: {conditions}): {condition_cnt} / {total_cnt}")
    return imgs

def collect_images_with_label_list(folders, conditions, data_limit):
    imgs_list = []
    imgs_dict = defaultdict(list)
    
    for folder in tqdm(folders):
        new_dict = collect_images_with_label(folder, conditions)
        
        for label, imgs in new_dict.items():
            imgs_dict[label].extend(imgs)

    total_num = 0
    for label, imgs in imgs_dict.items():
        total_num += len(imgs)
        print(f"{label:<8}: {len(imgs):,}")
    print(f"Total number of images: {total_num:,}\n")

    for label, imgs in imgs_dict.items():
        limit = data_limit[label] if data_limit is not None else len(imgs)
        sample_num = min(limit, len(imgs))
        sampled = sorted(random.sample(imgs, sample_num))
        imgs_list.extend(sampled)
        print(f"{label:<15}: {len(imgs):,} ({len(imgs) / total_num * 100:.2f})% -> {len(sampled):,}")

    return imgs_list

def find_indices(img_paths, img_names, recluster):

    name_to_idx = {os.path.basename(p): i for i, p in enumerate(img_names)}

    idx = []

    x = 0
    for p in img_paths:

        name = os.path.basename(p)

        if recluster and "_" in name:
            name = name.split("_", 1)[-1]

        if name in name_to_idx:
            idx.append(name_to_idx[name])
        else:
            print(f"Warning: {p} not found")
            x += 1
    if x > 0:
        print(f"Warning: {x} images not found")

    return np.array(idx)

def next_path(base_path, limit=5000):
    os.makedirs(os.path.dirname(base_path), exist_ok=True)
    parent = os.path.dirname(base_path)
    name = os.path.basename(base_path)

    idxs = [
        int(d.split("_")[-1])
        for d in os.listdir(parent)
        if d.startswith(name + "_") and d.split("_")[-1].isdigit()
    ]

    if not idxs:
        return_dir = os.path.join(parent, f"{name}_{0:03d}")
        return return_dir

    i = max(idxs)
    last_dir = os.path.join(parent, f"{name}_{i:03d}")

    if len(os.listdir(last_dir)) >= limit:
        return_dir = os.path.join(parent, f"{name}_{i+1:03d}")
        return return_dir

    return last_dir

def get_leaf_dirs(root_path):
    leaf_dirs = []
    for current_path, dirs, files in os.walk(root_path):
        if not dirs:  # 하위 폴더가 없으면 leaf
            leaf_dirs.append(current_path)
    return leaf_dirs


def check_duplicates(items, name, on_duplicate="error"):
    """
    items: 리스트
    name: 출력용 이름 (e.g., 'src_trees')
    on_duplicate: 'error' | 'confirm' | 'ignore'
    
    return: duplicates 리스트
    """
    counter = Counter(items)
    duplicates = [k for k, v in counter.items() if v > 1]

    if not duplicates:
        return []

    if on_duplicate == "error":
        print(f"❌ [ERROR] {name} 중복 발견:")
        for d in duplicates:
            print(f" - {d}")
        sys.exit(1)

    elif on_duplicate == "confirm":
        print(f"⚠️ [WARNING] {name} 중복 발견:")
        for d in duplicates:
            print(f" - {d}")

        user_input = input("\n그래도 계속 진행하시겠습니까? (y/n): ").strip().lower()
        if user_input not in ("y", "yes"):
            print("작업을 중단합니다.")
            sys.exit(1)

    elif on_duplicate == "ignore":
        pass

    else:
        raise ValueError(f"Unknown mode: {on_duplicate}")

    return duplicates
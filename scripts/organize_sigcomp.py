import os
import shutil
import pandas as pd
from random import randint, choice, seed

seed(42)

RAW_DIR = "data/raw/Testdata_SigComp2011/SigComp11-Offlinetestset/Dutch/"
OUT_DIR = "data/processed/signatures/"
LABEL_FILE = "data/processed/labels.csv"

os.makedirs(f"{OUT_DIR}/genuine", exist_ok=True)
os.makedirs(f"{OUT_DIR}/forged", exist_ok=True)

def parse_filename(fname, parent_folder):
    """Extract user ID and determine if signature is genuine or forged"""
    base = os.path.basename(fname)
    user_id = os.path.basename(parent_folder)  # Use folder name as user_id

    # Genuine if in Reference folder
    if "Reference" in parent_folder:
        return user_id, 1

    # If in Questioned folder, check if genuine or forged by filename pattern
    elif "Questioned" in parent_folder:
        try:
            sig_num = int(base.split('_')[0])  # e.g. '13' from '13_013.PNG'
        except:
            return user_id, None
        return user_id, 1 if sig_num <= 5 else 0

    return user_id, None

data = []

# Process Reference (genuine)
ref_dir = os.path.join(RAW_DIR, "Reference(646)")
for user_folder in os.listdir(ref_dir):
    user_path = os.path.join(ref_dir, user_folder)
    if not os.path.isdir(user_path):
        continue
    for fname in os.listdir(user_path):
        if not fname.lower().endswith(".png"):
            continue
        src = os.path.join(user_path, fname)
        user_id, is_genuine = parse_filename(fname, user_path)
        if is_genuine is None:
            continue
        gender = randint(0, 1)
        age_group = choice([0, 1, 2])

        dest = os.path.join(OUT_DIR, "genuine", f"{user_folder}_{fname}")
        shutil.copy(src, dest)
        data.append({
            "filename": f"genuine/{user_folder}_{fname}",
            "user_id": user_id,
            "is_genuine": is_genuine,
            "gender": gender,
            "age_group": age_group
        })

# Process Questioned (mixed genuine + forged)
q_dir = os.path.join(RAW_DIR, "Questioned(1287)")
for user_folder in os.listdir(q_dir):
    user_path = os.path.join(q_dir, user_folder)
    if not os.path.isdir(user_path):
        continue
    for fname in os.listdir(user_path):
        if not fname.lower().endswith(".png"):
            continue
        src = os.path.join(user_path, fname)
        user_id, is_genuine = parse_filename(fname, user_path)
        if is_genuine is None:
            continue
        label = "genuine" if is_genuine else "forged"
        gender = randint(0, 1)
        age_group = choice([0, 1, 2])

        dest = os.path.join(OUT_DIR, label, f"{user_folder}_{fname}")
        shutil.copy(src, dest)
        data.append({
            "filename": f"{label}/{user_folder}_{fname}",
            "user_id": user_id,
            "is_genuine": is_genuine,
            "gender": gender,
            "age_group": age_group
        })

# Save CSV
df = pd.DataFrame(data)
df.to_csv(LABEL_FILE, index=False)
print(f"✅ Saved {len(df)} entries to {LABEL_FILE}")

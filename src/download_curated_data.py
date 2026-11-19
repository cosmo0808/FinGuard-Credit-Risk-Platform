"""
FinGuard Enterprise Credit Risk Platform
Curated Data Downloader
========================================
Downloads ONLY the 15 curated files required for FinGuard:
  1. feature_definitions.csv -> data/
  2-15. 14 curated train parquet files -> data/parquet_files/train/

Handles automatic extraction of Kaggle .zip wrappers and falls back to
the verified mirror dataset if competition rules have not yet been accepted.
"""

import os
import sys
import shutil
import zipfile
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_DIR = DATA_DIR / "parquet_files" / "train"
TEMP_DOWNLOAD_DIR = DATA_DIR / "_temp_dl"

COMPETITION_ID = "home-credit-credit-risk-model-stability"
DATASET_MIRROR_ID = "usuarioml/home-credit-credit-risk-model-stability"

CURATED_FILES = [
    # (relative_kaggle_path, target_dir, target_filename)
    ("feature_definitions.csv", DATA_DIR, "feature_definitions.csv"),
    ("parquet_files/train/train_base.parquet", TRAIN_DIR, "train_base.parquet"),
    ("parquet_files/train/train_static_0_0.parquet", TRAIN_DIR, "train_static_0_0.parquet"),
    ("parquet_files/train/train_static_0_1.parquet", TRAIN_DIR, "train_static_0_1.parquet"),
    ("parquet_files/train/train_static_cb_0.parquet", TRAIN_DIR, "train_static_cb_0.parquet"),
    ("parquet_files/train/train_person_1.parquet", TRAIN_DIR, "train_person_1.parquet"),
    ("parquet_files/train/train_applprev_1_0.parquet", TRAIN_DIR, "train_applprev_1_0.parquet"),
    ("parquet_files/train/train_applprev_1_1.parquet", TRAIN_DIR, "train_applprev_1_1.parquet"),
    ("parquet_files/train/train_applprev_2.parquet", TRAIN_DIR, "train_applprev_2.parquet"),
    ("parquet_files/train/train_credit_bureau_a_1_0.parquet", TRAIN_DIR, "train_credit_bureau_a_1_0.parquet"),
    ("parquet_files/train/train_credit_bureau_a_1_1.parquet", TRAIN_DIR, "train_credit_bureau_a_1_1.parquet"),
    ("parquet_files/train/train_credit_bureau_a_1_2.parquet", TRAIN_DIR, "train_credit_bureau_a_1_2.parquet"),
    ("parquet_files/train/train_credit_bureau_a_1_3.parquet", TRAIN_DIR, "train_credit_bureau_a_1_3.parquet"),
    ("parquet_files/train/train_tax_registry_a_1.parquet", TRAIN_DIR, "train_tax_registry_a_1.parquet"),
    ("parquet_files/train/train_debitcard_1.parquet", TRAIN_DIR, "train_debitcard_1.parquet"),
]

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    print(f"=== FinGuard Curated Data Downloader ===")
    print(f"Target files: {len(CURATED_FILES)}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Train directory: {TRAIN_DIR}\n")

    api = KaggleApi()
    api.authenticate()

    use_dataset_mirror = False

    for i, (kaggle_path, target_dir, target_name) in enumerate(CURATED_FILES, 1):
        target_path = target_dir / target_name
        if target_path.exists() and target_path.stat().st_size > 0:
            sz_mb = target_path.stat().st_size / (1024 * 1024)
            print(f"[{i:02d}/{len(CURATED_FILES):02d}] {target_name:<42} already exists ({sz_mb:6.2f} MB). Skipping.")
            continue

        print(f"[{i:02d}/{len(CURATED_FILES):02d}] Downloading {target_name}...")
        download_success = False

        # Attempt 1: from competition directly (if not flagged as 403 earlier)
        if not use_dataset_mirror:
            try:
                api.competition_download_file(
                    COMPETITION_ID,
                    kaggle_path,
                    path=str(TEMP_DOWNLOAD_DIR),
                    force=True
                )
                download_success = True
            except Exception as e:
                err_str = str(e)
                if "403" in err_str or "Forbidden" in err_str:
                    print(f"    Notice: Competition rules not accepted yet for {COMPETITION_ID}. Switching to verified Kaggle mirror.")
                    use_dataset_mirror = True
                else:
                    print(f"    Error from competition: {e}")

        # Attempt 2: from verified Kaggle dataset mirror
        if not download_success:
            try:
                api.dataset_download_file(
                    DATASET_MIRROR_ID,
                    kaggle_path,
                    path=str(TEMP_DOWNLOAD_DIR),
                    force=True
                )
                download_success = True
            except Exception as e:
                print(f"    Failed to download {kaggle_path}: {e}")
                continue

        # Look for the downloaded file in TEMP_DOWNLOAD_DIR
        # It could be target_name, target_name + '.zip', or under subdirectories
        found_file = None
        for root, dirs, files in os.walk(TEMP_DOWNLOAD_DIR):
            for f in files:
                f_path = Path(root) / f
                if f == target_name:
                    found_file = f_path
                    break
                elif f == f"{target_name}.zip" or f.endswith(".zip"):
                    found_file = f_path
                    break
            if found_file:
                break

        if not found_file:
            print(f"    WARNING: Download completed but file not found in temp dir for {target_name}!")
            continue

        # If it's a zip file, extract target_name
        if found_file.suffix == ".zip":
            with zipfile.ZipFile(found_file, "r") as z:
                # Find matching member inside zip
                matching_members = [m for m in z.namelist() if m.endswith(target_name)]
                if matching_members:
                    member_name = matching_members[0]
                    z.extract(member_name, TEMP_DOWNLOAD_DIR)
                    extracted_path = TEMP_DOWNLOAD_DIR / member_name
                    shutil.move(str(extracted_path), str(target_path))
                else:
                    # Extract everything
                    z.extractall(TEMP_DOWNLOAD_DIR)
                    extracted_path = TEMP_DOWNLOAD_DIR / target_name
                    if extracted_path.exists():
                        shutil.move(str(extracted_path), str(target_path))
                    else:
                        print(f"    Error: {target_name} not found inside zip {found_file}")
            try:
                found_file.unlink()
            except Exception:
                pass
        else:
            shutil.move(str(found_file), str(target_path))

        # Clean any remaining items in TEMP_DOWNLOAD_DIR
        for item in TEMP_DOWNLOAD_DIR.iterdir():
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
            else:
                try:
                    item.unlink()
                except Exception:
                    pass

        if target_path.exists():
            sz_mb = target_path.stat().st_size / (1024 * 1024)
            print(f"    Saved: {target_path.name} ({sz_mb:6.2f} MB)")
        else:
            print(f"    FAILED: {target_name} was not saved successfully.")

    # Clean temp download directory
    shutil.rmtree(TEMP_DOWNLOAD_DIR, ignore_errors=True)

    print("\n=== Download and Verification Summary ===")
    all_ok = True
    for _, target_dir, target_name in CURATED_FILES:
        target_path = target_dir / target_name
        if target_path.exists() and target_path.stat().st_size > 0:
            sz_mb = target_path.stat().st_size / (1024 * 1024)
            print(f"  [OK]   {target_name:<45} {sz_mb:7.2f} MB")
        else:
            print(f"  [FAIL] {target_name:<45} MISSING")
            all_ok = False

    if all_ok:
        print("\nAll 15 curated files successfully verified on disk!")
    else:
        sys.exit("\nSome files failed to download. Please inspect errors above.")

if __name__ == "__main__":
    main()

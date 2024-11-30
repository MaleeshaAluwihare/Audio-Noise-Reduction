import zipfile
import os

zip_file_path = "./DeepFilterNet2.zip"

extract_to_path = "./"

os.makedirs(extract_to_path, exist_ok=True)

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to_path)

print(f"Extracted {zip_file_path} to {extract_to_path}")

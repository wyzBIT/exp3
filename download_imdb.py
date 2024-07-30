import os
import tarfile
import requests
from pathlib import Path
from tqdm import tqdm


def download_imdb_dataset(url, dest_folder):
    Path(dest_folder).mkdir(parents=True, exist_ok=True)

    dataset_file = os.path.join(dest_folder, os.path.basename(url))

    print(f"开始下载 {dataset_file}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 每次读取的块大小

    with open(dataset_file, 'wb') as file, tqdm(
            desc=dataset_file,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            bar.update(len(data))
            file.write(data)
    print("下载完成！")

    print(f"开始解压 {dataset_file}...")
    with tarfile.open(dataset_file, 'r:gz') as tar:
        tar.extractall(path=dest_folder)
    print("解压完成！")


url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
dest_folder = "data/imdb"

download_imdb_dataset(url, dest_folder)

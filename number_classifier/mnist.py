import gzip
import os
import urllib.request

import numpy as np
import pickle

class Mnist:
    def __init__(self, dataset_dir: str) -> None:
        self._url_base: str = 'http://yann.lecun.com/exdb/mnist/'
        self._mnist_files: str = {
            'train_img':'train-images-idx3-ubyte.gz',
            'train_label':'train-labels-idx1-ubyte.gz',
            'test_img':'t10k-images-idx3-ubyte.gz',
            'test_label':'t10k-labels-idx1-ubyte.gz'
        }
        self._img_dim = (1, 28, 28)
        self._img_size = self._img_dim[1] * self._img_dim[2]

        self.dataset_dir: str = dataset_dir
        os.makedirs(self.dataset_dir, exist_ok=True)
        self._mnist_pkl = f"{dataset_dir}/mnist.pkl"
    
    def _file_path(self, file_name: str) -> str:
        return f"{self.dataset_dir}/{file_name}"

    def download(self) -> None:
        for file_name in self._mnist_files.values():
            self._download_file(file_name)

    def _download_file(self, file_name: str) -> None:
        file_path = self._file_path(file_name)
        if os.path.exists(file_path):
            print(f"{file_name} already exists.")
            return
        
        print(f"Downloading {file_name}")
        urllib.request.urlretrieve(f'{self._url_base}/{file_name}', file_path)
        print("Done!")

    def _load_label(self, file_name: str) -> np.ndarray:
        file_path = self._file_path(file_name)
        with gzip.open(file_path, 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)

        return labels

    def _load_img(self, file_name: str) -> np.ndarray:
        file_path = self._file_path(file_name)
        with gzip.open(file_path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
            data = data.reshape(-1, self._img_size)

        return data
    
    def _convert_ndarray(self) -> dict:
        dataset = {}
        dataset["train_img"] = self._load_img(self._mnist_files['train_img'])
        dataset["train_label"] = self._load_label(self._mnist_files['train_label'])
        dataset["test_img"] = self._load_img(self._mnist_files['test_img'])
        dataset["test_label"] = self._load_label(self._mnist_files['test_label'])

        return dataset
    
    def init_dataset(self):
        self.download()
        dataset: dict = self._convert_ndarray()
        with open(self._mnist_pkl, "wb") as f:
            pickle.dump(dataset, f, -1)
    
    def load(self):
        if not os.path.exists(self._mnist_pkl):
            self.init_dataset()

        with open(self._mnist_pkl, 'rb') as f:
            dataset = pickle.load(f)

        return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label'])

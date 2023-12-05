import os
import tarfile
import importlib
import urllib.request
from tqdm import tqdm

DATASETS_PATH = f'{os.getcwd()}/datasets/data'
DATASETS_URLS = {
    'maps': 'https://efrosgans.eecs.berkeley.edu/pix2pix/datasets/maps.tar.gz',
    'facades': 'https://efrosgans.eecs.berkeley.edu/pix2pix/datasets/facades.tar.gz',
    'edges2shoes': 'https://efrosgans.eecs.berkeley.edu/pix2pix/datasets/edges2shoes.tar.gz',
    'edges2handbags': 'https://efrosgans.eecs.berkeley.edu/pix2pix/datasets/edges2handbags.tar.gz',
    'cityscapes': 'https://efrosgans.eecs.berkeley.edu/pix2pix/datasets/cityscapes.tar.gz',
    'night2day': 'https://efrosgans.eecs.berkeley.edu/pix2pix/datasets/night2day.tar.gz',
}

def load_configuration(config_file_name: str):
    """
    Loads the configuration dict from the especified file.
    The configuration file needs to be stored in the train_configs directory.

    Inputs:
        >> config_file_name: (str) Name of the file containing the configuration.
    Outputs:
        >> configuration: (Configuration) Configuration dict for the training process.

    """
    config_path = os.path.join("train_configs", f"{config_file_name}.py")
    
    spec = importlib.util.spec_from_file_location(config_file_name, config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    return module.configuration

def download_dataset(dataset_name: str):
    """
    Downloads the dataset and stores it at <root>/datasets/data
    Inputs:
        >> data_params: (Dataparams) Dataset params for training the model.
        >> split: (str) 'train' or 'test'
    Outputs: None
    """
    if dataset_name not in DATASETS_URLS:
        raise ValueError(f'{dataset_name} is not a valid dataset. Please choose between: {DATASETS_URLS.keys()}')
        
    if not os.path.exists(DATASETS_PATH):
        os.makedirs(DATASETS_PATH)

    if not os.path.exists(f'{DATASETS_PATH}/{dataset_name}'):
        print(f'====================> Downloading {dataset_name} dataset')
        download_file_with_progress(DATASETS_URLS[dataset_name], dataset_name)

    return 

def download_file_with_progress(url, dataset_name):
    """
    Download a file from the given URL to the datasets/data dir with a progress bar.

    Parameters:
    - url: The URL of the file to download.
    - dataset_name: The local path where the file will be saved.
    """
    print(f'*************')
    try:
        with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc="Downloading") as t:
            urllib.request.urlretrieve(url, f'{DATASETS_PATH}/{dataset_name}.tar.gz', reporthook=lambda blocknum, blocksize, totalsize: t.update(blocksize))
            print(f'DOWNLOADED')
            extract_tar_gz(f'{DATASETS_PATH}/{dataset_name}.tar.gz', f'{DATASETS_PATH}')
            print(f'EXTRACTED')
            os.remove(f'{DATASETS_PATH}/{dataset_name}.tar.gz')
            print(f'REMOVED')
        print(f"\nDownloaded: {url} to {dataset_name}")
    except Exception as e:
        print(f"Error downloading file: {e}")

def extract_tar_gz(file_path, extract_path: str):
    """
    Extracts the contents of a .tar.gz file.
    Parameters:
    >> file_path: Path to the .tar.gz file.
    >> extract_path: Destination path for extraction. Default is the current directory.
    """
    with tarfile.open(file_path, 'r:gz') as tar:
        tar.extractall(extract_path)
    print(f"Contents of {file_path} extracted to {extract_path}")
    return

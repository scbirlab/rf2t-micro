"""Get model weights for rf2t-micro."""

from typing import Iterable, Optional

import os
import tarfile

from carabiner import print_err
import requests
from tqdm.auto import tqdm

_WEIGHTS_URL = "https://files.ipd.uw.edu/pub/RoseTTAFold/weights.tar.gz"

def get_model_weights(path: Optional[str] = None) -> str:

    """Get the model weights filename.
    
    If not present in `path/.weights/rf2t-micro`, download there.

    """
    if path is None:
        path = os.path.expanduser("~")
    weight_dir = os.path.join(os.path.realpath(path), ".weights", "rf2t-micro")
    weight_filename = os.path.join(weight_dir, "RF2t.pt")
    files_to_keep = (os.path.basename(weight_filename), "Rosetta-DL_LICENSE.txt")

    if not os.path.exists(weight_filename):
        print_err(f"Weights file {weight_filename} does not exist. Downloading...")
        if not os.path.exists(weight_dir):
            os.makedirs(weight_dir)
        r = requests.get(_WEIGHTS_URL, stream=True)
        temp_file = os.path.join(weight_dir, "weights.tar.gz")
        try:
            with open(temp_file, 'wb') as f:
                for chunk in tqdm(r.iter_content(chunk_size=200 * 1024**2)): 
                    if chunk: # filter out keep-alive new chunks
                        f.write(chunk)
                        f.flush()
                        os.fsync(f.fileno())
        except KeyboardInterrupt as e:
            print_err(f"Deleting {temp_file}!")
            os.remove(temp_file)
            raise e
        with tarfile.open(temp_file) as tar:
            print_err(f"Extracting weights from {temp_file} to {weight_dir}...")
            tar.extractall(path=weight_dir, members=files_to_keep, filter='data')
        tardir = os.path.join(weight_dir, "weights")
        os.remove(temp_file)
        for filename in files_to_keep:
            source = os.path.join(tardir, filename)
            destination = os.path.join(weight_dir, filename)
            print_err(f"Moving {source} to {destination}.")
            os.rename(source, destination)
        os.rmdir(tardir)

    print_err(f"Model weights located at {weight_dir}.")        
    return weight_dir

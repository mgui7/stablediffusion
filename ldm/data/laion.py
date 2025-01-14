import webdataset as wds
from PIL import Image
import io
import os
import torchvision
from PIL import Image
import glob
import random
import numpy as np
import pytorch_lightning as pl
from tqdm import tqdm
from omegaconf import OmegaConf
from einops import rearrange
import torch
from webdataset.handlers import warn_and_continue
import sys
sys.path.append(os.path.join(sys.path[0],'../../'))

from ldm.util import instantiate_from_config
from ldm.data.base import PRNGMixin


def dict_collation_fn(samples, combine_tensors=True, combine_scalars=True):
    """Take a list  of samples (as dictionary) and create a batch, preserving the keys.
    If `tensors` is True, `ndarray` objects are combined into
    tensor batches.
    :param dict samples: list of samples
    :param bool tensors: whether to turn lists of ndarrays into a single ndarray
    :returns: single sample consisting of a batch
    :rtype: dict
    """
    keys = set.intersection(*[set(sample.keys()) for sample in samples])
    batched = {key: [] for key in keys}

    for s in samples:
        [batched[key].append(s[key]) for key in batched]

    result = {}
    for key in batched:
        if isinstance(batched[key][0], (int, float)):
            if combine_scalars:
                result[key] = np.array(list(batched[key]))
        elif isinstance(batched[key][0], torch.Tensor):
            if combine_tensors:
                result[key] = torch.stack(list(batched[key]))
        elif isinstance(batched[key][0], np.ndarray):
            if combine_tensors:
                result[key] = np.array(list(batched[key]))
        else:
            result[key] = list(batched[key])
    return result


class WebDataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, tar_base, batch_size, train=None, validation=None,
                 test=None, num_workers=4, multinode=True, min_size=None,
                 max_pwatermark=1.0,
                 **kwargs):
        super().__init__()
        print(f'Setting tar base to {tar_base}')
        self.tar_base = tar_base
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train = train
        self.validation = validation
        self.test = test
        self.multinode = multinode
        self.min_size = min_size  # filter out very small images
        self.max_pwatermark = max_pwatermark # filter out watermarked images

    def make_loader(self, dataset_config, train=True):
        if 'image_transforms' in dataset_config:
            image_transforms = [instantiate_from_config(tt) for tt in dataset_config.image_transforms]
        else:
            image_transforms = []

        image_transforms.extend([torchvision.transforms.ToTensor(),
                                 torchvision.transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])
        image_transforms = torchvision.transforms.Compose(image_transforms)

        if 'transforms' in dataset_config:
            transforms_config = OmegaConf.to_container(dataset_config.transforms)
        else:
            transforms_config = dict()

        transform_dict = {dkey: load_partial_from_config(transforms_config[dkey])
                if transforms_config[dkey] != 'identity' else identity
                for dkey in transforms_config}
        img_key = dataset_config.get('image_key', 'jpeg')
        transform_dict.update({img_key: image_transforms})

        if 'postprocess' in dataset_config:
            postprocess = instantiate_from_config(dataset_config['postprocess'])
        else:
            postprocess = None

        shuffle = dataset_config.get('shuffle', 0)
        shardshuffle = shuffle > 0

        nodesplitter = wds.shardlists.split_by_node if self.multinode else wds.shardlists.single_node_only

        if self.tar_base == "__improvedaesthetic__":
            print("## Warning, loading the same improved aesthetic dataset "
                    "for all splits and ignoring shards parameter.")
            tars = "pipe:aws s3 cp s3://s-laion/improved-aesthetics-laion-2B-en-subsets/aesthetics_tars/{000000..060207}.tar -"
        else:
            tars = os.path.join(self.tar_base, dataset_config.shards)

        dset = wds.WebDataset(
                tars,
                nodesplitter=nodesplitter,
                shardshuffle=shardshuffle,
                handler=wds.warn_and_continue).repeat().shuffle(shuffle)
        print(f'Loading webdataset with {len(dset.pipeline[0].urls)} shards.')

        dset = (dset
                .select(self.filter_keys)
                .decode('pil', handler=wds.warn_and_continue)
                .select(self.filter_size)
                .map_dict(**transform_dict, handler=wds.warn_and_continue)
                )
        if postprocess is not None:
            dset = dset.map(postprocess)
        dset = (dset
                .batched(self.batch_size, partial=False,
                    collation_fn=dict_collation_fn)
                )

        loader = wds.WebLoader(dset, batch_size=None, shuffle=False,
                               num_workers=self.num_workers)

        return loader

    def filter_size(self, x):
        try:
            valid = True
            if self.min_size is not None and self.min_size > 1:
                try:
                    valid = valid and x['json']['original_width'] >= self.min_size and x['json']['original_height'] >= self.min_size
                except Exception:
                    valid = False
            if self.max_pwatermark is not None and self.max_pwatermark < 1.0:
                try:
                    valid = valid and  x['json']['pwatermark'] <= self.max_pwatermark
                except Exception:
                    valid = False
            return valid
        except Exception:
            return False

    def filter_keys(self, x):
        try:
            return ("jpg" in x) and ("txt" in x)
        except Exception:
            return False

    def train_dataloader(self):
        return self.make_loader(self.train)

    def val_dataloader(self):
        return self.make_loader(self.validation, train=False)

    def test_dataloader(self):
        return self.make_loader(self.test, train=False)

if __name__ == "__main__":
    from omegaconf import OmegaConf
    config = OmegaConf.load("configs/stable-diffusion/inj-laion.yaml")
    datamod = WebDataModuleFromConfig(**config["data"]["params"])
    dataloader = datamod.val_dataloader()

    for i,batch in enumerate(dataloader):
        # print(batch.keys())
        # print(batch['jpg'].shape)
        # print(batch['txt'])
        print(i)
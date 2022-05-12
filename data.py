from torch.utils.data import DataLoader, Dataset
from abc import abstractmethod
from torch import Tensor
import json
import numpy as np
from sqlalchemy import create_engine
from torchvision.io import read_image
import pandas as pd
import configparser
import os
from gensim.models import Word2Vec


config = configparser.ConfigParser()
config.read('config.ini')
COVER_LOC = config['IMAGES']['ImageLocation']
dbconf = config["DATABASE"]
uname = dbconf['UserName']
pword = dbconf['Password']
addrs = dbconf['Address']
dname = dbconf['Database']
connstring = f'mysql+pymysql://{uname}:{pword}@{addrs}/{dname}?charset=utf8mb4'
ENGINE = create_engine(connstring)


class BandcampDatasetBase(Dataset):
    def __init__(self, engine=ENGINE, loc=COVER_LOC, transform=None):
        self.df = pd.read_sql('SELECT * FROM albums', engine)
        self.img_dir = loc
        self.transform = transform
        self.img_lines = self.df['id'] + '.jpg'

    def __len__(self):
        return len(self.img_lines)

    def get_img(self, item):
        img_path = os.path.join(self.img_dir, self.img_lines[item])
        image = read_image(img_path).moveaxis([0, 1, 2], [-1, -3, -2])
        if self.transform:
            image = self.transform(image)
        return image

    @abstractmethod
    def __getitem__(self, item):
        raise NotImplementedError


class BandcampTagDataset(BandcampDatasetBase):
    def __init__(self, **kwargs):
        super(BandcampTagDataset, self).__init__(**kwargs)
        self.tag_jsons = self.df['tags']
        self.w2v_model = Word2Vec.load('./models/tags.model')

    def __getitem__(self, item):
        image = self.get_img(item)
        tags = self.tag_jsons.iloc[item]
        tag_list = json.loads(tags)
        tag_list = np.concatenate([s.split(' ') for s in tag_list])
        cbow = np.mean(self.w2v_model.wv[tag_list], axis=0)
        return image, cbow

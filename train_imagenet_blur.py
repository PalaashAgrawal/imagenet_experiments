from datasets import list_datasets, load_dataset
import torch
from fastai.data.all import *
from fastai.vision.all import *
from fastai.distributed import *
from fastai.callback.wandb import *
from PIL import ImageFilter
import wandb

import warnings
warnings.filterwarnings("ignore")

id = "imagenet_blur"
model = xresnext50
lr = 1e-3
epochs = 25

#run by typing (on main node): nohup accelerate launch --config_file multinode_trainin_config/accelerate_config_host.yaml train_imagenet.py &
#and for client server: multinode_trainin_config/accelerate_config_client0.yaml 

import torchvision.transforms as transforms
path = Path('/home/kumarmg/dl_experiments/training imagenet from scratch')
bs = 128

class imagenet_dataset():
    def __init__(self, hf_identifier = "imagenet-1k", use_auth_token = True):
        self.imagenet_ds = load_dataset(hf_identifier ,split='train', use_auth_token=use_auth_token)
        self.label_names = self.imagenet_ds.features['label'].names
        
    def __len__(self): 
        return len(self.imagenet_ds)

    def __getitem__(self, n):
        data = self.imagenet_ds[n]
        return data['image'], data['label']
    
imagenet_ds = imagenet_dataset()

random_split_pct = 0.8
train_ds_len = int(random_split_pct*len(imagenet_ds))
train_ds, val_ds = torch.utils.data.random_split(imagenet_ds, [train_ds_len, len(imagenet_ds)-train_ds_len])

class hf_imagenet_item_tfms: 
    '''hugginface datasets version of imagenet-1k gives PIL.JpegImagePlugin.JpegImageFile files. 
    Also some of the images are B&W (1 channel instead of 3 channel)
    Transforms to convert this dataset to usable tensors
    '''
    def __init__(self, tfms:list = None, blur: bool = False, blur_std = 1.):
        self.blur = blur
        self.blur_std = blur_std
        # if blur: assert self.blur_std is not None #redundant since std (in self.apply_blur) defaults to 0 if blur_std is None 
            
        self.tfms = list(tfms or [lambda img: img.convert(mode='RGB'), self.apply_blur, lambda img: img.resize((256, 256)), 
                                  transforms.PILToTensor(), TensorImage])
    
    
    def apply_blur(self,x):
        # radius = torch.normal(mean=Tensor([0]), std=Tensor([self.blur_std or 0]))
        radius = Tensor(1).uniform_(0,2)
        return x.filter(ImageFilter.GaussianBlur(radius = radius)) if self.blur else x
    
    def __call__(self, x):
        "note that after_item takes output of create_item, which returns a tuple of x,y. So we need to return it as is."
        img, label = x
        ret = transforms.Compose(self.tfms)(img)
        return ret,label

class hf_imagenet_batch_tfms:
    def __init__(self, tfms:list = None):
        self.tfms = list(tfms or [IntToFloatTensor(), Normalize.from_stats(*imagenet_stats), *aug_transforms()])
    
    def __call__(self, x): 
        "after_batch however, unlike after_item, takes on x as input"
        return transforms.Compose(self.tfms)(x)

train_dl = DataLoader(train_ds, bs = bs, shuffle = True, drop_last = True, 
                          after_item =  hf_imagenet_item_tfms(blur=True),
                          after_batch = hf_imagenet_batch_tfms())

val_dl = DataLoader(val_ds, bs = bs*2, shuffle = False, drop_last = True, 
                     after_item = hf_imagenet_item_tfms(blur=True),
                     after_batch =hf_imagenet_batch_tfms())

dls = DataLoaders(train_dl, val_dl).cuda()
dls.vocab = imagenet_ds.label_names
    

def top_5_accuracy(x,y): return  top_k_accuracy(x,y, k=5)
def top_10_accuracy(x,y): return  top_k_accuracy(x,y, k=10)

learn = vision_learner(dls, model, 
                        loss_func = CrossEntropyLossFlat(), 
                        metrics=[accuracy,top_5_accuracy, top_10_accuracy], 
                        pretrained=False, 
                        normalize = False, cbs=WandbCallback()
                        ).to_fp16()

learn.path = path


wandb.init(
    project="imagenet_training_noblur",
    name = f"{id}_{model.__name__}_{epochs}_{lr}",
    # track hyperparameters and run metadata
    config={
    "learning_rate": lr,
    "architecture": model.__name__,
    "dataset": "imagenet-1k",
    "epochs": epochs,
    }
)

with learn.distrib_ctx(): learn.fit(epochs,lr)

learn.save(f"{id}_{model.__name__}_{epochs}_{lr}")
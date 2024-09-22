import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning import seed_everything
import os
import argparse
import numpy as np
from Datasets.datasets import *
import torch
from copy import deepcopy
import json
import shutil
import os
from models.models import MODELS
from torchvision.utils import save_image
import collections

# create floder
def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
# select dataset type

__dataset_val__ = {
            "MFI-WHU-test":Data_eval
            }
class CoolSystem(pl.LightningModule):
    def __init__(self):
        """初始化训练的参数"""
        super(CoolSystem, self).__init__()
        # val datasets
        self.validation_datasets = __dataset_val__[config["test_dataset"]](
                            config                        )
        self.val_batchsize = config["val_batch_size"]
        self.num_workers = config["num_workers"]
        self.save_path_eval= PATH+config["save_path_eval"]

        ensure_dir(self.save_path_eval)
        self.FusionNet =  MODELS[config["fusion_net"]](config)
        self.RecNet =  MODELS[config["rec_net"]](config)
        # Resume from pth ...
        if args.resume is not None:
            print("Loading from existing FusionNet chekpoint")
            ckpt = torch.load(args.resume,map_location=lambda storage, loc: storage)
            new_state_dict = collections.OrderedDict()
            new_state_dict2 = collections.OrderedDict()
            for k in ckpt['state_dict']:
                            # print(k)
                            if k[:10] != 'FusionNet.':
                                continue
                            name = k[10:]
                            new_state_dict[name] = ckpt['state_dict'][k]
                            
            for k in ckpt['state_dict']:
                            # print(k)
                            if k[:7] != 'RecNet.':
                                continue
                            name = k[7:]
                            new_state_dict2[name] = ckpt['state_dict'][k]
                            
            self.FusionNet.load_state_dict(new_state_dict,strict=True)
            self.RecNet.load_state_dict(new_state_dict2,strict=True)


    
    def val_dataloader(self):
        val_loader = data.DataLoader(
                        self.validation_datasets,
                        batch_size=self.val_batchsize,
                        num_workers=self.num_workers,
                        shuffle=False,
                        pin_memory=True,
                    )
        return val_loader


    def on_validation_epoch_start(self):
        self.pred_dic={}
        return super().on_validation_epoch_start()

    def validation_step(self, data, batch_idx):
        self.FusionNet.eval()
        
        """validation step"""
        real_A,real_B,file= data

        fusion = self.FusionNet(real_A, real_B)
        fusion1 = self.FusionNet(real_B, real_A)
        fusion = (fusion1+fusion)/2
        
        save_image(fusion,os.path.join(self.save_path_eval, file[0]))

        return
    

def main():
    # Parse the arguments
    parser = argparse.ArgumentParser(description='PyTorch Testing')
    parser.add_argument('-c', '--config', default='./configs/Eval_MMnet.json',type=str,
                            help='Path to the config file')
    parser.add_argument('-r', '--resume', default=None, type=str,
                            help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('-r1', '--resume_ckpt', default=None, type=str,
                            help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('-d', '--device', default='1', type=str,
                            help='indices of GPUs to enable (default: all)')
    parser.add_argument('-v', '--val', default=False, type=bool,
                            help='Valdation')
    parser.add_argument('-val_path', default=None,type=str, help='Path to the val path')
    
    global args
    args = parser.parse_args()
    global config
    config = json.load(open(args.config))

    # Set seeds.
    seed = 42 
    seed_everything(seed)
    

    # Setting up path
    global PATH
    PATH = "./"+config["experim_name"]+"/"+"/"+config["test_dataset"]+"/"+str(config["tags"])
    ensure_dir(PATH+"/")
    shutil.copy2(args.config, PATH)
    
    # init pytorch-litening
    ddp = DDPStrategy(process_group_backend="gloo",find_unused_parameters=True)
    model = CoolSystem()

    trainer = pl.Trainer(
        strategy=ddp,
        accelerator='gpu', devices=[0],
    )   
    trainer.validate(model,ckpt_path=args.val_path)




if __name__ == '__main__':
    main()
    
import sys
import os
import importlib 
import numpy as np 
import torch
import torch.optim as optim
import copy
import yaml

from easydict import EasyDict as edict
import torch.backends.cudnn as cudnn
from warmup_scheduler import GradualWarmupScheduler
import random

base_dir = os.getcwd()
sys.path.insert(0, base_dir)
import models.model as model  # noqa: E402
import ctools # noqa: E402

def setup_seed(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(config):
    # ===============================> Setup <================================

    setup_seed(123)

    dataloader = importlib.import_module("reader." + config.reader)
    torch.cuda.set_device(config.device) 
    cudnn.benchmark = True

    data = config.data
    save = config.save
    params = config.params

    
    print("===> Read data <===")
    data.origin, folder = ctools.readfolder( data.origin,  [config.person],  reverse=True )
    data.norm,   folder = ctools.readfolder( data.norm,  [config.person],  reverse=True )

    savename = folder[config.person] 
    dataset  = dataloader.loader(data, params.batch_size, shuffle=True, num_workers=6)

    print("===> Model building <===")
    net = model.Model()
    net.train()
    net.cuda()

    
    # Pretrain
    pretrain = config.pretrain
    if pretrain.enable and pretrain.device:
        net.load_state_dict(torch.load(pretrain.path, map_location={f"cuda:{pretrain.device}": f"cuda:{config.device}"}) )
    elif pretrain.enable and not pretrain.device:
        net.load_state_dict(torch.load(pretrain.path))
        
    print("===> optimizer building <===")
    optimizer = optim.Adam(net.parameters(), lr=params.lr, betas=(0.9,0.95))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=params.decay_step, gamma=params.decay)

    if params.warmup:
        scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=params.warmup, after_scheduler=scheduler)

    savepath = os.path.join(save.metapath, save.folder, f"checkpoint/{savename}")

    if not os.path.exists(savepath):
        os.makedirs(savepath)

    # =======================================> Training < ==========================
    print("===> Training <===")
    length = len(dataset)
    total  = length * params.epoch
    timer  = ctools.TimeCounter(total)


    optimizer.zero_grad()
    optimizer.step()
    scheduler.step()

    with open(os.path.join(savepath, "train_log"), 'w') as outfile:
        outfile.write(ctools.DictDumps(config) + '\n')

        for epoch in range(1, params.epoch+1):
            for i, (data, anno) in enumerate(dataset):

                # ------------------forward--------------------
                for key in data.keys():
                    if key != 'name':
                        data[key] = data[key].cuda()

                for key in anno.keys():
                    anno[key] = anno[key].cuda()
 
                loss, losslist = net.loss(data, anno)

                # -----------------backward--------------------
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                rest = timer.step()/3600

                # -----------------loger----------------------
                if i % 20 == 0:
                    log = f"[{epoch}/{params.epoch}]: " +\
                          f"[{i}/{length}] " +\
                          f"loss:{loss:.3f} " +\
                          f"loss_re:{losslist[0]:.3f} " +\
                          f"loss_cls:{losslist[1]:.3f} " +\
                          f"loss_o:{losslist[2]:.3f} " +\
                          f"loss_n:{losslist[3]:.3f} " +\
                          f"lr:{ctools.GetLR(optimizer)} "+\
                          f"rest time:{rest:.2f}h"

                    print(log)
                    outfile.write(log + "\n")
                    sys.stdout.flush()
                    outfile.flush()

            scheduler.step()

            if epoch % save.step == 0:
                torch.save(net.state_dict(), os.path.join(savepath, f"Iter_{epoch}_{save.model_name}.pt"))

if __name__ == "__main__":

    config_path = "/home/mi/zhangss/Gaze/IVGaze/config/train/config_iv.yaml"
    persons     = 3

    # 检查文件路径是否存在
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"The config file at {config_path} does not exist.")

    # 加载 YAML 文件并转换为 EasyDict（可以通过点操作符访问字典内容）
    with open(config_path, 'r') as f:
        config = edict(yaml.load(f, Loader=yaml.FullLoader))

    config      = config.train
    person      = int(persons)
   
    for i in range(person):
        config_i = copy.deepcopy(config)
        config_i.person = i
 
        print("=====================>> (Begin) Training params << =======================")

        print(ctools.DictDumps(config_i))

        print("=====================>> (End) Traning params << =======================")
        
        main(config_i)


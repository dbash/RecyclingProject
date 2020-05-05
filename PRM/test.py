from datasets import *
from model import fc_resnet50
from prm.prm import peak_response_mapping
from losses import multilabel_soft_margin_loss
from tensorboardX import SummaryWriter
from solver import Solver
import os
import yaml, json
from utils import *
from util import *
import PIL.Image
import argparse
import cv2
import pandas as pd
import torchvision
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np


CONFIG = 'config_KITTI_ped.yml'
KITTI_CLASS_NAMES = ['DontCare', 'Van', 'Cyclist', 'Pedestrian', 'Car', 'Truck', 'Misc', 'Tram', 'Person']    

def data_loader(args, test_path=False):

     balanced_image_class_info = pd.read_csv("/projectnb/saenkog/shawnlin/object-tracking/kitti-util/src/train_image_class_info.csv", sep=",")
     shuffled_image_class_info = balanced_image_class_info.sample(frac = 1.0)

     TRAIN_SIZE = int(len(shuffled_image_class_info) * 0.7)
     VAL_SIZE = int(len(shuffled_image_class_info) * 0.1)
     TEST_SIZE = int(len(shuffled_image_class_info) * 0.2)

     train_img_class_info = shuffled_image_class_info[:TRAIN_SIZE]
     val_img_class_info = shuffled_image_class_info[TRAIN_SIZE:TRAIN_SIZE+VAL_SIZE]
     test_img_class_info = shuffled_image_class_info[TRAIN_SIZE+VAL_SIZE:]

     print("Train size: %s" % len(train_img_class_info))
     print("Val size: %s" % len(val_img_class_info))
     print("Test size: %s" % len(test_img_class_info))

     tr_dataset = KittiDataset(train_img_class_info)
     tst_dataset = KittiDataset(test_img_class_info)
     val_dataset = KittiDataset(val_img_class_info)

     dataloaders = {
         "train": DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn),
         "val": DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn),
         "test": DataLoader(tst_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn)
     }

     return dataloaders["train"], dataloaders["val"]


def main(args):

    with open(args.conf, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    
    train_trans = image_transform(**config['train_transform'])
    test_trans = image_transform(**config['test_transform'])

    config['dataset'].update({'transform': train_trans,
                              'target_transform': None})

    # dataset = pascal_voc_classification(**config['dataset'])
    dataset = classification(**config['dataset'])    
    config['data_loaders']['dataset'] = dataset
    # data_loader = fetch_voc(**config['data_loaders'])
    train_loader, eval_loader = data_loader(args, test_path=False)
    data_iters = iter(eval_loader)
    
    # inp, var
    raw_imgs = next(data_iters)[0]
    solver = Solver(config, epoch=19)

    # if args.train:
		# train_logger = SummaryWriter(log_dir = os.path.join(config['log'], 'train'), comment = 'training')
        # solver.train(data_loader, train_logger)
    if args.run_demo:
        # Load demo images and pre-computed object proposals
        # change the idx to test different samples
        idx = 15
        # try:
        #     raw_img = PIL.Image.open('./data/sample%d.png' % idx).convert('RGB')
        # except:
        #     raw_img = PIL.Image.open('./data/sample%d.jpg' % idx).convert('RGB')
        proposals=[]
        for i in range(10):
            proposals.append(np.genfromtxt('data/' + str(idx) + '/mask'+str(i)+'.csv', delimiter=','))

        for j in range(1):
            raw_img = raw_imgs[j]
            raw_img = raw_img.permute(1,2,0).numpy().astype(np.uint8)
            raw_img = PIL.Image.fromarray(raw_img)
            raw_size = raw_img.size
            input_var = test_trans(raw_img).unsqueeze(0).cuda().requires_grad_()
            # print(raw_size, input_var.shape)
            # with open('./data/sample%d.json' % idx, 'r') as f:
            # proposals = list(map(rle_decode, json.load(f)))
            # print(len(np.where(proposals[2] == 1)[0]))
            seg_res = solver.inference(input_var, raw_img, args.model_epoch, proposals=proposals,class_names=config['dataset']['class_names'])
            if seg_res is None:
                continue
            seg_res = cv2.resize(seg_res, raw_size)
            cv2.imwrite('data/sample%d_seg.png' % idx, seg_res * 255)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train','-T', type=bool, default=False, help='set train mode up, default False')
    parser.add_argument('--run_demo','-D', type=bool, default=True, help='run demo, default True')
    parser.add_argument('--conf','-C', type=str, default=CONFIG, help='config file, default VOC2012')
    parser.add_argument("--model_epoch", type=int, default=-1, help='model index to be loaded for demo, default using latest with -1')
    parser.add_argument("--batch_size", type=int, default=16)
    
    args = parser.parse_args()
    main(args)

'''
Select Top-k target object as postive images
'''

import sys
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.init as init
import numpy as np
import utils
import opts
from tqdm import tqdm

from dataset_vqacp_extract_img_id import Dictionary, VQAFeatureDataset

from UpDn import Model

@ torch.no_grad()
def evaluate(model, dataloader):
    model.train(True)
    num_data = len(dataloader.dataset) 
    image_id_to_topk_object_index = {}
    for i, (v, b, q, a, q_id, i_id) in enumerate(tqdm(dataloader)):
        v = v.cuda()
        b = b.cuda()
        q = q.cuda()
        pred, att = model(q, v, False)
        print(att.size())
        _, topk_img_idx = torch.topk(att.squeeze(), k=36, dim=-1, sorted=True)
        for index, q_idx in enumerate(q_id):
            if q_idx.item() not in image_id_to_topk_object_index:
                image_id_to_topk_object_index[q_idx.item()] = topk_img_idx[index].cpu().numpy().tolist()
            else:
                assert "duplicated image idx: {}".format(q_idx)
    assert num_data == len(image_id_to_topk_object_index), 'different number: num_data: {}, image_id number: {}'.format(num_data, len(image_id_to_topk_object_index))
    return image_id_to_topk_object_index    

if __name__ == '__main__':
    opt = opts.parse_opt()

    dictionary = Dictionary.load_from_file(opt.dataroot + 'dictionary.pkl')
    opt.ntokens = dictionary.ntoken

    model = Model(opt)
    model = model.cuda()
    # model.apply(weights_init_kn)
    print('loading %s' % opt.checkpoint_path)
    model_data = torch.load(opt.checkpoint_path)

    model = nn.DataParallel(model).cuda()
    model.load_state_dict(model_data.get('model_state', model_data))

    train_dset = VQAFeatureDataset('train', dictionary, opt.dataroot, opt.img_root, ratio=opt.ratio, adaptive=False)  # load labeld data

    train_loader = DataLoader(train_dset, opt.batch_size, shuffle=False, num_workers=1, collate_fn=utils.trim_collate)
    opt.use_all = 1

    image_id_to_image_idx = evaluate(model, train_loader)
    np.save(opt.output, image_id_to_image_idx)


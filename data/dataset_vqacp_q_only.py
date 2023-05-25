from __future__ import print_function
import os
import json
import _pickle as cPickle
import utils
import warnings

from torch.utils.data import Dataset


class VQAFeatureDataset(Dataset):
    def __init__(self, name, dictionary, dataroot, image_dataroot, ratio, adaptive=False):
        super(VQAFeatureDataset, self).__init__()
        self.questions = json.load(open(os.path.join(dataroot, 'vqacp_v2_train_questions.json')))

    def __getitem__(self, index):
        question_data = self.questions[index]
        question = question_data['question']
        question_id = question_data['question_id']
        
        return question, question_id

    def __len__(self):
        return len(self.questions)


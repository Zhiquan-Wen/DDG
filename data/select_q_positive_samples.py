import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import utils
import nlpaug.augmenter.word as naw
from dataset_vqacp_q_only import VQAFeatureDataset

# from nlpaug.util import Action
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel, pipeline
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--input', required=True, help='path to the train question')
    args = parser.parse_args()

    questions = json.load(open(os.path.join(args.input, 'vqacp_v2_train_questions.json'), 'r'))
    questions = sorted(questions, key=lambda x: x['question_id'])
    aug_questions = json.load(open(os.path.join(args.input, 'vqacp_v2_train_aug_questions.json'), 'r'))
    aug_questions = sorted(aug_questions, key=lambda x: x['question_id'])
    print('finish load questions !')


    aug = naw.SynonymAug(aug_src='ppdb', model_path=os.path.join(args.input, 'ppdb-2.0-tldr'))
    print('initial synonymaug model')


    augment_data = []
    count = 0
    for q, aug_q in tqdm(zip(questions, aug_questions)):
        assert q['question_id'] == aug_q['question_id'], 'mismatch question id'
        sentences = aug_q['question']
        target = None
        for idx, i in enumerate(sentences):
            if ' - ' in i:
                continue
            if i == q['question']:
                continue
            target = i

        if target is None:
            target = aug.augment(q['question'])
            count += 1 
            print('augment data')

        augment_data.append({'question_id': q['question_id'], 'question': target})
    print(count)
    with open(os.path.join(args.output, 'vqacp_v2_train_aug_questions_second.json'), 'w') as f:
        json.dump(augment_data, f)

if __name__ == '__main__':
    main()




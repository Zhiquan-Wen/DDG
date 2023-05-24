"""
This code is modified from Hengyuan Hu's repository.
https://github.com/hengyuan-hu/bottom-up-attention-vqa
"""
from __future__ import print_function
import os
import json
import _pickle as cPickle
import numpy as np
import utils
import warnings
import pdb
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    # import h5py
from xml.etree.ElementTree import parse
import torch
from torch.utils.data import Dataset
# import zarr
import random
from mio import MioWriter, MIO
import struct
COUNTING_ONLY = False


# Following Trott et al. (ICLR 2018)
#   Interpretable Counting for Visual Question Answering
def is_howmany(q, a, label2ans):
    if 'how many' in q.lower() or \
            ('number of' in q.lower() and 'number of the' not in q.lower()) or \
                    'amount of' in q.lower() or \
                    'count of' in q.lower():
        if a is None or answer_filter(a, label2ans):
            return True
        else:
            return False
    else:
        return False


def answer_filter(answers, label2ans, max_num=10):
    for ans in answers['labels']:
        if label2ans[ans].isdigit() and max_num >= int(label2ans[ans]):
            return True
    return False


class Dictionary(object):
    def __init__(self, word2idx=None, idx2word=None):
        if word2idx is None:
            word2idx = {}
        if idx2word is None:
            idx2word = []
        self.word2idx = word2idx
        self.idx2word = idx2word

    @property
    def ntoken(self):
        return len(self.word2idx)

    @property
    def padding_idx(self):
        return len(self.word2idx)

    def tokenize(self, sentence, add_word):
        sentence = sentence.lower()
        sentence = sentence.replace(',', '').replace('?', '').replace('\'s', ' \'s')
        words = sentence.split()
        tokens = []
        if add_word:
            for w in words:
                tokens.append(self.add_word(w))
        else:
            for w in words:
                # the least frequent word (`bebe`) as UNK for Visual Genome dataset
                tokens.append(self.word2idx.get(w, self.padding_idx - 1))
        return tokens

    def dump_to_file(self, path):
        cPickle.dump([self.word2idx, self.idx2word], open(path, 'wb'))
        print('dictionary dumped to %s' % path)

    @classmethod
    def load_from_file(cls, path):
        print('loading dictionary from %s' % path)
        word2idx, idx2word = cPickle.load(open(path, 'rb'))
        d = cls(word2idx, idx2word)
        return d

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


def _create_entry(img, question, answer):
    if None != answer:
        answer.pop('image_id')
        answer.pop('question_id')
    entry = {
        'question_id': question['question_id'],
        'image_id': question['image_id'],
        'image': img,
        'question': question['question'],
        'train': False,
        'answer': answer}
    return entry

def _create_entry_aug(img, question, aug_question, answer):
    if None != answer:
        answer.pop('image_id')
        answer.pop('question_id')
    entry = {
        'question_id': question['question_id'],
        'image_id': question['image_id'],
        'image': img,
        'question': question['question'],
        'aug_question': aug_question['question'],
        'train': True,
        'answer': answer}
    return entry


def _load_dataset(dataroot, name, label2ans,ratio=1.0):
    """Load entries

    img_id2val: dict {img_id -> val} val can be used to retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'test'
    """
    question_path = os.path.join(dataroot, 'vqacp_v2_%s_questions.json' % (name))
    questions = sorted(json.load(open(question_path, 'r')), key=lambda x: x['question_id'])
    if name == 'train':
        aug_question_path = os.path.join(dataroot, 'vqacp_v2_%s_aug_questions_second.json' % (name))
        aug_questions = sorted(json.load(open(aug_question_path, 'r')), key=lambda x: x['question_id'])

    # train, val
    answer_path = os.path.join(dataroot, 'cache', '%s_target.pkl' % name)
    answers = cPickle.load(open(answer_path, 'rb'))
    answers = sorted(answers, key=lambda x: x['question_id'])[0:len(questions)]

    utils.assert_eq(len(questions), len(answers))

    if ratio < 1.0:
        # sampling traing instance to construct smaller training set.
        index = random.sample(range(0,len(questions)), int(len(questions)*ratio))
        questions_new = [questions[i] for i in index]
        answers_new = [answers[i] for i in index]
        if name == 'train':
            aug_questions_new = [aug_questions[i] for i in index]
    else:
        questions_new = questions
        answers_new = answers
        if name == 'train':
            aug_questions_new = aug_questions

    entries = []
    if name == 'train':
        for question, aug_question, answer in zip(questions_new, aug_questions_new, answers_new):
            utils.assert_eq(question['question_id'], answer['question_id'])
            utils.assert_eq(question['image_id'], answer['image_id'])
            img_id = question['image_id']
            if not COUNTING_ONLY or is_howmany(question['question'], answer, label2ans):
                entries.append(_create_entry_aug(img_id, question, aug_question, answer))
    else:
        for question, answer in zip(questions_new, answers_new):
            utils.assert_eq(question['question_id'], answer['question_id'])
            utils.assert_eq(question['image_id'], answer['image_id'])
            img_id = question['image_id']
            if not COUNTING_ONLY or is_howmany(question['question'], answer, label2ans):
                entries.append(_create_entry(img_id, question, answer))

    return entries


class VQAFeatureDataset(Dataset):
    def __init__(self, name, dictionary, dataroot, img_root, ratio, adaptive=False):
        super(VQAFeatureDataset, self).__init__()
        assert name in ['train', 'test']

        ans2label_path = os.path.join(dataroot, 'cache', 'train_test_ans2label.pkl')
        label2ans_path = os.path.join(dataroot, 'cache', 'train_test_label2ans.pkl')
        self.ans2label = cPickle.load(open(ans2label_path, 'rb'))
        self.label2ans = cPickle.load(open(label2ans_path, 'rb'))
        if name == 'train':
            self.qid_to_img_idx = np.load(os.path.join(dataroot, 'q_id_with_sorted_object_index.npy'), allow_pickle=True).item()
        self.num_ans_candidates = len(self.ans2label)

        self.dictionary = dictionary
        self.adaptive = adaptive

        print('loading image features in MIO')
        # Load image features
        self.m = MIO(img_root)
        print('loading image features in MIO done!')

        # extract the corresponding image ids
        self.ids = {}
        for i in range(self.m.size):
            id_= struct.unpack("<I", self.m.get_collection_metadata(i))[0]
            self.ids[id_] = i

        self.entries = _load_dataset(dataroot, name, self.label2ans, ratio)
        self.tokenize()
        self.tensorize()

        self.v_dim = 2048

    def tokenize(self, max_length=14):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        for entry in self.entries:
            tokens = self.dictionary.tokenize(entry['question'], False)
            tokens = tokens[:max_length]

            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = tokens + padding

            if entry['train']:
                tokens_pos = self.dictionary.tokenize(entry['aug_question'], False)
                tokens_pos = tokens_pos[:max_length]
                if len(tokens_pos) < max_length:
                    padding = [self.dictionary.padding_idx] * (max_length - len(tokens_pos))
                    tokens_pos = tokens_pos + padding
                
                utils.assert_eq(len(tokens_pos), max_length)
    
            utils.assert_eq(len(tokens), max_length)
            entry['q_token'] = tokens
            if entry['train']:
                entry['q_pos_token'] = tokens_pos

    def tensorize(self):
        for entry in self.entries:
            question = torch.from_numpy(np.array(entry['q_token']))
            entry['q_token'] = question

            if entry['train']:
                question_pos = torch.from_numpy(np.array(entry['q_pos_token']))
                entry['q_pos_token'] = question_pos

            answer = entry['answer']
            if None != answer:
                labels = np.array(answer['labels'])
                scores = np.array(answer['scores'], dtype=np.float32)
                if len(labels):
                    labels = torch.from_numpy(labels)
                    scores = torch.from_numpy(scores)
                    entry['answer']['labels'] = labels
                    entry['answer']['scores'] = scores
                else:
                    entry['answer']['labels'] = None
                    entry['answer']['scores'] = None

    def __getitem__(self, index):
        entry = self.entries[index]
        img_id = entry['image_id']
        true_feature_id = self.ids[img_id]
        content_bytes = self.m.fetchone(true_feature_id)
        features = torch.from_numpy(np.frombuffer(content_bytes, dtype=np.float32).reshape(2048, 36)).permute(1, 0)

        question = entry['q_token']
        question_id = entry['question_id']
        answer = entry['answer']
        if entry['train']:
            aug_question = entry['q_pos_token']
            aug_img_idx = torch.LongTensor(self.qid_to_img_idx[question_id])
        if None != answer:
            labels = answer['labels']
            scores = answer['scores']
            target = torch.zeros(self.num_ans_candidates)
            if labels is not None:
                target.scatter_(0, labels, scores)
            # return features, spatials, question, target, question_id
            if entry['train']:
                return features, torch.FloatTensor([0]), question, aug_question, aug_img_idx, target, question_id
            else:
                return features, torch.FloatTensor([0]), question, target, question_id
        else:
            # return features, spatials, question, question_id
            return features, torch.FloatTensor([0]), question, question_id

    def __len__(self):
        return len(self.entries)
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import utils
# import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
# import nlpaug.augmenter.sentence as nas
# import nlpaug.flow as nafc
from dataset_vqacp_q_only import VQAFeatureDataset

# from nlpaug.util import Action
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel, pipeline

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--input', required=True, help='path to the train question')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer_de_en = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")
    model_de_en = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-de-en").to(device)
    print("initial de to en model !")

    tokenizer_en_fr = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
    model_en_fr = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-fr").to(device)
    en_fr_pipeline = pipeline('translation', model=model_en_fr, tokenizer=tokenizer_en_fr, device=0)
    print("initial en to fr model")

    tokenizer_en_de = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
    model_en_de = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-de").to(device)
    en_de_pipeline = pipeline('translation', model=model_en_de, tokenizer=tokenizer_en_de, device=0)
    print("initial en to de model")

    tokenizer_fr_en = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
    model_fr_en = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-fr-en").to(device)
    print("initial fr to en model")

    tokenizer_similarity = AutoTokenizer.from_pretrained('hiiamsid/sentence_similarity_spanish_es')
    model_similarity = AutoModel.from_pretrained('hiiamsid/sentence_similarity_spanish_es').to(device)
    print('initial similarity model')

    train_dset = VQAFeatureDataset('train', None, args.input, None, ratio=1.0, adaptive=False)

    loader = DataLoader(train_dset, 256, shuffle=False, num_workers=1, collate_fn=utils.trim_collate)
    print('finish load dataset')

    augment_data = []
    count = 0
    for q, qid in tqdm(loader):

        total_output = [[i] for i in q]
        question = list(q)
        en_to_fr = en_fr_pipeline(question)
        batch_en_to_fr = [i['translation_text'] for i in en_to_fr]
        en_to_de = en_de_pipeline(question)
        batch_en_to_de = [i['translation_text'] for i in en_to_de]

        fr_to_en_input_ids = tokenizer_fr_en(batch_en_to_fr, padding=True, truncation=True, return_tensors="pt").input_ids.to(device)
        fr_to_en_outputs = model_fr_en.generate(input_ids=fr_to_en_input_ids, num_beams=5, num_return_sequences=5)
        output_fr_to_en = tokenizer_fr_en.batch_decode(fr_to_en_outputs, skip_special_tokens=True) # batch * 5

        output_fr_to_en_split = [i for i in chunks(output_fr_to_en, 5)]

        de_to_en_input_ids = tokenizer_de_en(batch_en_to_de, padding=True, truncation=True, return_tensors="pt").input_ids.to(device)
        de_to_en_outputs = model_de_en.generate(input_ids=de_to_en_input_ids, num_beams=5, num_return_sequences=5)
        output_de_to_en = tokenizer_de_en.batch_decode(de_to_en_outputs, skip_special_tokens=True)  # batch * 5

        output_de_to_en_split = [i for i in chunks(output_de_to_en, 5)]

        # data_aug = aug.augment(question)  # batch

        # total_output = [list(set(i+j+k+[l])) for i,j,k,l in zip(total_output, output_fr_to_en_split, output_de_to_en_split, data_aug)]
        total_output = [list(set(i+j+k)) for i,j,k in zip(total_output, output_fr_to_en_split, output_de_to_en_split)]

        for output, idx in zip(total_output, qid):
            if len(output) == 1:
                augment_data.append({'question_id': idx, 'question': output})
                continue

            encoded_input = tokenizer_similarity(output, padding=True, truncation=True, return_tensors='pt').to(device)
            with torch.no_grad():
                model_output = model_similarity(**encoded_input)
            sentences_embedding = mean_pooling(model_output, encoded_input['attention_mask'])
            similarity = torch.nn.functional.cosine_similarity(sentences_embedding[0], sentences_embedding[1:], -1)
            
            if len(similarity) < 3:
                augment_question = output[1:]
            else:
                _, max_index = torch.topk(similarity, k=3)
                true_index = max_index + 1
                augment_question = [output[i] for i in true_index]

            augment_data.append({'question_id': idx.item(), 'question': augment_question})

        print('original: ', q[-3:], 'augment: ', augment_data[-3:])

    with open(os.path.join(args.output, 'vqacp_v2_train_aug_questions.json'), 'w') as f:
        json.dump(augment_data, f)

if __name__ == "__main__":
    main()
    





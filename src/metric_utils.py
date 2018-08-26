import cPickle as pickle
import os
import sys
from metric_bleu_utils import Bleu
from metric_rouge_utils import Rouge

def score_all(ref, hypo):
    scorers = [
        (Bleu(4),["Bleu_1","Bleu_2","Bleu_3","Bleu_4"]),
        (Rouge(),"ROUGE_L"),
    ]
    final_scores = {}
    for scorer,method in scorers:
        score,scores = scorer.compute_score(ref,hypo)
        if type(score)==list:
            for m,s in zip(method,score):
                final_scores[m] = s
        else:
            final_scores[method] = score

    return final_scores

def score(ref, hypo):
    scorers = [
        (Bleu(4),["Bleu_1","Bleu_2","Bleu_3","Bleu_4"])

    ]
    final_scores = {}
    for scorer,method in scorers:
        score,scores = scorer.compute_score(ref,hypo)
        if type(score)==list:
            for m,s in zip(method,score):
                final_scores[m] = s
        else:
            final_scores[method] = score

    return final_scores

def evaluate_captions(ref,cand):
    hypo = {}
    refe = {}
    for i, caption in enumerate(cand):
        hypo[i] = [caption,]
        refe[i] = ref[i]
    final_scores = score(refe, hypo)
    return 1*final_scores['Bleu_4'] + 1*final_scores['Bleu_3'] + 0.5*final_scores['Bleu_1'] + 0.5*final_scores['Bleu_2']

def evaluate(data_path='./data', split='val', get_scores=False):
    reference_path = os.path.join(data_path, "%s/%s.references.pkl" %(split, split))
    candidate_path = os.path.join(data_path, "%s/%s.candidate.captions.pkl" %(split, split))

    # load caption data
    with open(reference_path, 'rb') as f:
        ref = pickle.load(f)
    with open(candidate_path, 'rb') as f:
        cand = pickle.load(f)

    # make dictionary
    hypo = {}
    for i, caption in enumerate(cand):
        hypo[i] = [caption]

    # compute bleu score
    final_scores = score_all(ref, hypo)

    # print out scores
    print 'Bleu_1:\t',final_scores['Bleu_1']
    print 'Bleu_2:\t',final_scores['Bleu_2']
    print 'Bleu_3:\t',final_scores['Bleu_3']
    print 'Bleu_4:\t',final_scores['Bleu_4']
    print 'METEOR:\t',final_scores['METEOR']
    print 'ROUGE_L:',final_scores['ROUGE_L']
    print 'CIDEr:\t',final_scores['CIDEr']

    if get_scores:
        return final_scores

def my_evaluate(f1, f2, get_scores=False):
    

    # load data
    with open(f1, 'r') as f1:
        cand = f1.readlines()
    with open(f2, 'r') as f2:
        ref = f2.readlines()

    # make dictionary
    hypo = {}
    refs = {}
    for i, caption in enumerate(cand):
        hypo[i] = [" ".join(caption.strip().split()[:-1])]
    for i, caption in enumerate(ref):
        refs[i] = [" ".join(caption.strip().split()[:-1])]

    # compute bleu score
    final_scores = score_all(refs, hypo)

    # print out scores
    print('Bleu_1:\t',final_scores['Bleu_1'])
    print('Bleu_2:\t',final_scores['Bleu_2'])
    print('Bleu_3:\t',final_scores['Bleu_3'])
    print('Bleu_4:\t',final_scores['Bleu_4'])
    # print('METEOR:\t',final_scores['METEOR'])
    print('ROUGE_L:',final_scores['ROUGE_L'])
    # print('CIDEr:\t',final_scores['CIDEr'])

if __name__ == "__main__":
    f1 = "./data/test.result.pred"
    # f2 = "./data/test.result3.ref"
    f2 = "./test.result.ref"
    my_evaluate(f1, f2)



















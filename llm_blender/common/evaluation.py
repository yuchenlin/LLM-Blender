
import psutil
import os
import numpy as np
import spacy
import bert_score
import torch
import gc
from copy import deepcopy
from evaluate import load
from sacrebleu import sentence_bleu, corpus_bleu
from nltk import word_tokenize
from typing import List, Optional, Union, Dict, Tuple
from absl import logging
from torch import split
from tqdm import tqdm
from nltk import sent_tokenize
from rouge_score import rouge_scorer
from tqdm.contrib.concurrent import process_map
from pycocoevalcap.spice.spice import Spice
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
logging.set_verbosity(logging.WARNING)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

SUPPORTED_METRICS = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum', 'bleu', 'bleurt', "cider", "spice", "bleu4", "bertscore", "bartscore"]
METRIC_WEIGHTS = {
    "rouge1": 1.0,
    "rouge2": 1.0,
    "rougeL": 1.0,
    "rougeLsum": 1.0,
    "bleu": 0.01,
    "bleu4": 0.01,
    "bleurt": 1.0,
    "cider": 0.01,
    "spice": 0.01,
    "bertscore": 1.0,
    "bartscore": 1.0,
    "gpt4": 1.0, # custom
} # scale to 0-1

def pre_rouge_processing(summary):
    summary = summary.replace("<n>", " ")
    summary = "\n".join(sent_tokenize(summary))
    return summary

def eval_rouge(
    hypotheses: List[List[str]],
    references: List[List[str]],
    rouge_types: List[str]=['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
    ) -> Dict[str, float]:
    """
    Evaluate the hypothesis and reference using the metric.

    Args:
        hypotheses: the hypotheses
        references: the references
        rouge_types: the rouge types to be used.

    Returns:
        A dict of rouge scores.
        key is the rouge type, value is the rouge score, in same shape with hypotheses.
    """
    assert len(hypotheses) == len(references)
    assert set(rouge_types) <= set(['rouge1', 'rouge2', 'rougeL', 'rougeLsum']), "Rouge types should be in ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']"
    scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True, split_summaries=True)
    rouge_scores = {rouge_type: [[] for _ in range(len(hypotheses))] for rouge_type in rouge_types}
    with tqdm(total=len(hypotheses), desc="Evaluating rouge") as pbar:
        for i, (hypo_group, ref) in enumerate(zip(hypotheses, references)):
            for hypo in hypo_group:
                scores = scorer.score_multi(ref, pre_rouge_processing(hypo))
                for rouge_type in rouge_types:
                    rouge_scores[rouge_type][i].append(scores.get(rouge_type).fmeasure)
            pbar.update(1)
    return rouge_scores

def eval_bleu(
    hypotheses: List[List[str]],
    references: List[List[str]],
    ) -> List[float]:
    """
    Evaluate the hypothesis and reference using the metric.

    Args:
        hypotheses: the hypotheses
        references: the references

    Returns:
        A list of bleu scores, in same shape with hypotheses.
    """
    assert len(hypotheses) == len(references), f"Length of hypotheses {len(hypotheses)} and references {len(references)} should be the same."
    bleu_scores = []
    with tqdm(total=len(hypotheses), desc="Evaluating bleu") as pbar:
        for i, (hypo_group, ref) in enumerate(zip(hypotheses, references)):
            bleu_scores.append([])
            for hypo in hypo_group:
                bleu_scores[i].append(sentence_bleu(hypo, ref).score)
            pbar.update(1)
    return bleu_scores

def eval_bleurt(
    hypotheses: List[List[str]],
    references: List[List[str]]
    ) -> List[float]:
    """
    Evaluate the hypothesis and reference using the metric.

    Args:
        hypotheses: the hypotheses
        references: the references
    """
    torch.cuda.empty_cache()
    assert len(hypotheses) == len(references)
    bleurt_scorer = load('bleurt')
    bleurt_scores = []
    with tqdm(total=len(hypotheses), desc="Evaluating bleurt") as pbar:
        for i, (hypo_group, ref) in enumerate(zip(hypotheses, references)):
            bleurt_scores.append([])
            for hypo in hypo_group:
                result = bleurt_scorer.compute(predictions=[hypo], references=ref)
                bleurt_scores[i].append(result['scores'][0])
            pbar.update(1)
    del bleurt_scorer
    return bleurt_scores

# from bart_score import BARTScorer
# >>> bart_scorer = BARTScorer(device='cuda:0', checkpoint='facebook/bart-large-cnn')
# >>> bart_scorer.load(path='bart.pth')
# >>> bart_scorer.score(['This is interesting.'], ['This is fun.'], batch_size=4)
# [out]
# [-2.336203098297119]

def eval_bartscore(
    hypotheses: List[List[str]],
    references: List[List[str]]
    ) -> List[float]:
    """
    Evaluate the hypothesis and reference using the metric.

    Args:
        hypotheses: the hypotheses
        references: the references
    """
    torch.cuda.empty_cache()
    assert len(hypotheses) == len(references)
    from bart_score import BARTScorer
    bart_scorer = BARTScorer(device='cuda:0', checkpoint='facebook/bart-large-cnn')
    if not os.path.exists(os.path.join(os.path.dirname(__file__), 'bart_score.pth')):
        print("bart_score.pth trained on ParaBank not found.")
        print("Please download bart_score.pth from bartscore github repo, then put it here: ", os.path.join(os.path.dirname(__file__), 'bart_score.pth'))
        print("Using the default bart-large-cnn model instead.")
    else:
        bart_scorer.load(path=os.path.join(os.path.dirname(__file__), 'bart_score.pth'))
    bart_scores = []
    with tqdm(total=len(hypotheses), desc="Evaluating bartscore") as pbar:
        for i, (hypo_group, ref) in enumerate(zip(hypotheses, references)):
            bart_scores.append(
                bart_scorer.score(hypo_group, ref*len(hypo_group), batch_size=4)
            )
            pbar.update(1)
            assert len(bart_scores[i]) == len(hypo_group)
    del bart_scorer
    return bart_scores

def eval_bleu4(
    hypotheses: List[List[str]],
    references: List[List[str]],
    ) -> List[float]:
    """
    Evaluate the hypothesis and reference using the metric.

    Args:
        hypotheses: the hypotheses
        references: the references

    Returns:
        A list of bleu scores, in same shape with hypotheses.
    """
    print("Evaluating bleu4")
    assert len(hypotheses) == len(references)
    # tokenization
    nlp = spacy.load("en_core_web_sm")
    disable_pipes = list(nlp.pipe_names)
    disable_pipes.remove('tagger')
    nlp.disable_pipes(*disable_pipes)
    for i in tqdm(range(len(hypotheses)), desc="Tokenizing"):
        for j in range(len(hypotheses[i])):
            hypotheses[i][j] = " ".join([token.text for token in nlp(hypotheses[i][j])])
        for j in range(len(references[i])):
            references[i][j] = " ".join([token.text for token in nlp(references[i][j])])

    bleu4_scorer = Bleu(4)
    gts = {}
    res = {}
    hypo_ids_per_ref = []
    id = 0

    for i, (hypo_group, ref) in enumerate(zip(hypotheses, references)):
        hypo_ids_per_ref.append([])
        for hypo in hypo_group:
            gts[id] = ref
            res[id] = [hypo]
            hypo_ids_per_ref[i].append(id)
            id += 1

    score, scores = bleu4_scorer.compute_score(gts, res)
    for method in zip(("Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"), score):
        print("%s: %0.3f" % method)
    bleu4_scores = scores[3]
    bleu4_scores = [[bleu4_scores[hypo_id]*100 for hypo_id in hypo_ids] for hypo_ids in hypo_ids_per_ref]
    return bleu4_scores


def eval_cider(
    hypotheses: List[List[str]],
    references: List[List[str]],
    ) -> List[float]:
    """
    Evaluate the hypothesis and reference using the metric.

    Args:
        hypotheses: the hypotheses
        references: the references
    """
    print("Evaluating cider")
    assert len(hypotheses) == len(references)

    # tokenization
    nlp = spacy.load("en_core_web_sm")
    disable_pipes = list(nlp.pipe_names)
    disable_pipes.remove('tagger')
    nlp.disable_pipes(*disable_pipes)
    for i in tqdm(range(len(hypotheses)), desc="Tokenizing"):
        for j in range(len(hypotheses[i])):
            hypotheses[i][j] = " ".join([token.text for token in nlp(hypotheses[i][j])])
        for j in range(len(references[i])):
            references[i][j] = " ".join([token.text for token in nlp(references[i][j])])

    cider_scorer = Cider()
    gts = {}
    res = {}
    hypo_ids_per_ref = []
    id = 0

    for i, (hypo_group, ref) in enumerate(zip(hypotheses, references)):
        hypo_ids_per_ref.append([])
        for hypo in hypo_group:
            gts[id] = ref
            res[id] = [hypo]
            hypo_ids_per_ref[i].append(id)
            id += 1

    score, scores = cider_scorer.compute_score(gts, res)
    cider_scores = [[scores[hypo_id]*10 for hypo_id in hypo_ids] for hypo_ids in hypo_ids_per_ref]
    return cider_scores

def eval_bertscore(
    hypotheses: List[List[str]],
    references: List[List[str]],
    model_type="bert-base-multilingual-cased",
    lang="en",
    ) -> List[float]:
    """
    Evaluate the hypothesis and reference using bertscore.
    BertScore officially recommends using microsoft/deberta-xlarge-mnli as the model.
    the default multilingual model is bert-base-multilingual-cased.

    Args:
        hypotheses: the hypotheses
        references: the references
    """
    torch.cuda.empty_cache()
    print("Evaluating bertscore")
    assert len(hypotheses) == len(references)
    hypotheses = np.array(hypotheses)
    references = np.array(references)
    scores = np.zeros_like(hypotheses, dtype=np.float32)
    for group_id in range(len(hypotheses[0])):
        print("Evaluating group %d" % group_id)
        hypo_group = hypotheses[:, group_id]
        P, R, F1 = bert_score.score(hypo_group.tolist(), references.tolist(), lang=lang, verbose=True, model_type=model_type, batch_size=16)
        scores[:, group_id] = F1.numpy()
    gc.collect()
    torch.cuda.empty_cache()
    return scores.tolist()

def eval_spice(
    hypotheses: List[List[str]],
    references: List[List[str]]
    ) -> List[float]:
    """
    Evaluate the hypothesis and reference using the metric.

    Args:
        hypotheses: the hypotheses
        references: the references
    """
    print("Evaluating spice")
    assert len(hypotheses) == len(references)
    # tokenization
    nlp = spacy.load("en_core_web_sm")
    disable_pipes = list(nlp.pipe_names)
    disable_pipes.remove('tagger')
    nlp.disable_pipes(*disable_pipes)
    for i in tqdm(range(len(hypotheses)), desc="Tokenizing"):
        for j in range(len(hypotheses[i])):
            hypotheses[i][j] = " ".join([token.text for token in nlp(hypotheses[i][j])])
        for j in range(len(references[i])):
            references[i][j] = " ".join([token.text for token in nlp(references[i][j])])

    spice_scorer = Spice()
    gts = {}
    res = {}
    hypo_ids_per_ref = []
    id = 0
    for i, (hypo_group, ref) in enumerate(zip(hypotheses, references)):
        hypo_ids_per_ref.append([])
        for hypo in hypo_group:
            gts[id] = ref
            res[id] = [hypo]
            hypo_ids_per_ref[i].append(id)
            id += 1

    score, scores = spice_scorer.compute_score(gts, res)
    spice_scores = [[scores[hypo_id]['All']['f']*100.0 for hypo_id in hypo_ids] for hypo_ids in hypo_ids_per_ref]
    return spice_scores



def compute_new_n_gram(source:str, candidate:str):
    """
        computer the new n-grams in the candidate compared to source text
    """
    # text
    text = source.lower()
    text_words = word_tokenize(text)
    text_bigrams = [[text_words[j], text_words[j + 1]] for j in range(len(text_words) - 1)]
    text_trigrams = [[text_words[j], text_words[j + 1], text_words[j + 2]] for j in range(len(text_words) - 2)]
    text_quadrigrams = [[text_words[j], text_words[j + 1], text_words[j + 2], text_words[j + 3]] for j in range(len(text_words) - 3)]

    # candidate
    candidate = candidate.lower().replace("<n>", " ")
    candidate_words = word_tokenize(candidate)

    unigrams, bigrams, trigrams, quadrigrams = 0, 0, 0, 0
    for j in range(len(candidate_words)):
        if not(candidate_words[j] in text_words):
            unigrams += 1
        if j < len(candidate_words) - 1:
            bigram = [candidate_words[j], candidate_words[j + 1]]
            if not(bigram in text_bigrams):
                bigrams += 1
        if j < len(candidate_words) - 2:
            trigram = [candidate_words[j], candidate_words[j + 1], candidate_words[j + 2]]
            if not(trigram in text_trigrams):
                trigrams += 1
        if j < len(candidate_words) - 3:
            quadrigram = [candidate_words[j], candidate_words[j + 1], candidate_words[j + 2], candidate_words[j + 3]]
            if not(quadrigram in text_quadrigrams):
                quadrigrams += 1
    new_unigram, new_bigram, new_trigram, new_quadrigram = 0, 0, 0, 0
    if len(candidate_words) > 0:
        new_unigram = unigrams / (len(candidate_words) - 0)
    if len(candidate_words) > 1:
        new_bigram = bigrams / (len(candidate_words) - 1)
    if len(candidate_words) > 2:
        new_trigram = trigrams / (len(candidate_words) - 2)
    if len(candidate_words) > 3:
        new_quadrigram = quadrigrams / (len(candidate_words) - 3)
    return new_unigram, new_bigram, new_trigram, new_quadrigram


def eval_novel_n_gram(
    sources: List[str],
    hypotheses: Union[List[List[str]], List[str]],
    ) -> List[float]:
    """
        evaluate the novel n-gram in the hypotheses compared to the origianl soiurce
    """
    print("Evaluating novel n-gram")
    assert len(hypotheses) == len(sources)
    for i in range(len(hypotheses)):
        if isinstance(hypotheses[i], str):
            hypotheses[i] = [hypotheses[i]]

    new_unigrams, new_bigrams, new_trigrams, new_quadrigrams = [], [], [], []
    for i, (source, hypo_group) in tqdm(enumerate(zip(sources, hypotheses)), desc="evaluate novel n-grams"):
        new_unigrams.append([])
        new_bigrams.append([])
        new_trigrams.append([])
        new_quadrigrams.append([])
        for hypo in hypo_group:
            new_unigram, new_bigram, new_trigram, new_quadrigram = \
                compute_new_n_gram(source, hypo)
            new_unigrams[i].append(new_unigram)
            new_bigrams[i].append(new_bigram)
            new_trigrams[i].append(new_trigram)
            new_quadrigrams[i].append(new_quadrigram)

    new_unigrams = np.array(new_unigrams)
    m_uni = 100 * np.mean(new_unigrams)
    new_bigrams = np.array(new_bigrams)
    m_bi = 100 * np.mean(new_bigrams)
    new_trigrams = np.array(new_trigrams)
    m_tri = 100 * np.mean(new_trigrams)
    new_quadrigrams = np.array(new_quadrigrams)
    m_quadri = 100 * np.mean(new_quadrigrams)
    print("New unigrams: {:.2f}, bigrams: {:.2f}, trigrams: {:.2f}, quadrigrams: {:.2f}".format(m_uni, m_bi, m_tri, m_quadri))
    # nested remove list with single element
    if all([len(score) == 1 for score in new_unigrams]):
        new_unigrams = [score[0] for score in new_unigrams]
    if all([len(score) == 1 for score in new_bigrams]):
        new_bigrams = [score[0] for score in new_bigrams]
    if all([len(score) == 1 for score in new_trigrams]):
        new_trigrams = [score[0] for score in new_trigrams]
    if all([len(score) == 1 for score in new_quadrigram]):
        new_quadrigram = [score[0] for score in new_quadrigram]
    return new_unigrams, new_bigrams, new_trigrams, new_quadrigrams

def eval_distinct_n_grams(texts:Union[List[List[str]], List[str]]):
    print("evaluating distinct n-grams")
    for i in range(len(texts)):
        if isinstance(texts[i], str):
            texts[i] = [texts[i]]

    uni_unigrams, uni_bigrams, uni_trigrams, uni_quadrigrams = [], [], [], []
    for i, text_group in tqdm(enumerate(texts), desc='evaluting distinct n-grams'):
        unigrams = []
        bigrams = []
        trigrams = []
        quadrigrams = []
        for text in text_group:
            text = text.lower()
            text_words = word_tokenize(text)
            text_bigrams = [(text_words[j], text_words[j + 1]) for j in range(len(text_words) - 1)]
            text_trigrams = [(text_words[j], text_words[j + 1], text_words[j + 2]) for j in range(len(text_words) - 2)]
            text_quadrigrams = [(text_words[j], text_words[j + 1], text_words[j + 2], text_words[j + 3]) for j in range(len(text_words) - 3)]
            unigrams.extend(text_words)
            bigrams.extend(text_bigrams)
            trigrams.extend(text_trigrams)
            quadrigrams.extend(text_quadrigrams)
        unigrams = set(unigrams)
        bigrams = set(unigrams)
        trigrams = set(trigrams)
        quadrigrams = set(quadrigrams)
        uni_unigrams.append(len(unigrams))
        uni_bigrams.append(len(bigrams))
        uni_trigrams.append(len(trigrams))
        uni_quadrigrams.append(len(quadrigrams))
    print(f"Mean unique 1-grams: {np.mean(uni_unigrams)}")
    print(f"Mean unique 2-grams: {np.mean(uni_bigrams)}")
    print(f"Mean unique 3-grams: {np.mean(uni_trigrams)}")
    print(f"Mean unique 4-grams: {np.mean(uni_quadrigrams)}")
    return uni_unigrams, uni_bigrams, uni_trigrams, uni_quadrigrams

def eval_self_bleu(texts:List[List[str]]):
    print("evaluating self bleu")
    for i in range(len(texts)):
        assert isinstance(texts[i], list)

    self_bleus = []
    for i, text_group in tqdm(enumerate(texts), desc='evaluting distinct n-grams'):
        group_self_bleus = []
        for j in range(len(text_group)):
            hypo = text_group[j]
            refs = text_group[:j] + text_group[j+1:]
            group_self_bleus.append(sentence_bleu(hypothesis=hypo, references=refs).score)
        self_bleus.append(np.mean(group_self_bleus))
    print(f"self BLEUs mean: {np.mean(self_bleus)}")
    return self_bleus

def _overall_eval_multi_process(data):
    candidates, targets, metrics = data
    s = psutil.Process(os.getpid())
    cpu_id = s.cpu_num()
    print("Worker {} is evaluating".format(cpu_id))
    return overall_eval(candidates, targets, metrics)

def _overall_eval(candidates, targets, metrics:List[str]):
    do_flatten = False
    # deepcopy in case it will make change to the passed in candidates and targets
    candidates = deepcopy(candidates)
    targets = deepcopy(targets)
    assert len(candidates) == len(targets), f"candidates and targets should have the same length, but got {len(candidates)} and {len(targets)}"
    # if there are no available targets, return None
    if all([target == '' for target in targets]) or \
        all([target == [] for target in targets]):
        return {
            metric: [
                [0 for _ in range(len(candidates[i]))]
                for i in range(len(candidates))]
            for metric in metrics
        }
    for i in range(len(candidates)):
        if isinstance(candidates[i], str):
            do_flatten = True
            candidates[i] = [candidates[i]]
        if isinstance(targets[i], str):
            targets[i] = [targets[i]]


    scores = {}
    rouge_tyeps = [metric for metric in metrics if metric.startswith('rouge')]
    if rouge_tyeps:
        _candidates, _targets = deepcopy(candidates), deepcopy(targets)
        rouge_scores = eval_rouge(_candidates, _targets, rouge_types=rouge_tyeps)
        scores.update(rouge_scores)
    if 'bleu' in metrics:
        _candidates, _targets = deepcopy(candidates), deepcopy(targets)
        bleu_scores = eval_bleu(_candidates, _targets)
        scores.update({'bleu': bleu_scores})
    if 'bleu4' in metrics:
        _candidates, _targets = deepcopy(candidates), deepcopy(targets)
        bleu4_scores = eval_bleu4(_candidates, _targets)
        scores.update({'bleu4': bleu4_scores})
    if 'bleurt' in metrics:
        _candidates, _targets = deepcopy(candidates), deepcopy(targets)
        bleurt_scores = eval_bleurt(_candidates, _targets)
        scores.update({'bleurt': bleurt_scores})
    if 'cider' in metrics:
        _candidates, _targets = deepcopy(candidates), deepcopy(targets)
        cider_scores = eval_cider(_candidates, _targets)
        scores.update({'cider': cider_scores})
    if 'spice' in metrics:
        _candidates, _targets = deepcopy(candidates), deepcopy(targets)
        spice_scores = eval_spice(_candidates, _targets)
        scores.update({'spice': spice_scores})
    if 'bartscore' in metrics:
        _candidates, _targets = deepcopy(candidates), deepcopy(targets)
        bartscore_scores = eval_bartscore(_candidates, _targets)
        scores.update({'bartscore': bartscore_scores})
    if 'bertscore' in metrics:
        _candidates, _targets = deepcopy(candidates), deepcopy(targets)
        bertscore_scores = eval_bertscore(_candidates, _targets)
        scores.update({'bertscore': bertscore_scores})
    if do_flatten:
        for metric in scores:
            assert all([len(score) == 1 for score in scores[metric]])
            scores[metric] = [score[0] for score in scores[metric]]
    return scores

def overall_eval(
    candidates:Union[List[List[str]], List[str]],
    targets: Union[List[str], List[List[str]]],
    metrics:List[str],
    num_workers:int=1
    ) -> Dict[str, List[float]]:
    """
    Args:
        candidates: the candidates
        targets: the targets
        metrics: the metrics to be evaluated
        num_workers: the number of workers to be used
    Return:
        A dict of scores, same shape with candidates for each metric
    """
    if num_workers > 1:
        cpu_num = psutil.cpu_count(logical=False)
        num_workers = min(num_workers, cpu_num)
        print("Using {} workers to evaluate".format(num_workers))
        chunk_size = len(candidates) // num_workers + 1
        candidates_chunks = [candidates[i:i + chunk_size] for i in range(0, len(candidates), chunk_size)]
        targets_chunks = [targets[i:i + chunk_size] for i in range(0, len(targets), chunk_size)]
        datas = [(candidates_chunks[i], targets_chunks[i], metrics) for i in range(len(candidates_chunks))]
        scores_chunks = process_map(_overall_eval_multi_process, datas, chunksize=1, max_workers=num_workers)
        scores = {}
        for chunk in scores_chunks:
            for k, v in chunk.items():
                scores[k] = scores.get(k, []) + v
    else:
        scores = _overall_eval(candidates, targets, metrics)
    return scores

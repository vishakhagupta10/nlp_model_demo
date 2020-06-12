"""  evaluation script for v1.1 of the SQuAD dataset. """
from __future__ import print_function
from collections import Counter
import string

import re
import argparse
import json
import sys

#    return re.sub(r'\b(a|an|the)\b', ' ', text)


def f1_score(prediction, ground_truth, tokenizer):
    #prediction_tokens = tokenizer.tokenize(normalize_answer(prediction))
    #ground_truth_tokens = tokenizer.tokenize(normalize_answer(ground_truth))
    prediction_tokens = tokenizer.tokenize(prediction)
    ground_truth_tokens = tokenizer.tokenize(ground_truth)
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth, tokenizer):
    return (' '.join(tokenizer.tokenize(prediction)) ==
            ' '.join(tokenizer.tokenize(ground_truth)))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths, tokenizer):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth, tokenizer)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate(dataset, predictions, tokenizer):
    acc = f1 = exact_match = total = 0
    for article in dataset:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                total += 1
                if str(qa['id']) not in predictions and int(qa['id']) not in predictions:
                    message = 'Unanswered question ' + str(qa['id']) + \
                              ' will receive score 0.'
                    print(message, file=sys.stderr)
                    continue
                ground_truths = list(map(lambda x: x['text'], qa['answers']))
                try:
                    prediction = predictions[str(qa['id'])]
                except KeyError:
                    prediction = predictions[int(qa['id'])]
                if ground_truths[0].lower() in prediction:
                    acc += 1
                exact_match += metric_max_over_ground_truths(
                    exact_match_score, prediction, ground_truths, tokenizer)
                f1 += metric_max_over_ground_truths(
                    f1_score, prediction, ground_truths, tokenizer)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    acc = 100.0 * acc / total
    return {'exact_match': exact_match, 'f1': f1, 'acc': acc}


if __name__ == '__main__':
    import tokenization
    tokenizer = tokenization.FullTokenizer(vocab_file='model_repo/uncased_L-12_H-768_A-12/vocab.txt ')

    parser = argparse.ArgumentParser(
        description='Evaluation for ASQ-fac dataset ')
    parser.add_argument('dataset_file', help='Dataset file')
    parser.add_argument('prediction_file', help='Prediction File')
    args = parser.parse_args()
    with open(args.dataset_file) as dataset_file:
        dataset_json = json.load(dataset_file)
        dataset = dataset_json['data']
    with open(args.prediction_file) as prediction_file:
        predictions = json.load(prediction_file)
    print(json.dumps(evaluate(dataset, predictions, tokenizer)))

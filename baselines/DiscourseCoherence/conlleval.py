"""
参考:
https://www.clips.uantwerpen.be/conll2000/chunking/output.html
https://github.com/sighsmile/conlleval

This script applies to IOB2 or IOBES tagging scheme.
If you are using a different scheme, please convert to IOB2 or IOBES.
IOB2:
- B = begin, 
- I = inside but not the first, 
- O = outside
e.g. 
John   lives in New   York  City  .
B-PER  O     O  B-LOC I-LOC I-LOC O
IOBES:
- B = begin, 
- E = end, 
- S = singleton, 
- I = inside but not the first or the last, 
- O = outside
e.g.
John   lives in New   York  City  .
S-PER  O     O  B-LOC I-LOC E-LOC O
prefix: IOBES
chunk_type: PER, LOC, etc.
"""
from __future__ import division, print_function, unicode_literals

import sys
from collections import defaultdict

def split_tag(chunk_tag):
    """
    split chunk tag into IOBES prefix and chunk_type
    e.g. 
    B-PER -> (B, PER)
    O -> (O, None)
    """
    # print(chunk_tag,type(chunk_tag),chunk_tag.split('-', maxsplit=1))
    if chunk_tag == 'O':
        return ('O', None)
    # return chunk_tag.split('-', maxsplit=1)
    return (None,chunk_tag)

def is_chunk_end(prev_tag, tag):
    """
    check if the previous chunk ended between the previous and current word
    e.g. 
    (B-PER, I-PER) -> False
    (B-LOC, O)  -> True
    Note: in case of contradicting tags, e.g. (B-PER, I-LOC)
    this is considered as (B-PER, B-LOC)
    """
    prefix1, chunk_type1 = split_tag(prev_tag)
    prefix2, chunk_type2 = split_tag(tag)

    if prefix1 == 'O':
        return False
    if prefix2 == 'O':
        return prefix1 != 'O'

    if chunk_type1 != chunk_type2:
        return True

    return prefix2 in ['B', 'S'] or prefix1 in ['E', 'S']

def is_chunk_start(prev_tag, tag):
    """
    check if a new chunk started between the previous and current word
    """
    prefix1, chunk_type1 = split_tag(prev_tag)
    prefix2, chunk_type2 = split_tag(tag)

    if prefix2 == 'O':
        return False
    if prefix1 == 'O':
        return prefix2 != 'O'

    if chunk_type1 != chunk_type2:
        return True

    return prefix2 in ['B', 'S'] or prefix1 in ['E', 'S']


def calc_metrics(tp, p, t, percent=True):
    """
    compute overall precision, recall and FB1 (default values are 0.0)
    if percent is True, return 100 * original decimal value
    """
    precision = tp / p if p else 0
    recall = tp / t if t else 0
    fb1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    if percent:
        return 100 * precision, 100 * recall, 100 * fb1
    else:
        return precision, recall, fb1


def count_chunks(true_seqs, pred_seqs):
    """
    true_seqs: a list of true tags
    pred_seqs: a list of predicted tags
    return: 
    correct_chunks: a dict (counter), 
                    key = chunk types, 
                    value = number of correctly identified chunks per type
    true_chunks:    a dict, number of true chunks per type
    pred_chunks:    a dict, number of identified chunks per type
    correct_counts, true_counts, pred_counts: similar to above, but for tags
    """
    correct_chunks = defaultdict(int)
    true_chunks = defaultdict(int)
    pred_chunks = defaultdict(int)

    correct_counts = defaultdict(int)
    true_counts = defaultdict(int)
    pred_counts = defaultdict(int)

    prev_true_tag, prev_pred_tag = 'O', 'O'
    correct_chunk = None

    for true_tag, pred_tag in zip(true_seqs, pred_seqs):
        if true_tag == pred_tag:
            correct_counts[true_tag] += 1
        true_counts[true_tag] += 1
        pred_counts[pred_tag] += 1

        _, true_type = split_tag(true_tag)
        _, pred_type = split_tag(pred_tag)

        if correct_chunk is not None:
            true_end = is_chunk_end(prev_true_tag, true_tag)
            pred_end = is_chunk_end(prev_pred_tag, pred_tag)

            if pred_end and true_end:
                correct_chunks[correct_chunk] += 1
                correct_chunk = None
            elif pred_end != true_end or true_type != pred_type:
                correct_chunk = None

        true_start = is_chunk_start(prev_true_tag, true_tag)
        pred_start = is_chunk_start(prev_pred_tag, pred_tag)

        if true_start and pred_start and true_type == pred_type:
            correct_chunk = true_type
        if true_start:
            true_chunks[true_type] += 1
        if pred_start:
            pred_chunks[pred_type] += 1

        prev_true_tag, prev_pred_tag = true_tag, pred_tag
    if correct_chunk is not None:
        correct_chunks[correct_chunk] += 1

    # print("true_seqs",true_seqs)
    # print("pred_seqs",pred_seqs)
    # print("")

    return (correct_chunks, true_chunks, pred_chunks, 
        correct_counts, true_counts, pred_counts)

def get_result(correct_chunks, true_chunks, pred_chunks,
    correct_counts, true_counts, pred_counts, verbose=True):
    """
    if verbose, print overall performance, as well as preformance per chunk type;
    otherwise, simply return overall prec, rec, f1 scores
    """
    # sum counts
    sum_correct_chunks = sum(correct_chunks.values())
    sum_true_chunks = sum(true_chunks.values())
    sum_pred_chunks = sum(pred_chunks.values())

    sum_correct_counts = sum(correct_counts.values())
    sum_true_counts = sum(true_counts.values())

    nonO_correct_counts = sum(v for k, v in correct_counts.items() if k != 'O')
    nonO_true_counts = sum(v for k, v in true_counts.items() if k != 'O')

    chunk_types = sorted(list(set(list(true_chunks) + list(pred_chunks))))

    # compute overall precision, recall and FB1 (default values are 0.0)
    prec, rec, f1 = calc_metrics(sum_correct_chunks, sum_pred_chunks, sum_true_chunks)
    res = (prec, rec, f1)
    if not verbose:
        return res

    # print overall performance, and performance per chunk type
    
    print("processed %i tokens with %i phrases; " % (sum_true_counts, sum_true_chunks), end='')
    print("found: %i phrases; correct: %i.\n" % (sum_pred_chunks, sum_correct_chunks), end='')
        
    print("accuracy: %6.2f%%; (non-O)" % (100*nonO_correct_counts/nonO_true_counts))
    print("accuracy: %6.2f%%; " % (100*sum_correct_counts/sum_true_counts), end='')
    print("precision: %6.2f%%; recall: %6.2f%%; FB1: %6.2f" % (prec, rec, f1),end="")
    print("  (%d & %d) = %d" % (sum_pred_chunks,sum_true_chunks,sum_correct_chunks))


    # for each chunk type, compute precision, recall and FB1 (default values are 0.0)
    for t in chunk_types:
        prec, rec, f1 = calc_metrics(correct_chunks[t], pred_chunks[t], true_chunks[t])
        print("%17s: " %t , end='')
        print("precision: %6.2f%%; recall: %6.2f%%; FB1: %6.2f" %
                    (prec, rec, f1), end='')
        print("  (%d & %d) = %d" % (pred_chunks[t],true_chunks[t],correct_chunks[t]))

    return res
    # you can generate LaTeX output for tables like in
    # http://cnts.uia.ac.be/conll2003/ner/example.tex
    # but I'm not implementing this

def evaluate(true_seqs, pred_seqs, verbose=True):
    (correct_chunks, true_chunks, pred_chunks,
        correct_counts, true_counts, pred_counts) = count_chunks(true_seqs, pred_seqs)
    result = get_result(correct_chunks, true_chunks, pred_chunks,
        correct_counts, true_counts, pred_counts, verbose=verbose)
    return result

def RE_get_result(correct_counts, true_counts, pred_counts, verbose=True):
    """
    if verbose, print overall performance, as well as preformance per chunk type;
    otherwise, simply return overall prec, rec, f1 scores
    """
    # sum counts
    sum_correct_counts = sum(correct_counts.values())
    sum_true_counts = sum(true_counts.values())
    sum_pred_counts = sum(pred_counts.values())

    nonO_correct_counts = sum(v for k, v in correct_counts.items() if k != 'O')
    nonO_true_counts = sum(v for k, v in true_counts.items() if k != 'O')

    chunk_types = sorted(list(set(list(true_counts) + list(pred_counts))))

    # compute overall precision, recall and FB1 (default values are 0.0)
    prec, rec, f1 = calc_metrics(sum_correct_counts, sum_pred_counts, sum_true_counts)
    res = (prec, rec, f1)
    if not verbose:
        return res

    # print overall performance, and performance per chunk type

    # print("processed %i tokens with %i phrases; " % (sum_true_counts, sum_true_chunks), end='')
    print("processed %i relations; " % (sum_true_counts), end='')
    print("found: %i relations; correct: %i.\n" % (sum_pred_counts, sum_correct_counts), end='')
        
    print("accuracy: %6.2f%%; (non-O)" % (100*nonO_correct_counts/nonO_true_counts))
    print("accuracy: %6.2f%%; " % (100*sum_correct_counts/sum_true_counts), end='')
    print("precision: %6.2f%%; recall: %6.2f%%; FB1: %6.2f" % (prec, rec, f1),end="")
    print("  (%d & %d) = %d" % (sum_pred_counts,sum_true_counts,sum_correct_counts))


    # for each chunk type, compute precision, recall and FB1 (default values are 0.0)
    for t in chunk_types:
        prec, rec, f1 = calc_metrics(correct_counts[t], pred_counts[t], true_counts[t])
        print("%17s: " %t , end='')
        print("precision: %6.2f%%; recall: %6.2f%%; FB1: %6.2f" %
                    (prec, rec, f1), end='')
        print("  (%d & %d) = %d" % (pred_counts[t],true_counts[t],correct_counts[t]))

    return res

def RE_count_chunks(true_seqs, pred_seqs):
    """
    true_seqs: a list of true tags
    pred_seqs: a list of predicted tags
    return: 
    correct_chunks: a dict (counter), 
                    key = chunk types, 
                    value = number of correctly identified chunks per type 
    true_chunks:    a dict, number of true chunks per type / 每种类型的真实的chunk的数量
    pred_chunks:    a dict, number of identified chunks per type / 每种类型识别到的（预测的）chunk的数量
    correct_counts, true_counts, pred_counts: similar to above, but for tags
    """
    correct_counts = defaultdict(int)
    true_counts = defaultdict(int)
    pred_counts = defaultdict(int)

    for true_tag, pred_tag in zip(true_seqs, pred_seqs):
        if true_tag == pred_tag and pred_tag!="0" and pred_tag!="ROOT":
            correct_counts[true_tag] += 1
        if true_tag!='0' and true_tag!="ROOT":
            true_counts[true_tag] += 1
        if pred_tag!='0' and pred_tag!="ROOT":
            pred_counts[pred_tag] += 1

    return (correct_counts, true_counts, pred_counts)

def RE_evaluate(true_seqs, pred_seqs, verbose=True):
    (correct_counts, true_counts, pred_counts) = RE_count_chunks(true_seqs, pred_seqs)
    result = RE_get_result(correct_counts, true_counts, pred_counts, verbose=verbose)
    return result

def evaluate_conll_file(fileIterator):
    """
    NER评估函数
    :param fileIterator: NER得到的txt文件
    :return: 如下评估指标信息，着重关注最后一行的准确率
    eg:
        processed 4503502 tokens with 93009 phrases; found: 92829 phrases; correct: 89427.
        accuracy:  97.43%; (non-O)
        accuracy:  99.58%; precision:  96.34%; recall:  96.15%; FB1:  96.24
                    COM: precision:  96.34%; recall:  96.15%; FB1:  96.24  92829

        分别表示：
        txt文件一共包含 4503502 个字符， 其中共 93009 个实体（gold）， 模型预测实体共有 92829 个， 其中正确的有 89427.
        只看实体名（non-O）的字符级准确率 97.43%（字符级）
        所有的字符级准确率 99.58%（字符级）     后面三个 p/r/f 和下一行相同。
                    实体为COM的短语级别 precision/recall/FB1 分别为96.34%; 96.15%; 96.24 (这三个都是短语级，当整个实体的BI全部预测正确才算正确)
    """
    true_seqs, pred_seqs = [], []
    for line in fileIterator:
        cols = line.strip().split()
        # each non-empty line must contain >= 3 columns
        if not cols:
            true_seqs.append('O')
            pred_seqs.append('O')
        elif len(cols) < 3:
            raise IOError("conlleval: too few columns in line %s\n" % line)
        else:
            # extract tags from last 2 columns
            true_seqs.append(cols[-2])
            pred_seqs.append(cols[-1])
    return RE_evaluate(true_seqs, pred_seqs)

if __name__ == '__main__':
    """
    usage:     conlleval < file
    """
    evaluate_conll_file(sys.stdin)
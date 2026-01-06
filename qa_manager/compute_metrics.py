from qa_manager.normalize_answers import *
from tqdm import tqdm
from collections import Counter

import os


# def normalize_answer_final(answer):
#     pre_answer = answer.split('\n\n')[-1].split('Answer: ')[-1].split('The answer is: ')[-1]
#     final_answer = normalize_answer(pre_answer)
#     return final_answer

def normalize_answer_final(answer):
    # 如果 answer 是 list，把它合并成一个字符串
    if isinstance(answer, list):
        answer = ' '.join(answer)
    
    pre_answer = (
        answer
        .split('\n\n')[-1]
        .split('Answer: ')[-1]
        .split('The answer is: ')[-1]
    )
    final_answer = normalize_answer(pre_answer)
    return final_answer

def answer_post_refine(answer):
    return answer.split("Answer: ")[-1]

# def compute_scores(predict_answers, golden_answers):
#     assert len(predict_answers) == len(golden_answers), "预测答案和标准答案的长度不相等"
#     final_metric = {"acc": 0, "em": 0, "f1": 0, "precision": 0, "recall": 0}
#     total = len(predict_answers)

#     for prediction, ground_truth in zip(predict_answers, golden_answers):
#         normalized_prediction = normalize_answer_final(prediction)
#         normalized_ground_truth = normalize_answer_final(ground_truth)

#         if normalized_prediction in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
#             continue
#         if normalized_ground_truth in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
#             continue
        
#         if normalized_ground_truth in normalized_prediction:# or normalized_prediction in normalized_ground_truth:
#             final_metric["acc"] += 1.0

#         if normalized_prediction == normalized_ground_truth:
#             final_metric["em"] += 1.0

#         prediction_tokens = normalized_prediction.split()
#         ground_truth_tokens = normalized_ground_truth.split()
#         common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
#         num_same = sum(common.values())
#         if num_same == 0:
#             continue

#         precision = 1.0 * num_same / len(prediction_tokens)
#         recall = 1.0 * num_same / len(ground_truth_tokens)
#         f1 = (2 * precision * recall) / (precision + recall)

#         final_metric["f1"] += f1
#         final_metric["precision"] += precision
#         final_metric["recall"] += recall

#     for k in ['acc', 'em', 'f1', 'precision', 'recall']:
#         final_metric[k] /= total

#     return final_metric


# def compute_scores(predict_answers, golden_answers):
#     assert len(predict_answers) == len(golden_answers), "预测答案和标准答案的长度不相等"
#     final_metric = {"acc": 0, "em": 0, "f1": 0, "precision": 0, "recall": 0}
#     total = len(predict_answers)

#     for prediction, ground_truths in zip(predict_answers, golden_answers):
#         if isinstance(ground_truths, str):
#             ground_truths = [ground_truths]
#         elif isinstance(ground_truths, list):
#             pass
#         else:
#             raise ValueError("The answer must be str or list format.")

#         temp_metric_dict = {"acc": 0, "em": 0, "f1": 0, "precision": 0, "recall": 0}
#         normalized_prediction = normalize_answer(prediction)
#         for ground_truth in ground_truths:
#             normalized_ground_truth = normalize_answer(ground_truth)

#             if normalized_prediction in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
#                 continue
#             if normalized_ground_truth in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
#                 continue
            
#             if normalized_ground_truth in normalized_prediction:# or normalized_prediction in normalized_ground_truth:
#                 temp_metric_dict["acc"] = 1.0

#             if normalized_prediction == normalized_ground_truth:
#                 temp_metric_dict["em"] = 1.0

#             prediction_tokens = normalized_prediction.split()
#             ground_truth_tokens = normalized_ground_truth.split()
#             common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
#             num_same = sum(common.values())
#             if num_same == 0:
#                 continue

#             precision = 1.0 * num_same / len(prediction_tokens)
#             recall = 1.0 * num_same / len(ground_truth_tokens)
#             f1 = (2 * precision * recall) / (precision + recall)

#             temp_metric_dict["precision"] = max(precision, temp_metric_dict["precision"])
#             temp_metric_dict["recall"] =  max(recall, temp_metric_dict["recall"])
#             temp_metric_dict["f1"] = max(f1, temp_metric_dict["f1"])
        
#         for k in ['acc', 'em', 'f1', 'precision', 'recall']:
#             final_metric[k] += temp_metric_dict[k]

#     for k in ['acc', 'em', 'f1', 'precision', 'recall']:
#         final_metric[k] /= total

#     return final_metric

def compute_scores(predict_answers, golden_answers):
    assert len(predict_answers) == len(golden_answers), "预测答案和标准答案的长度不相等"
    
    # 结果累积
    final_metric = {"acc": 0, "em": 0, "f1": 0, "precision": 0, "recall": 0}

    # 分母统计
    acc_em_count = 0       # acc + em 的分母（所有样本）
    fpr_count = 0          # f1 + precision + recall 的分母（普通样本）

    for prediction, ground_truths in zip(predict_answers, golden_answers):
        if isinstance(ground_truths, str):
            ground_truths = [ground_truths]
        elif not isinstance(ground_truths, list):
            raise ValueError("The answer must be str or list format.")

        temp_metric_dict = {"acc": 0, "em": 0, "f1": 0, "precision": 0, "recall": 0}
        normalized_prediction = normalize_answer(prediction)

        acc_em_count += 1  # 所有样本都统计

        # 检查是否是特殊标签
        if normalized_prediction in ["yes", "no", "noanswer"] and all(
            normalize_answer(gt) in ["yes", "no", "noanswer"] for gt in ground_truths
        ):
            # 特殊标签只计算 acc 和 em
            for ground_truth in ground_truths:
                if normalized_prediction == normalize_answer(ground_truth):
                    temp_metric_dict["acc"] = 1.0
                    temp_metric_dict["em"] = 1.0
            # 累加到结果
            final_metric["acc"] += temp_metric_dict["acc"]
            final_metric["em"] += temp_metric_dict["em"]
            continue

        # 普通答案参与所有指标
        fpr_count += 1

        for ground_truth in ground_truths:
            normalized_ground_truth = normalize_answer(ground_truth)

            # 判断准确率
            if normalized_ground_truth in normalized_prediction:
                temp_metric_dict["acc"] = 1.0

            # 完全匹配
            if normalized_prediction == normalized_ground_truth:
                temp_metric_dict["em"] = 1.0

            # token-level 计算
            prediction_tokens = normalized_prediction.split()
            ground_truth_tokens = normalized_ground_truth.split()
            common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
            num_same = sum(common.values())
            if num_same == 0:
                continue

            precision = 1.0 * num_same / len(prediction_tokens)
            recall = 1.0 * num_same / len(ground_truth_tokens)
            f1 = (2 * precision * recall) / (precision + recall)

            temp_metric_dict["precision"] = max(precision, temp_metric_dict["precision"])
            temp_metric_dict["recall"] =  max(recall, temp_metric_dict["recall"])
            temp_metric_dict["f1"] = max(f1, temp_metric_dict["f1"])

        # 累加指标
        for k in ['acc', 'em', 'f1', 'precision', 'recall']:
            final_metric[k] += temp_metric_dict[k]

    # 分母区分计算
    final_metric["acc"] /= acc_em_count
    final_metric["em"] /= acc_em_count
    if fpr_count > 0:
        final_metric["f1"] /= fpr_count
        final_metric["precision"] /= fpr_count
        final_metric["recall"] /= fpr_count
    else:
        final_metric["f1"] = final_metric["precision"] = final_metric["recall"] = 0.0

    return final_metric
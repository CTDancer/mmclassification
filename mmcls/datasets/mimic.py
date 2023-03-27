import os.path as osp
import numpy as np
import json
import copy
import random
from sklearn.metrics import roc_auc_score
from .multi_label import MultiLabelDataset
from .builder import DATASETS
import wandb


@DATASETS.register_module()
class MIMIC(MultiLabelDataset):
    """CLASSES = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
               'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
               'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other',
               'Pneumonia', 'Pneumothorax', 'Support Devices']"""
    CLASSES = [str(i) for i in range(14)]

    def __init__(self,
                 text_file=None,
                 class_balance=False,
                 **kwargs):
        self.text_file = text_file
        self.class_balance = class_balance
        self.subject_infos = {}
        super().__init__(**kwargs)

    def load_annotations(self):
        assert isinstance(self.ann_file, str)

        if self.text_file is not None:
            with open(self.text_file, 'r') as tf:
                bag_text = json.load(tf)

        data_infos = []
        with open(self.ann_file, 'r') as f:
            for i, line in enumerate(f):
                filename, class_name = line.strip().split(' ')
                subject_id, study_id, filename = filename.split('_')
                gt_label = [int(label) for label in class_name]  # class_to_idx
                bag = f'{subject_id}_{study_id}'
                info = {
                    'img_prefix': None,
                    'img_info': {
                        'filename': osp.join(self.data_prefix, filename)
                    },
                    'gt_label': np.array(gt_label, dtype=np.int64),
                    'bag': bag,
                    'subject': subject_id,
                    'study': study_id,
                    'img_text': ''
                }
                if self.text_file is not None:
                    info['img_text'] = bag_text[bag]
                data_infos.append(info)

        for i, info in enumerate(data_infos):
            self.subject_infos[info['bag']] = self.subject_infos.get(info['bag'], []) + [i]

        if self.class_balance:
            self.class_files = [[] for c in range(len(self.CLASSES) * 2)]
            for i, info in enumerate(data_infos):
                for c in range(len(self.CLASSES)):
                    self.class_files[c + info['gt_label'][c] * 14].append(i)
                    # self.class_files: [neg_0, neg_1, ..., neg_13, pos_0, ..., pos_13]
            for c in range(len(self.CLASSES) * 2):
                random.shuffle(self.class_files[c])
            self.class_counter = 0
            self.img_counter = [0 for c in range(28)]

        return data_infos

    def __getitem__(self, idx):
        if not self.class_balance:
            return self.prepare_data(idx)
        c = self.class_counter
        i = self.img_counter[c]
        idx = self.class_files[c][i]
        self.img_counter[c] = (i + 1) % len(self.class_files[c])
        if i == 0:
            random.shuffle(self.class_files[c])
        self.class_counter = (self.class_counter + 1) % 28
        results = copy.deepcopy(self.data_infos[idx])
        results.update({'gt_balance': c % 14})  # used to update loss only for current class
        return self.pipeline(results)

    def evaluate(self,
                 results,
                 metric='accuracy',
                 metric_options=None,
                 indices=None,
                 logger=None):
        if metric_options is None:
            metric_options = dict(thr=0.5)
        if isinstance(metric, str):
            metrics = [metric]
        else:
            metrics = metric
        allowed_metrics = [
            'accuracy', 'class_accuracy', 'mean_accuracy', 'auc', 'class_auc',
            'bag_accuracy', 'bag_class_accuracy', 'mean_bag_accuracy', 'bag_auc',
            'bag_class_auc'
        ]
        eval_results = {}
        results = np.vstack(results)
        gt_labels = self.get_gt_labels()
        if indices is not None:
            gt_labels = gt_labels[indices]
        num_imgs = len(results)
        assert len(gt_labels) == num_imgs, 'dataset testing results should '\
            'be of the same length as gt_labels.'

        invalid_metrics = set(metrics) - set(allowed_metrics)
        if len(invalid_metrics) != 0:
            raise ValueError(f'metric {invalid_metrics} is not supported.')

        thr = metric_options.get('thr', 0.5)
        eps = np.finfo(np.float32).eps

        agg_hard = metric_options.get('agg_hard', False)
        vote_hard = metric_options.get('vote_hard', False)
        if any(metric.startswith('bag_') for metric in metrics):
            bags = list(self.subject_infos.keys())
            bag_gt_labels = np.array([gt_labels[self.subject_infos[b][0]] for b in bags])
            if agg_hard:
                raise NotImplementedError
            else:
                bag_results = np.array([np.mean(results[self.subject_infos[b]], axis=0) for b in bags])
            pos_inds = bag_results >= thr  # only use threshold now
            true = pos_inds == bag_gt_labels
            false = pos_inds != bag_gt_labels
            if 'bag_accuracy' in metrics:
                acc = true.sum() / np.maximum(true.sum() + false.sum(), eps)
                eval_results.update({'bag_accuracy': acc * 100.0})
            if 'mean_bag_accuracy' in metrics or 'bag_class_accuracy' in metrics:
                acc = true.sum(0) / np.maximum(true.sum(0) + false.sum(0), eps)
                for i in range(len(self.CLASSES)):
                    eval_results.update({f'bag_accuracy_{i}': acc[i] * 100.0})
                eval_results.update({'mean_bag_accuracy': acc.mean() * 100.0})
            if 'bag_auc' in metrics:
                auc = roc_auc_score(bag_gt_labels, bag_results)
                eval_results.update({'bag_auc': auc})
            if 'bag_class_auc' in metrics:
                for i in range(len(self.CLASSES)):
                    auc = roc_auc_score(bag_gt_labels[:, i], bag_results[:, i])
                    eval_results.update({f'bag_auc_{i}': auc})

        pos_inds = results >= thr  # only use threshold now
        true = pos_inds == gt_labels
        false = pos_inds != gt_labels

        if 'accuracy' in metrics:
            acc = true.sum() / np.maximum(true.sum() + false.sum(), eps)
            eval_results.update({'accuracy': acc * 100.0})

        if 'mean_accuracy' in metrics or 'class_accuracy' in metrics:
            acc = true.sum(0) / np.maximum(true.sum(0) + false.sum(0), eps)
            for i in range(len(self.CLASSES)):
                eval_results.update({f'accuracy_{i}': acc[i] * 100.0})
            eval_results.update({'mean_accuracy': acc.mean() * 100.0})

        if 'auc' in metrics:
            auc = roc_auc_score(gt_labels, results)
            eval_results.update({'auc': auc})

        if 'class_auc' in metrics:
            for i in range(len(self.CLASSES)):
                auc = roc_auc_score(gt_labels[:, i], results[:, i])
                eval_results.update({f'auc_{i}': auc})

        print(eval_results)
        wandb.log("{}: {}".format(eval_results.key, eval_results.val))

        return eval_results

from math import log
import numpy as np

def calc_entropy(dataset):
    num_samples = len(dataset)
    cnt_label = {}
    for feat_vec in dataset:
        curr_label = feat_vec[-1]
        if curr_label not in cnt_label.keys():
            cnt_label[curr_label] = 0
        cnt_label[curr_label] += 1
    shannon_ent = 0.
    for key in cnt_label:
        prob = float(cnt_label[key]) / num_samples
        shannon_ent -= prob * log(prob, 2)
    return shannon_ent

def create_dataset():
    dataSet = [[1,1,'yes'], 
               [1,1,'yes'],
               [1,0,'no'],
               [0,1,'no'],
               [0,1,'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


def split_dataset(dataset, axis, value):
    ret_dataset = []
    for feat_vec in dataset:
        # print("feat_vec: ", feat_vec)
        if feat_vec[axis] == value:
            reduce_feat_vec = feat_vec[:axis]
            # print("reduce_feat_vec: ", reduce_feat_vec)
            reduce_feat_vec.extend(feat_vec[axis+1:])
            # print("reduce_feat_vec: ", reduce_feat_vec)
            ret_dataset.append(reduce_feat_vec)
            # input('s')
    return ret_dataset

def choose_best_feature(dataset):
    num_features = len(dataset[0])
    base_entropy = calc_entropy(dataset)
    best_infogain = 0.
    best_feature_ind = -1
    for i in range(num_features):
        feat_list = [example[i] for example in dataset]
        unique_val = set(feat_list)
        new_entropy = 0.
        for value in unique_val:
            sub_dataset = split_dataset(dataset, i, value)
            prob = len(sub_dataset)/float(len(dataset))
            new_entropy += prob * calc_entropy(sub_dataset)
        infogain = base_entropy - new_entropy
        if (infogain > best_infogain):
            best_infogain = infogain
            best_feature_ind = i
    return best_feature_ind

def majority_cnt(class_list):
    class_cnt = {}
    for vote in class_list:
        if vote not in class_list.keys():
            class_cnt[vote] = 0
        class_cnt[vote] += 1
    sorted_class_cnt = sorted(class_cnt.items(),
        key = operator.itemgetter(1), reverse=True)
    return sorted_class_cnt[0][0]

def create_tree(dataset, labels):
    class_list = [example[-1] for example in dataset]
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    if len(dataset[0]) == 1:
        return majority_cnt(class_list)
    best_feature = choose_best_feature(dataset)
    best_feature_label = labels[best_feature]
    print(best_feature, best_feature_label)
    print("#######################")
    m_tree = {best_feature_label:{}}
    del(labels[best_feature])
    feature_val = [example[best_feature] for example in dataset]
    unique_val = set(feature_val)
    for value in unique_val:
        sub_labels = labels[:]
        m_tree[best_feature_label][value] = create_tree(split_dataset(dataset, best_feature, value),
                                                        sub_labels)
    return m_tree

    
my_data, labels = create_dataset()
m_tree = create_tree(my_data, labels)
print(m_tree)
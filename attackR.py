#决策树模型，规则攻击
from art.attacks.inference.membership_inference import MembershipInferenceBlackBoxRuleBased
import numpy as np
from data import load_nursery_data, train_decision_tree

# 加载数据
x_train, y_train, x_test, y_test = load_nursery_data()


# 训练模型
model, art_classifier = train_decision_tree(x_train, y_train)
#这里创建了一个基于规则的黑盒成员资格推断攻击对象。art_classifier 是一个训练好的模型，它被用来对数据点进行预测，从而让攻击者可以推断数据点是否属于训练集。
attack = MembershipInferenceBlackBoxRuleBased(art_classifier)

# infer attacked feature
inferred_train = attack.infer(x_train, y_train)
inferred_test = attack.infer(x_test, y_test)

# check accuracy
#inferred_train 是布尔数组，True表示攻击者正确预测该数据点是训练集的成员
train_acc = np.sum(inferred_train) / len(inferred_train)
#测试集的准确度，非成员（训练时未使用的数据点）的准确度
test_acc = 1 - (np.sum(inferred_test) / len(inferred_test))
#加权平均
acc = (train_acc * len(inferred_train) + test_acc * len(inferred_test)) / (len(inferred_train) + len(inferred_test))
print(f"Members Accuracy: {train_acc:.4f}")
print(f"Non Members Accuracy {test_acc:.4f}")
print(f"Attack Accuracy {acc:.4f}")

#predicted: 一个数组或列表，包含模型对每个数据点的预测结果，通常是一个二进制值（0或1）。actual: 一个数组或列表，包含每个数据点的实际标签，与 predicted 长度相同。positive_value: 一个整数，表示正类别的值，默认为1。
#score: 用于记录预测为正类别且实际也为正类别的样本数（真正例，TruePositives）。num_positive_predicted: 用于记录预测为正类别的样本数。num_positive_actual: 用于记录实际为正类别的样本数。

def calc_precision_recall(predicted, actual, positive_value=1):
    score = 0  # both predicted and actual are positive
    num_positive_predicted = 0  # predicted positive
    num_positive_actual = 0  # actual positive
    for i in range(len(predicted)):
        if predicted[i] == positive_value:
            num_positive_predicted += 1
        if actual[i] == positive_value:
            num_positive_actual += 1
        if predicted[i] == actual[i]:
            if predicted[i] == positive_value:
                score += 1

    if num_positive_predicted == 0:
        precision = 1
    else:
        precision = score / num_positive_predicted  # the fraction of predicted “Yes” responses that are correct
    if num_positive_actual == 0:
        recall = 1
    else:
        recall = score / num_positive_actual  # the fraction of “Yes” responses that are predicted correctly

    return precision, recall


# rule-based训练数据
print(f"A{np.column_stack((x_train, y_train)).astype(str)}")
print("aa")

#精确度、召回率
print(calc_precision_recall(np.concatenate((inferred_train, inferred_test)),
                            np.concatenate((np.ones(len(inferred_train)), np.zeros(len(inferred_test))))))
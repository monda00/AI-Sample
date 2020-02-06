from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()

X = iris['data']
y = iris['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

def predict_iris(feature):
    if feature[2] < 2 and feature[3] < 0.6:
        return 0
    elif 2 <= feature[2] < 5 and 0.6 <= feature[3] < 1.7:
        return 1
    else:
        return 2

def compute_score(pred, ans):
    correct_answer_num = 0
    for p, a in zip(pred, ans):
        if p == a:
            correct_answer_num += 1
    return correct_answer_num / len(pred)

pred = []
for feature in X_test:
    pred.append(predict_iris(feature))

score = compute_score(pred, y_test)

print('Score is', score)

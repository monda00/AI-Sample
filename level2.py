from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()

X = iris['data']
y = np_utils.to_categorical(iris['target'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# if文を大量に入れて知識を表現する
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

iris = load_iris()

X = iris['data']
y = iris['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

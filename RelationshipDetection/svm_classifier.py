from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def extract_features():
    texts = []
    labels = ['mother-of', 'father-of', 'brother-of', 'sister-of', 'son-of', 'daughter-of', 'grandfather-of',
              'grandmother-of', 'husband-of', 'wife-of', 'uncle-of', 'aunt-of', 'friend-of']

    # TODO

    return texts, labels


texts, labels = extract_features()

# test-train split - holdout method
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=1)

clf = SVC(X_train, y_train)
score = clf.score(X_test, y_test)


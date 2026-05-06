from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.metrics import accuracy_score
from diversity_curves.utils import make_splits
import numpy as np
from sklearn.metrics import accuracy_score
from diversity_curves.utils import make_splits

def knn_classifier(X, y, precomputed=True, seed=42, outer_k=10, holdout_test_size=0.1, k=5, groups=None):
    y = np.asarray(y)
    test_accs = []

    indices = make_splits(np.array([i for i in range(len(y))]), y, outer_k=outer_k, inner_k=None, holdout_test_size=holdout_test_size, seed=seed, stratify=True, groups=groups)

    for repeat in range(outer_k):
        idx_tr, idx_va, idx_te = indices[repeat]
        if precomputed:
            X_train, X_test = X[idx_tr,:], X[idx_te,:]
            X_train, X_test = X_train[:,idx_tr], X_test[:,idx_tr]
        else:
            X_train, X_test = X[idx_tr,:], X[idx_te,:]
        y_train, y_test = y[idx_tr], y[idx_te]

        k = min(k, len(y_train))
        if precomputed:
            clf = KNeighborsClassifier(n_neighbors=k, metric="precomputed")
        else:
            clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        test_accs.append(acc)
        print(f"Fold {repeat}: test accuracy = {acc:.4f} (k={k})")

    print()
    print("Mean test accuracy: {:.4f}".format(np.mean(test_accs)))
    print("Std test accuracy:  {:.4f}".format(np.std(test_accs)))
    return test_accs

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
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


def knn_classifier_cv(X, y, precomputed=True, seed=42, outer_k=10, holdout_test_size=0.1, stratify=True, tune=False, scale=False):
    y = np.asarray(y)
    test_accs = []
    test_aurocs = []
    test_accs_balanced =[]
    conf_matrices = []
    n = X.shape[0]

    indices = make_splits(np.array([i for i in range(len(y))]), y, outer_k=outer_k, inner_k=None, holdout_test_size=holdout_test_size, seed=seed, stratify=stratify)

    for repeat in range(outer_k):
        idx_tr, idx_va, idx_te = indices[repeat]
        if precomputed:
            X_train, X_test, X_val = X[idx_tr,:], X[idx_te,:], X[idx_va,:]
            X_train, X_test, X_val = X_train[:,idx_tr], X_test[:,idx_tr], X_val[:,idx_tr]
        else:
            X_train, X_test = X[idx_tr,:], X[idx_te,:]
            X_val = X[idx_va,:]
        y_train, y_test = y[idx_tr], y[idx_te]
        y_val = y[idx_va]

        if scale and not precomputed:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            X_val = scaler.transform(X_val)

        k = min(5, len(y_train))
        if precomputed:
            clf = KNeighborsClassifier(n_neighbors=k, metric="precomputed")
        else:
            clf = KNeighborsClassifier(n_neighbors=k)

        if tune:
            param_dist = {
                'n_neighbors': [1, 3, 5, 7,9]
            }

            grid = GridSearchCV(KNeighborsClassifier(), param_dist , refit=True,  cv=3, n_jobs=-1) 
            grid.fit(X_val,y_val)
            clf = grid.best_estimator_
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        acc_b = balanced_accuracy_score(y_test, preds)
        conf_matrix = confusion_matrix(y_test, preds)
        try:
            auc_score = roc_auc_score(y_test, clf.predict_proba(X_test), multi_class='ovr')
        except:
            auc_score = roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])
        test_accs.append(acc)
        test_aurocs.append(auc_score)
        test_accs_balanced.append(acc_b)
        conf_matrices.append(conf_matrix)
        print(f"Fold {repeat}: test accuracy = {acc:.4f}, test balanced accuracy = {acc_b:.4f}, test AUROC = {auc_score:.4f}")

    print()
    print("Mean test accuracy: {:.4f}".format(np.mean(test_accs)))
    print("Std test accuracy:  {:.4f}".format(np.std(test_accs)))

    print("Mean balanced test accuracy: {:.4f}".format(np.mean(test_accs_balanced)))
    print("Std balanced test accuracy:  {:.4f}".format(np.std(test_accs_balanced)))

    print("Mean test AUROC: {:.4f}".format(np.mean(test_aurocs)))
    print("Std test AUROC:  {:.4f}".format(np.std(test_aurocs)))
    return test_accs, test_aurocs, test_accs_balanced, conf_matrices
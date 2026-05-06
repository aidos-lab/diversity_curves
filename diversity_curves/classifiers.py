from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from diversity_curves.utils import make_splits
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc
from diversity_curves.utils import make_splits
from sklearn.svm import SVC
import pandas as pd

# use diff_matrix rows as features and ys as labels

def knn_classifier(X, y, precomputed=True, seed=42, outer_k=10, holdout_test_size=0.1, k=5, groups=None):
    #X = diff_matrix.copy()
    #y = np.array(ys)

    y = np.asarray(y)
    test_accs = []
    n = X.shape[0]

    indices = make_splits(np.array([i for i in range(len(y))]), y, outer_k=outer_k, inner_k=None, holdout_test_size=holdout_test_size, seed=seed, stratify=True, groups=groups)

    for repeat in range(outer_k):
        idx_tr, idx_va, idx_te = indices[repeat]
        if precomputed:
            X_train, X_test = X[idx_tr,:], X[idx_te,:]
            X_train, X_test = X_train[:,idx_tr], X_test[:,idx_tr]
        else:
            X_train, X_test = X[idx_tr,:], X[idx_te,:]
        y_train, y_test = y[idx_tr], y[idx_te]

        #if len(y_test) == 0 or len(y_train) == 0:
        #    print(f"Fold {repeat}: skipped (empty split)")
        #    continue

        # choose k safely (<= n_train)
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

def knn_regressor(X, y, precomputed=True, seed=42):
    #X = diff_matrix.copy()
    #y = np.array(ys)

    y = np.asarray(y)
    test_accs = []
    n = X.shape[0]

    indices = make_splits(np.array([i for i in range(len(y))]), y, outer_k=10, inner_k=None, holdout_test_size=0.1, seed=seed, stratify=False)

    for repeat in range(10):
        idx_tr, idx_va, idx_te = indices[repeat]
        if precomputed:
            X_train, X_test = X[idx_tr,:], X[idx_te,:]
            X_train, X_test = X_train[:,idx_tr], X_test[:,idx_tr]
        else:
            X_train, X_test = X[idx_tr,:], X[idx_te,:]
        y_train, y_test = y[idx_tr], y[idx_te]

        #if len(y_test) == 0 or len(y_train) == 0:
        #    print(f"Fold {repeat}: skipped (empty split)")
        #    continue

        # choose k safely (<= n_train)
        k = min(5, len(y_train))
        if precomputed:
            clf = KNeighborsRegressor(n_neighbors=k, metric="precomputed")
        else:
            clf = KNeighborsRegressor(n_neighbors=k)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc = mean_squared_error(y_test, preds)
        test_accs.append(acc)
        print(f"Fold {repeat}: test MSE = {acc:.4f} (k={k})")

    print()
    print("Mean test MSE: {:.4f}".format(np.mean(test_accs)))
    print("Std test MSE:  {:.4f}".format(np.std(test_accs)))
    return test_accs

def svm_classifier(X, y, precomputed=True, seed=42, outer_k=10, holdout_test_size=0.1):
    #X = diff_matrix.copy()
    #y = np.array(ys)

    y = np.asarray(y)
    test_accs = []
    n = X.shape[0]

    indices = make_splits(np.array([i for i in range(len(y))]), y, outer_k=outer_k, inner_k=None, holdout_test_size=holdout_test_size, seed=seed, stratify=True)

    for repeat in range(outer_k):
        idx_tr, idx_va, idx_te = indices[repeat]
        if precomputed:
            X_train, X_test = X[idx_tr,:], X[idx_te,:]
            X_train, X_test = X_train[:,idx_tr], X_test[:,idx_tr]
        else:
            X_train, X_test = X[idx_tr,:], X[idx_te,:]
        y_train, y_test = y[idx_tr], y[idx_te]

        #if len(y_test) == 0 or len(y_train) == 0:
        #    print(f"Fold {repeat}: skipped (empty split)")
        #    continue

        # choose k safely (<= n_train)
        k = min(5, len(y_train))
        if precomputed:
            clf = SVC(kernel="precomputed")#KNeighborsClassifier(n_neighbors=k, metric="precomputed")
        else:
            clf = SVC()
        
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        test_accs.append(acc)
        print(f"Fold {repeat}: test accuracy = {acc:.4f} (k={k})")

    print()
    print("Mean test accuracy: {:.4f}".format(np.mean(test_accs)))
    print("Std test accuracy:  {:.4f}".format(np.std(test_accs)))
    return test_accs


from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, roc_auc_score, balanced_accuracy_score

def svm_classifier_cv(X, y, precomputed=True, seed=42, outer_k=10, holdout_test_size=0.1, stratify=True, tune=False):
    #X = diff_matrix.copy()
    #y = np.array(ys)

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

            #X_val = X[idx_va,:]
            #X_val = X_val[:,idx_tr]
        else:
            X_train, X_test = X[idx_tr,:], X[idx_te,:]
            X_val = X[idx_va,:]
        y_train, y_test = y[idx_tr], y[idx_te]
        y_val = y[idx_va]

        #if len(y_test) == 0 or len(y_train) == 0:
        #    print(f"Fold {repeat}: skipped (empty split)")
        #    continue

        # choose k safely (<= n_train)
        k = min(5, len(y_train))
        if precomputed:
            clf = SVC(kernel="precomputed", probability=True)#KNeighborsClassifier(n_neighbors=k, metric="precomputed")
        else:
            clf = SVC(probability=True)#KNeighborsClassifier(n_neighbors=k)

        if tune:
            #param_grid = {'kernel': ('linear', 'rbf','poly') , 
            #    'C':[5, 10, 100],
            #    'gamma': [1,0.1,0.01,0.001], 
            #    'degree' : [1,2,3,4,5,6]}

            #grid = GridSearchCV(SVC(probability=True) , param_grid , refit=True)# , verbose=3)
            param_dist = {
                'kernel': ['rbf'],
                'C': loguniform(1e-2, 1e2),
                'gamma': loguniform(1e-4, 1e-1)
            }

            grid = RandomizedSearchCV(SVC(probability=True), param_dist , refit=True,  n_iter=20, cv=3, n_jobs=-1,random_state=42)# , verbose=3)
            grid.fit(X_val,y_val)
            
            #clf.fit(X_train, y_train)
            clf = grid.best_estimator_
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        acc_b = balanced_accuracy_score(y_test, preds)
        conf_matrix = confusion_matrix(y_test, preds)
        #y_score = clf.decision_function(X_test)
        auc_score = roc_auc_score(y_test, clf.predict_proba(X_test), multi_class='ovr')
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


def svm_classifier_bay(X, X2, y, precomputed=True, seed=42, outer_k=10, holdout_test_size=0.1, stratify=True, tune=False):
    """Train on X, test on X2."""

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
            X_train, X_test, X_val = X[idx_tr,:], X2[idx_te,:], X[idx_va,:]
            X_train, X_test, X_val = X_train[:,idx_tr], X_test[:,idx_tr], X_val[:,idx_tr]

            #X_val = X[idx_va,:]
            #X_val = X_val[:,idx_tr]
        else:
            X_train, X_test = X[idx_tr,:], X2[idx_te,:]
            X_val = X[idx_va,:]
        y_train, y_test = y[idx_tr], y[idx_te]
        y_val = y[idx_va]

        #if len(y_test) == 0 or len(y_train) == 0:
        #    print(f"Fold {repeat}: skipped (empty split)")
        #    continue

        # choose k safely (<= n_train)
        k = min(5, len(y_train))
        if precomputed:
            clf = SVC(kernel="precomputed", probability=True)#KNeighborsClassifier(n_neighbors=k, metric="precomputed")
        else:
            clf = SVC(probability=True)#KNeighborsClassifier(n_neighbors=k)

        if tune:
            #param_grid = {'kernel': ('linear', 'rbf','poly') , 
            #    'C':[5, 10, 100],
            #    'gamma': [1,0.1,0.01,0.001], 
            #    'degree' : [1,2,3,4,5,6]}

            #grid = GridSearchCV(SVC(probability=True) , param_grid , refit=True)# , verbose=3)
            param_dist = {
                'kernel': ['rbf'],
                'C': loguniform(1e-2, 1e2),
                'gamma': loguniform(1e-4, 1e-1)
            }

            grid = RandomizedSearchCV(SVC(probability=True), param_dist , refit=True,  n_iter=20, cv=3, n_jobs=-1,random_state=42)# , verbose=3)
            grid.fit(X_val,y_val)
            
            #clf.fit(X_train, y_train)
            clf = grid.best_estimator_
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        acc_b = balanced_accuracy_score(y_test, preds)
        conf_matrix = confusion_matrix(y_test, preds)
        #y_score = clf.decision_function(X_test)
        #auc_score = roc_auc_score(y_true=y_test, y_score=clf.predict_proba(X_test), multi_class='ovr')
        try:
            auc_score = roc_auc_score(y_test, clf.predict_proba(X_test), multi_class='ovr')
        except:
            auc_score = roc_auc_score(y_test, clf.predict_proba(X_test)[:,1], multi_class='ovr')
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

from diversity_curves.utils import make_inner_split

def knn_classifier_fixed(X, y, idx_tr, idx_te, precomputed=True, seed=42, outer_k=10, holdout_test_size=0.1, stratify=True, tune=False):
    #X = diff_matrix.copy()
    #y = np.array(ys)

    y = np.asarray(y)
    test_accs = []
    test_aurocs = []
    test_accs_balanced =[]
    conf_matrices = []
    n = X.shape[0]
    #indices = make_splits(np.array([i for i in range(len(y))]), y, outer_k=outer_k, inner_k=None, holdout_test_size=holdout_test_size, seed=seed, stratify=stratify)
    all_idxs = idx_tr.copy()

    targets = y[idx_tr]

    for repeat in range(outer_k):

        idx_tr, idx_va = make_inner_split(all_idxs, targets, inner_k=10, holdout_test_size=holdout_test_size, seed=repeat)

        #idx_tr, idx_va, idx_te = indices[repeat]
        if precomputed:
            X_train, X_test, X_val = X[idx_tr,:], X[idx_te,:], X[idx_va,:]
            X_train, X_test, X_val = X_train[:,idx_tr], X_test[:,idx_tr], X_val[:,idx_tr]

            #X_val = X[idx_va,:]
            #X_val = X_val[:,idx_tr]
        else:
            X_train, X_test = X[idx_tr,:], X[idx_te,:]
            X_val = X[idx_va,:]
        y_train, y_test = y[idx_tr], y[idx_te]
        y_val = y[idx_va]

        #if len(y_test) == 0 or len(y_train) == 0:
        #    print(f"Fold {repeat}: skipped (empty split)")
        #    continue

        # choose k safely (<= n_train)
        k = min(5, len(y_train))
        if precomputed:
            clf = KNeighborsClassifier(n_neighbors=k, metric="precomputed")
        else:
            clf = KNeighborsClassifier(n_neighbors=k)

        if tune:
            #param_grid = {'kernel': ('linear', 'rbf','poly') , 
            #    'C':[5, 10, 100],
            #    'gamma': [1,0.1,0.01,0.001], 
            #    'degree' : [1,2,3,4,5,6]}

            #grid = GridSearchCV(SVC(probability=True) , param_grid , refit=True)# , verbose=3)
            param_dist = {
                'n_neighbors': [1, 3, 5, 7, 9, 12, 15],
            }

            grid = RandomizedSearchCV(KNeighborsClassifier(), param_dist , refit=True,  n_iter=20, cv=3, n_jobs=-1,random_state=42)# , verbose=3)
            grid.fit(X_val,y_val)
            
            #clf.fit(X_train, y_train)
            clf = grid.best_estimator_
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        acc_b = balanced_accuracy_score(y_test, preds)
        conf_matrix = confusion_matrix(y_test, preds)
        #y_score = clf.decision_function(X_test)
        #auc_score = roc_auc_score(y_test, clf.predict_proba(X_test), multi_class='ovr')
        try:
            auc_score = roc_auc_score(y_test, clf.predict_proba(X_test), multi_class='ovr')
        except:
            auc_score = roc_auc_score(y_test, clf.predict_proba(X_test)[:,1], multi_class='ovr')
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

def knn_classifier_cv(X, y, precomputed=True, seed=42, outer_k=10, holdout_test_size=0.1, stratify=True, tune=False):
    #X = diff_matrix.copy()
    #y = np.array(ys)

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

            #X_val = X[idx_va,:]
            #X_val = X_val[:,idx_tr]
        else:
            X_train, X_test = X[idx_tr,:], X[idx_te,:]
            X_val = X[idx_va,:]
        y_train, y_test = y[idx_tr], y[idx_te]
        y_val = y[idx_va]

        #if len(y_test) == 0 or len(y_train) == 0:
        #    print(f"Fold {repeat}: skipped (empty split)")
        #    continue

        # choose k safely (<= n_train)
        k = min(5, len(y_train))
        if precomputed:
            clf = KNeighborsClassifier(n_neighbors=k, metric="precomputed")
        else:
            clf = KNeighborsClassifier(n_neighbors=k)

        if tune:
            #param_grid = {'kernel': ('linear', 'rbf','poly') , 
            #    'C':[5, 10, 100],
            #    'gamma': [1,0.1,0.01,0.001], 
            #    'degree' : [1,2,3,4,5,6]}

            #grid = GridSearchCV(SVC(probability=True) , param_grid , refit=True)# , verbose=3)
            param_dist = {
                'n_neighbors': [1, 3, 5, 7, 9, 12],
            }

            grid = RandomizedSearchCV(KNeighborsClassifier(), param_dist , refit=True,  n_iter=20, cv=3, n_jobs=-1,random_state=42)# , verbose=3)
            grid.fit(X_val,y_val)
            
            #clf.fit(X_train, y_train)
            clf = grid.best_estimator_
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        acc_b = balanced_accuracy_score(y_test, preds)
        conf_matrix = confusion_matrix(y_test, preds)
        #y_score = clf.decision_function(X_test)
        #auc_score = roc_auc_score(y_test, clf.predict_proba(X_test), multi_class='ovr')
        try:
            auc_score = roc_auc_score(y_test, clf.predict_proba(X_test), multi_class='ovr')
        except:
            auc_score = roc_auc_score(y_test, clf.predict_proba(X_test)[:,1], multi_class='ovr')
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

def svm_regressor(X, y, precomputed=True, seed=42, outer_k=10, holdout_test_size=0.1):
    #X = diff_matrix.copy()
    #y = np.array(ys)

    y = np.asarray(y)
    test_accs = []
    n = X.shape[0]


    indices = make_splits(np.array([i for i in range(len(y))]), y, outer_k=outer_k, inner_k=None, holdout_test_size=holdout_test_size, seed=seed, stratify=False)

    for repeat in range(outer_k):
        idx_tr, idx_va, idx_te = indices[repeat]
        if precomputed:
            X_train, X_test = X[idx_tr,:], X[idx_te,:]
            X_train, X_test = X_train[:,idx_tr], X_test[:,idx_tr]
        else:
            X_train, X_test = X[idx_tr,:], X[idx_te,:]
        y_train, y_test = y[idx_tr], y[idx_te]

        #if len(y_test) == 0 or len(y_train) == 0:
        #    print(f"Fold {repeat}: skipped (empty split)")
        #    continue

        # choose k safely (<= n_train)
        k = min(5, len(y_train))
        if precomputed:
            from sklearn.svm import SVR
            clf = SVR(kernel="precomputed")
        else:
            from sklearn.svm import SVR
            clf = SVR()
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc = mean_squared_error(y_test, preds)
        test_accs.append(acc)
        print(f"Fold {repeat}: test MSE = {acc:.4f} (k={k})")

    print()
    print("Mean test MSE: {:.4f}".format(np.mean(test_accs)))
    print("Std test MSE:  {:.4f}".format(np.std(test_accs)))
    return test_accs

def linear_classifier(X, y, seed=42, outer_k=10, holdout_test_size=0.1):
    # find best cutoff on df_areas["area"] to predict df_areas["label"]
    #make_splits(all_idxs, targets, outer_k=10, inner_k=None, holdout_test_size=0.1, seed=42, stratify=True)

    y = np.asarray(y)
    areas = X
    labels = y
    test_accs = []

    indices = make_splits(np.array([i for i in range(len(areas))]), labels, outer_k=outer_k, inner_k=None, holdout_test_size=holdout_test_size, seed=seed, stratify=True)

    for repeat in range(outer_k):
        # evaluate many candidate thresholds
        ths = np.linspace(areas.min() - 1e-6, areas.max() + 1e-6, 400)

        #index_test = np.arange(len(areas)) % 10 == repeat
        idx_tr, idx_va, idx_te = indices[repeat]
        areas_train = areas[idx_tr]
        labels_train = labels[idx_tr]
        areas_test = areas[idx_te]
        labels_test = labels[idx_te]

        # initialize with default prediction arrays so we can safely index later
        best_acc = (None, -1, np.zeros_like(labels_train))
        best_acc_test = (None, -1, np.zeros_like(labels_test))
        best_f1 = (None, -1, np.zeros_like(labels_train))
        best_youden = (None, -1)

        for t in ths:
            # try both directions (predict 1 when area >= t, and when area <= t)
            preds_ge = (areas_train >= t).astype(int)
            preds_le = (areas_train <= t).astype(int)

            preds_ge_test = (areas_test >= t).astype(int)
            preds_le_test = (areas_test <= t).astype(int)

            for preds, preds_test in ((preds_ge, preds_ge_test), (preds_le, preds_le_test)):
                acc = accuracy_score(labels_train, preds)
                # avoid errors when no positive predictions by setting zero_division
                f1 = f1_score(labels_train, preds, zero_division=0)

                if acc > best_acc[1]:
                    best_acc = (t, acc, preds.copy())
                    best_acc_test = (t, accuracy_score(labels_test, preds_test), preds_test.copy())

                if f1 > best_f1[1]:
                    best_f1 = (t, f1, preds.copy())

        # summary prints
        #print("Fold {}:".format(repeat))
        #print("Best accuracy threshold (train): {:.6f}, accuracy: {:.4f}".format(best_acc[0], best_acc[1]))
        #print("Confusion matrix (best-acc, train):\n", confusion_matrix(labels_train, best_acc[2]))
        if best_acc_test[0] is not None:
            print(f"Fold {repeat}: test accuracy = {best_acc_test[1]:.4f} (threshold={best_acc_test[0]:.6f})")
            #print("Best accuracy threshold (test, using same rule): {:.6f}, accuracy: {:.4f}".format(best_acc_test[0], best_acc_test[1]))
            #print("Confusion matrix (best-acc, test):\n", confusion_matrix(labels_test, best_acc_test[2]))
            test_accs.append(best_acc_test[1])
        print()
    print("Mean test accuracy: {:.4f}".format(np.mean(test_accs)))
    print("Std test accuracy:  {:.4f}".format(np.std(test_accs)))
    return test_accs


# ------------------------------------------------------------------------
# Example: WL + kNN for a PyTorch Geometric dataset (e.g., MUTAG)
# ------------------------------------------------------------------------

def wl_knn_classify(dataset, labels, k=5, seed=42, iters=2):
    labels = np.asarray(labels)
    X, y = wl_dataset_features(dataset, num_iterations=iters)
    test_accs = []

    indices = make_splits(np.array([i for i in range(len(labels))]), labels, outer_k=10, inner_k=None, holdout_test_size=0.1, seed=seed, stratify=True)

    for repeat in range(10):
        # evaluate many candidate thresholds
        #ths = np.linspace(areas.min() - 1e-6, areas.max() + 1e-6, 400)

        #index_test = np.arange(len(areas)) % 10 == repeat
        idx_tr, idx_va, idx_te = indices[repeat]

        X_train, X_test = X[idx_tr,:], X[idx_te,:]
        y_train, y_test = y[idx_tr], y[idx_te]
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(X_train, y_train)

        acc = clf.score(X_test, y_test)
        print(f"Fold {repeat}: test accuracy = {acc:.4f} (k={k})")
        test_accs.append(acc)
    return test_accs#, clf


import torch
from collections import defaultdict
from torch_geometric.data import Data
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import hashlib

### Check if we trust this, generated code

def hash_label(label):
    """Hash a label (string or int) into a short integer."""
    return int(hashlib.md5(str(label).encode()).hexdigest(), 16) % (10**8)


def wl_features(data, num_iterations=2):
    """
    Compute WL subtree hashed features for a single graph.
    Returns a dict-like histogram {hash: count}.
    """
    edge_index = data.edge_index
    x = data.x.squeeze().tolist()  # assume node labels exist
    if isinstance(x, int):
        x = [x]

    # Initial labels
    labels = [hash_label(l) for l in x]

    # adjacency list
    adj = defaultdict(list)
    for src, dst in edge_index.t().tolist():
        adj[src].append(dst)
        adj[dst].append(src)

    feature_count = defaultdict(int)

    # Count initial labels
    for lbl in labels:
        feature_count[lbl] += 1

    # WL iterations
    for _ in range(num_iterations):
        new_labels = []
        for node in range(len(labels)):
            neigh = sorted(labels[n] for n in adj[node])
            merged = (str(labels[node]) + "_" + "_".join(map(str, neigh)))
            new_labels.append(hash_label(merged))
        labels = new_labels

        # Count new labels
        for lbl in labels:
            feature_count[lbl] += 1

    return feature_count


def wl_dataset_features(dataset, num_iterations=2):
    """
    Convert an entire dataset into a WL feature matrix (NumPy array).
    """
    feature_dicts = []
    global_hash_set = set()

    # First pass: build all feature dicts, track all hashes
    for data in dataset:
        f = wl_features(data, num_iterations)
        feature_dicts.append(f)
        global_hash_set |= set(f.keys())

    # Assign each hash an index
    hash_to_idx = {h: i for i, h in enumerate(sorted(global_hash_set))}
    dim = len(hash_to_idx)

    # Second pass: build dense feature matrix
    X = np.zeros((len(dataset), dim), dtype=np.float32)
    y = np.zeros(len(dataset), dtype=np.int64)

    for i, (data, fdict) in enumerate(zip(dataset, feature_dicts)):
        for h, count in fdict.items():
            X[i, hash_to_idx[h]] = count
        y[i] = data.y.item()

    return X, y


# ------------------------
# Usage example:
# ------------------------
# from torch_geometric.datasets import TUDataset
# dataset = TUDataset(root='data', name='MUTAG')
# acc, clf = wl_knn_classify(dataset, k=5, iters=2)
# print("Accuracy:", acc)

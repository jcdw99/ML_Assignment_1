from sklearn import tree
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


save_model = False
k_split_trials = 20

def get_real_test_data():
    data_test = pd.read_csv('cleaner_test.csv')
    data_test = data_test.drop('Unnamed: 0', axis=1)
    x_test = data_test.drop(['target'], axis=1)
    y_test = data_test.target
    return (x_test, y_test)

def gen_train_data(fix_imbalance=True):
    data_train = pd.read_csv('cleaner_train.csv')
    data_train = data_train.drop('Unnamed: 0', axis=1)
    x_train = data_train.drop(['target'], axis=1)
    y_train = data_train.target
    split_seed = int(np.random.random()*1000)
    print('Split_Seed:', split_seed)
    k =  7
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=split_seed)
    if fix_imbalance:
        sm = SMOTE(sampling_strategy='auto', k_neighbors=k, random_state=1)
        x_res, y_res = sm.fit_resample(x_train, y_train)
    return (x_res, y_res, x_test, y_test)

def read_saved_model():
    filename = 'models/tree_model.trained'
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model

def test_unseen_data(model):
    print('             ----------RUNNING REAL DATA------------')
    x_data, y_data = get_real_test_data()
    y_pred = model.predict(x_data)
    conf_mat = confusion_matrix(y_data, y_pred)
    plt.figure(figsize=(7,5))
    sns.heatmap(conf_mat, annot=True)
    plt.xlabel("Prediction")
    plt.ylabel("Truth")
    plt.show()
    print(classification_report(y_data, y_pred))

def fit_model_kfolds():
    data_train = pd.read_csv('cleaner_train.csv')
    data_train = data_train.drop('Unnamed: 0', axis=1)
    X = data_train.drop(['target'], axis=1)
    y = data_train.target

    kf = KFold(n_splits=10, shuffle=True)
    criterions = ['gini', 'entropy', 'log_loss']
    splitters = ['best', 'random']

    for crit in criterions:
        for splitter in splitters:
            means = []
            stds = []    
            for i in range(k_split_trials):
                kf = KFold(n_splits=10, shuffle=True)
                accs = []

                for train_index, test_index in kf.split(X):
                    X_train, X_test = X.loc[train_index], X.loc[test_index]
                    y_train, y_test = y.loc[train_index], y.loc[test_index]
                    clf = tree.DecisionTreeClassifier(criterion=crit, splitter=splitter)
                    clf = clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)
                    accs.append(accuracy_score(y_test, y_pred))
                
                desc = np.array(pd.DataFrame(accs).describe())
                means.append(desc[1][0])
                stds.append(desc[2][0])

            means.append(np.mean(means))
            stds.append(np.mean(stds))
            data = {
                'means':means,
                'std': stds
            }

            print("tree_" + splitter + "_" + crit  + "  leads to mean: " + str(np.mean(np.array(means)))[:8])
            print("tree_" + splitter + "_" + crit  + "  leads to std: " + str(np.std(np.array(means)))[:8])
            print('\n')
            pd.DataFrame(data).to_csv('results/tree/' + splitter + "_" + crit + ".csv", index=None)

def fit_model(x_train, y_train, x_test, y_test):
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(x_train, y_train)

    if save_model:
        filename = 'tree_model.trained'
        pickle.dump(clf, open(filename, 'wb'))

    y_pred = clf.predict(x_test)
    conf_mat = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(7,5))
    sns.heatmap(conf_mat, annot=True)
    plt.xlabel("Prediction")
    plt.ylabel("Truth")
    plt.show()
    print(clf.score(x_test, y_test))
    print(classification_report(y_test, y_pred))




fit_model_kfolds()
exit()
real_mode = False
# x_train, y_train, x_test, y_test = gen_train_data()
# fit_model(x_train, y_train, x_test, y_test)
if real_mode:
    knn = read_saved_model()
    test_unseen_data(knn)


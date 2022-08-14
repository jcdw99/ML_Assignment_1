from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from scipy.spatial import distance

import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np



# k = 7
save_model = True
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
   
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=split_seed)
    if fix_imbalance:
        sm = SMOTE(sampling_strategy='minority', k_neighbors=7, random_state=1)
        x_res, y_res = sm.fit_resample(x_train, y_train)
        return (x_res, y_res, x_test, y_test)
    return (x_train, y_train, x_test, y_test)
def read_saved_model():
    filename = 'models/knn_model.trained'
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
    ks = [3, 5, 7, 9]
    weights = ['uniform', 'distance']
    distances = ['manhattan', 'euclidean', 'minkowski', 'chebyshev']

    for k in ks:
        for weight in weights:
            for dist in distances:
                means = []
                stds = []

                for i in range(k_split_trials):
                    kf = KFold(n_splits=10, shuffle=True)
                    accs = []

                    for train_index, test_index in kf.split(X):
                        X_train, X_test = X.loc[train_index], X.loc[test_index]
                        y_train, y_test = y.loc[train_index], y.loc[test_index]
                        neigh = KNeighborsClassifier(n_neighbors=k, weights=weight, metric=dist)
                        neigh.fit(X_train, y_train)
                        y_pred = neigh.predict(X_test)
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

                print("knn_" + str(k) + "_" + weight + "_" + dist + "  leads to mean: " + str(np.mean(np.array(means)))[:8])
                print("knn_" + str(k) + "_" + weight + "_" + dist + "  leads to std: " + str(np.std(np.array(means)))[:8])
                print('\n')
                pd.DataFrame(data).to_csv("results/knn/" + str(k) + "_" + weight + "_" + dist + ".csv", index=None)



def fit_model(x_train, y_train, x_test, y_test, knn=True):
    neigh = KNeighborsClassifier(n_neighbors=3, metric='manhattan', weights='uniform')
    neigh.fit(x_train, y_train)

    if save_model:
        filename = 'knn_model.trained'
        pickle.dump(neigh, open(filename, 'wb'))

    y_pred = neigh.predict(x_test)
    conf_mat = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(7,5))
    sns.heatmap(conf_mat, annot=True)
    plt.xlabel("Prediction")
    plt.ylabel("Truth")
    plt.show()
    print(neigh.score(x_test, y_test))
    print(classification_report(y_test, y_pred))




fit_model_kfolds()
exit()
real_mode = False
# x_train, y_train, x_test, y_test = gen_train_data()
# fit_model(x_train, y_train, x_test, y_test)
# exit()
if real_mode:
    knn = read_saved_model()
    test_unseen_data(knn)


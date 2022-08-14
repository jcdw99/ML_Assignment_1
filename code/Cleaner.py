import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

def clean_csv():
    global raw
    train = True
    avg_mode = False
    path_to_csv = 'resources/breastCancerTrain.csv' if train else 'resources/breastCancerTest.csv'
    raw = pd.read_csv(path_to_csv, sep=';')
    raw = raw.drop(['id', 'Gender'], axis=1)

    def clean_diagnosis():
        global raw
        raw['diagnosis'] = np.array((raw['diagnosis'] == 'M')).astype(int)

    def clean_float_field_avg_used(field):
        global raw
        raw[field] = raw[field].map(lambda x: -1 if (x == '?' or x == '0') else np.abs(float(x)))
        raw[field] = pd.to_numeric(raw[field])

        begnign_avg = np.mean(np.array(raw.loc[(raw['diagnosis'] == 0) & (raw[field] > 0)][field]))
        malig_avg = np.mean(np.array(raw.loc[(raw['diagnosis'] == 1) & (raw[field] > 0)][field]))

        bools = (raw[field] < 0) & (raw['diagnosis'] == 1)
        raw.loc[bools, field] = malig_avg
        bools = (raw[field] < 0) & (raw['diagnosis'] == 0)
        raw.loc[bools,field] = begnign_avg

    def clean_float_field_knn_used():
        global raw
        float_fields = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean',
                'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean', 'radius_se',
                'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se',
                'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
                'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst', 'Bratio']
        
        for field in float_fields:
            raw[field] = raw[field].map(lambda x: np.nan if ((x =='?') or (x == '0')) else np.abs(float(x)))
            description = raw[field].describe()
            upper = description['75%'] + 0.5 * (description['75%'] - description['25%'])
            lower = description['25%'] - 0.5 * (description['75%'] - description['25%'])
            raw[field] = raw[field].map(lambda x: np.nan if (x > upper) or (x < lower) else np.abs(float(x)))

        raw[field] = pd.to_numeric(raw[field])

        imputer = KNNImputer(n_neighbors=2)
        raw = imputer.fit_transform(raw)
        

    clean_diagnosis()
    float_fields = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean',
        'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean', 'radius_se',
        'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se',
        'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
        'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst', 'Bratio']

    if avg_mode:
        for field in float_fields:
            print('trying', field)
            clean_float_field_knn_used(field)
    else:
        clean_float_field_knn_used()

    out_path = 'cleaner_train.csv' if train else 'cleaner_test.csv'
    raw = pd.DataFrame(raw)

    raw = raw.rename(columns={0: 'target', 1:'radius_mean', 2:'texture_mean', 3:'perimeter_mean', 4:'area_mean', 5:'smoothness_mean', 6:'compactness_mean',
                7:'concavity_mean', 8:'concave points_mean', 9:'symmetry_mean', 10:'fractal_dimension_mean', 11:'radius_se',
                12:'texture_se', 13:'perimeter_se', 14:'area_se', 15:'smoothness_se', 16:'compactness_se', 17:'concavity_se', 18:'concave points_se',
                19:'symmetry_se', 20:'fractal_dimension_se', 21:'radius_worst', 22:'texture_worst', 23:'perimeter_worst', 24:'area_worst', 25:'smoothness_worst',
                26:'compactness_worst', 27:'concavity_worst', 28:'concave points_worst', 29:'symmetry_worst', 30:'fractal_dimension_worst', 31:'Bratio'})
    
    norm = True
    if norm:
        for field in float_fields:
            raw[field] = (raw[field] - (raw[field]).mean()) / raw[field].std()

    raw.to_csv(out_path, index=True)
    print(raw['target'].value_counts()) 
clean_csv()
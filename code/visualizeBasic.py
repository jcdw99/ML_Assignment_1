import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

cleaner = pd.read_csv("cleaner_train.csv")
cleaner = cleaner.drop('Unnamed: 0', axis=1)
malig = cleaner.loc[cleaner['target'] == 1]
ben = cleaner.loc[cleaner['target'] == 0]

def describe_data():
    float_fields = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean',
        'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean', 'radius_se',
        'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se',
        'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
        'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst', 'Bratio']
    
    for field in float_fields:
        list1 = ['Benign', *list(np.array(ben[field].describe()))]
        list2 = ['Malig', *list(np.array(malig[field].describe()))]
        print('\n')
        print('                                            ' + field + '                       ')
        print(tabulate([list1, list2], headers=['Type', 'Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'max']))
        print('\n')

def scatter(field1, field2):
    plt.scatter(malig[field1], malig[field2], color='red', label='Malignant')
    plt.scatter(ben[field1], ben[field2], color='blue', label='Benign')
    plt.legend()
    plt.xlabel("Area Worst")
    plt.ylabel("Area SE")
    plt.show()

describe_data()
scatter('area_worst', 'area_se')

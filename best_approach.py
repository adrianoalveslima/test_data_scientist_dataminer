from operator import index
from joblib import dump, load
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
from util import read_dataset, input_data
from sklearn.preprocessing import StandardScaler
import pandas as pd


def main():
    dataset_train = read_dataset("https://raw.githubusercontent.com/dataminerdbm/test_data_scientist/main/treino.csv")
    dataset_test = read_dataset("https://raw.githubusercontent.com/dataminerdbm/test_data_scientist/main/teste.csv")

    dataset_train.replace(to_replace=[None], value=np.nan, inplace=True)
    dataset_test.replace(to_replace=[None], value=np.nan, inplace=True)

    raw_dataset_values_train = dataset_train.drop(columns=['inadimplente'])

    transformed_values_train = input_data(raw_dataset_values_train)
    transformed_values_test = input_data(dataset_test)

    # Deve-se utilizar a mesma escala de dados para o treinamento e teste
    # https://datascience.stackexchange.com/questions/27615/should-we-apply-normalization-to-test-data-as-well
    scaler = StandardScaler()
    standardized_values_train = scaler.fit_transform( transformed_values_train )
    standardized_values_test = scaler.transform( transformed_values_test )

    standardized_values_train = pd.DataFrame(standardized_values_train, columns=raw_dataset_values_train.keys())
    standardized_values_test = pd.DataFrame(standardized_values_test, columns=dataset_test.keys())

    train_x = standardized_values_train
    train_y = dataset_train.inadimplente

    test_x = standardized_values_test

    #model = AdaBoostClassifier(n_estimators=100)
    #model.fit(train_x, train_y)

    filename = 'test_data_scientist_dataminer/modelo-adaboost.joblib'
    #dump(model, filename)

    loaded_model = load(filename)

    predictions = loaded_model.predict(test_x)

    dataset_test_raw_df = read_dataset("https://raw.githubusercontent.com/dataminerdbm/test_data_scientist/main/teste.csv")
    dataset_test_raw_df['inadimplente'] = predictions
    dataset_test_raw_df.to_csv("test_data_scientist_dataminer/teste.csv", index=False)


if __name__ == "__main__":
    main()
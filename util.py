from sklearn.metrics import f1_score
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from numpy import nan


# Utiliza-se micro devido ao desbalanceamento entre as classes
def train_test_model(model, train_x, train_y, test_x, test_y):
    model.fit(train_x, train_y)

    predictions = model.predict(test_x)

    f_measure = f1_score(test_y, predictions, average='micro') * 100
    print("F-measure de %.2f%%" % f_measure)

    return model


#def delete_rows_missing(dataset):
    # Exclui linhas com valores faltantes
    #dataset.dropna(inplace=True)
    # Sumariza a forma dos dados com linhas faltantes removidas
    #print(dataset.shape)

    # Verifica-se que as colunas 'salario_mensal' e 'numero_de_dependentes' são as que possuem valores faltantes
    #print(dataset.isnull().sum())


def print_results_cv(results):
    mean = results.mean()

    standard_deviation = results.std()
    mean_f_measure = (mean * 100)
    print("F-measure média: %.2f" % mean_f_measure)
    print("Intervalo de f-measure: [%.2f, %.2f]" % ((mean - 2 * standard_deviation) * 100, (mean + 2 * standard_deviation) * 100))


# carregar conjunto de dados a partir da url fornecida
def read_dataset(uri):
    dataset = pd.read_csv(uri)

    # print(dataset.describe())
    # print(dataset.head(20))

    # print(dataset.shape)
    return dataset


# Inputar dados
def input_data(raw_dataset_values):
    imputer = SimpleImputer(missing_values=nan, strategy='median')
    transformed_values = imputer.fit_transform(raw_dataset_values)

    return transformed_values


# Reescalando dados
def rescale_data(transformed_values, raw_dataset_values):
    scaler = StandardScaler()
    scaler.fit(transformed_values)
    standardized_values = scaler.transform(transformed_values)
    standardized_values = pd.DataFrame(
        standardized_values, columns=raw_dataset_values.keys())

    return standardized_values
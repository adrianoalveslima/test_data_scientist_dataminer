import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.dummy import DummyClassifier
from util import train_test_model, print_results_cv, read_dataset, input_data, rescale_data
from sklearn.model_selection import GroupKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
import time
from sklearn.manifold import TSNE
from sklearn.feature_selection import RFE, RFECV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from imblearn.under_sampling import RandomUnderSampler


# Calculando correlação e criando figura
def calc_corr_fig(standardized_values):
    plt.figure(figsize=(17, 15))
    matrix_correlation = standardized_values.corr()

    sns.heatmap(matrix_correlation, annot=True, fmt=".1f")
    plt.show()


# Visualizar em 2 dimensões os dados para facilitar o direcionamento da análise
def tsne_scatterplot(x_without_corr_feat, y):
    tsne = TSNE(n_components=2)
    tsne_x = tsne.fit_transform(x_without_corr_feat)
    sns.scatterplot(x=tsne_x[:, 0], y=tsne_x[:, 1], hue=y)
    plt.show()


def select_features(model, train_x, train_y, picker, test_x=None):
    picker.fit(train_x, train_y)
    train_picker = picker.transform(train_x)
    test_picker = None
    if test_x is not None:
        test_picker = picker.transform(test_x)

    return {'train_picker': train_picker, 'test_picker': test_picker, 'picker': picker}


def validate_models_holdout(train_x, train_y, test_x, test_y, models, k_size):
    for model in models:
        print("Imprimindo resultados da abordagem holdout para o %s" %
              model.__class__)
        t0 = time.time()

        model = train_test_model(model, train_x, train_y, test_x, test_y)

        # Se o classificador não possuir lista de importância de características, seleciona-se por métrica de filtro
        try:
            picker = select_features(model, train_x, train_y, RFE(
                estimator=model, n_features_to_select=k_size, step=1), test_x)
        except:
            picker = select_features(model, train_x, train_y, SelectKBest(
                mutual_info_classif, k=k_size), test_x)

        print("Selecionando-se as características com %s" % picker['picker'])
        train_test_model(model, picker['train_picker'],
                         train_y, picker['test_picker'], test_y)

        print("Tempo do modelo %s: %d" %
              (model.__class__, round(time.time()-t0, 3)))


def validate_models_cv(x, y, random_groups, models, k_size):
    for model in models:
        print("Imprimindo resultados da abordagem por validação cruzada para o %s" %
              model.__class__)

        cv = GroupKFold(n_splits=5)

        t0 = time.time()
        results = cross_val_score(
            model, x, y, cv=cv, groups=random_groups, scoring='f1_micro')
        print_results_cv(results)

        try:
            picker = select_features(model, x, y, RFECV(estimator=model, cv=5, step=1, scoring="f1_micro"), None)
        except:
            picker = select_features(model, x, y, SelectKBest(
                mutual_info_classif, k=k_size), None)

        print("Selecionando-se as características com %s" % picker['picker'])
        results = cross_val_score(
            model, picker['train_picker'], y, cv=cv, groups=random_groups, scoring='f1_micro')

        print("Tempo do modelo %s: %d" %
              (model.__class__, round(time.time()-t0, 3)))

        print_results_cv(results)


def main():
    dataset = read_dataset("https://raw.githubusercontent.com/dataminerdbm/test_data_scientist/main/treino.csv")

    dataset.replace(to_replace=[None], value=np.nan, inplace=True)

    raw_dataset_values = dataset.drop(columns=['inadimplente'])

    transformed_values = input_data(raw_dataset_values)

    standardized_values = rescale_data(transformed_values, raw_dataset_values)

    # calc_corr_fig(standardized_values)

    x = standardized_values
    # Remove-se as demais características correlacionadas, mantendo-se apenas uma
    x_without_corr_feat = standardized_values.drop(
        columns=['vezes_passou_de_30_59_dias', 'numero_de_vezes_que_passou_60_89_dias'])
    y = dataset.inadimplente

    SEED = 7707
    np.random.seed(SEED)
    # Realiza-se a estratificação dos dados tendo em vista o desbalanceamento da base
    train_x, test_x, train_y, test_y = train_test_split(
        x, y, test_size=0.3, stratify=y)

    train_x_without_corr_feat, test_x_without_corr_feat, train_y_without_corr_feat, test_y_without_corr_feat = train_test_split(
        x_without_corr_feat, y, test_size=0.3, stratify=y)

    undersample = RandomUnderSampler(sampling_strategy='majority')

    X_without_corr_feat_under, y_without_corr_feat_under = undersample.fit_resample(x_without_corr_feat, y)
    x_under, y_under = undersample.fit_resample(x, y)
    train_x_under, train_y_under = undersample.fit_resample(train_x, train_y)
    train_x_without_corr_feat_under, train_y_without_corr_feat_under = undersample.fit_resample(train_x_without_corr_feat, train_y_without_corr_feat)

    #tsne_scatterplot(x_without_corr_feat, y)

    # Os classificadores validados foram escolhidos de acordo com o aspecto da base de dados:
    # características numéricas, multidimensional com alto número de instâncias e problema não linearmente separável
    models = [DummyClassifier(), KNeighborsClassifier(), DecisionTreeClassifier(),
              GaussianNB(), AdaBoostClassifier(n_estimators=100), RandomForestClassifier(),
              BaggingClassifier(base_estimator=GaussianNB(), n_estimators=100)]
    k_size = 5

    # Criando aleatoridade nos grupos de folds (para evitar repetição). Abordagem mais adequada para bases desbalanceadas
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html#sklearn.model_selection.GroupKFold
    x_under['idade_r'] = x_under.idade + np.random.randint(-2, 3, size=14662)
    x_under.idade_r = x_under.idade + abs(x_under.idade.min()) + 1

    print("Validando modelos com todas as características")
    validate_models_cv(x_under, y_under, x_under.idade_r, models, k_size)
    validate_models_holdout(train_x_under, train_y_under, test_x, test_y, models, k_size)

    print("Validando modelos sem as características correlacionadas")
    validate_models_cv(X_without_corr_feat_under, y_without_corr_feat_under, x_under.idade_r, models, k_size)
    validate_models_holdout(train_x_without_corr_feat_under, train_y_without_corr_feat_under,
                            test_x_without_corr_feat, test_y_without_corr_feat, models, k_size)


if __name__ == "__main__":
    main()

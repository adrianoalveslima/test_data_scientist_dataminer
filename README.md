### Introdução
A análise do problema apresentado foi iniciada a partir do estudo das características do conjunto de dados. Assim, gerou-se uma matriz de correlação para averiguar eventuais características que poderiam ser descartadas, conforme Figura 'correlacao_caracteristicas.png'. Para verificar o tipo do problema, gerou-se um gráfico (tsne_scatterplort.png) através da transformação do espaço dimensional em 2 eixos (t-SNE), percebendo-se tratar de um problema não linearmente separável. Partindo destas análises preliminares, optou-se por validar métodos que lidam melhor com problemas não linearmente separáveis e também aplicar diferentes métodos de seleção de características. Para isso, foram utilizadas duas abordagens de validação, sendo a primeira por holdout dividindo a base de treinamento em 70% e 30% para treinamento e validação, respectivamente. Em comparação, também optou-se por uma validação cruzada com dados estratificados. Verificou-se que a base de treino está desbalanceada com 7.331 inadimplentes e 102.669 adimplentes. Sendo assim, métricas como Precision, Recall, F_Score e curva ROC são mais adequadas para avaliação. Tem-se 110.000 instâncias de treino e 40.000 instâncias de teste. Desta forma, também foi realizado um undersampling na base de treino para corrigir o desbalanceamento das classes.

Com relação a estrutura dos arquivos do projeto:
**best_approach.py** Este arquivo é a implementação final a partir da melhor abordagem verificada nas validações. Obs.: algumas linhas estão comentadas, pois o modelo já foi previamente construído e está salvo neste projeto com o nome 'modelo-rf.joblib'.

**validate_best_model.py** Neste arquivo está a implementação das diferentes análises realizadas: função que gera a matriz de correlação, geração do gráfico de distribuição das instâncias em 2 dimensões por meio do t-SNE e métodos de validação por holdout ou validação cruzada. A saída desse arquivo está em 'validate_best_model_output.txt'.

**util.py** Contém funções comumente utilizadas nos diferentes arquivos.

### Pré-processamento:
Características não normalizadas na base original e algumas faltantes. Apesar de alguns métodos terem capacidade de lidar com dados faltantes 
(NaiveBayes, Árvores de Decisão), o scikit-learn não tem suporte. A estratégia para lidar com os dados faltantes foi 'inputar' dados, 
tendo em vista que na exclusão das instâncias com valores 'missing', perderia-se quase 20% da base de treinamento
https://scikit-learn.org/stable/modules/impute.html

Verifica-se que as colunas 'salario_mensal' e 'numero_de_dependentes' são as que possuem valores faltantes.

Percebe-se que há correlação (método de pearson) entre as características 'vezes_passou_de_30_59_dias' e 'numero_de_vezes_que_passou_60_89_dias';
'numero_vezes_passou_90_dias' e 'vezes_passou_de_30_59_dias'; 'numero_vezes_passou_90_dias' e 'numero_de_vezes_que_passou_60_89_dias'.

#### Outras informações:

Foram executadas validações eliminando-se as características correlacionadas e também selecionando as 5 (e outras combinações com 2, 6, 7). Apesar de não haver perca considerável, optou-se por utilizar todos os atributos, tendo em vista que o melhor desempenho foi atingido com o RandomForest (em holdout e validação cruzada), o qual utiliza árvores de decisão como estimador base (o qual faz uma seleção de características naturalmente). Ou seja, para a melhor abordagem, não foi considerada a etapa de seleção de características.

Validou-se também seleção dinâmica de classificadores ou de ensembles com pool (GaussinaNB, DecisionTree e KNN e utilizando todos os classificadores e ensembles da configuração) com diferentes abordagens (KNORA-U, KNORA-E, MCB, METADES). No entanto, os resultados não foram promissores e não foram efetuados testes mais profundos.

O arquivo final com o resultado da avaliação está sob o nome 'teste.csv' e possui a coluna 'inadimplente' com a predição realizada a partir do modelo criado.
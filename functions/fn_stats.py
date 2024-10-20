import pandas as pd



def describe( # Função para otimizar a exibição dos dados no describe
    df, # Passando o dataframe como parâmetro da função
): 
    with pd.option_context(
        'display.float_format', '{:.2f}'.format, # Formatando os dados com 2 casas decimais
        'display.max_columns', None # Removendo a limitação de visualização de apenas 20 colunas
    ):
        display(df.describe()) # Exibindo o describe dos dados



def inspect_outliers( # Função para inspecionar outliers
        dataframe, # Passando o dataframe como parâmetro da função
        column, # Passando a coluna como parâmetro da função
        whisker_width=1.5 # Passando o valor para considerar outliers como parâmetro da função, com valor padrão de 1,5
    ): 

    q1 = dataframe[column].quantile(0.25) # Definindo o 1º quartil
    q3 = dataframe[column].quantile(0.75) # Definindo o 3º quartil
    iqr = q3 - q1 # Calculando o IQR
    lower_bound = q1 - whisker_width * iqr # Calculando o limite inferior
    upper_bound = q3 + whisker_width * iqr # Calculando o limite superior

    return dataframe[ # Retornando o dataframa apenas com as linhas que são outliers inferiores e superiores
        (dataframe[column] < lower_bound) | (dataframe[column] > upper_bound) 
    ]
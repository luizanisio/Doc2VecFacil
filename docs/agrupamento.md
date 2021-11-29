# Dicas de uma forma rápida para agrupar os documentos vetorizados
- Com vetores que possibilitam a comparação de similaridade de documentos, pode-se agrupar por similaridade um conjunto de arquivos, por exemplo.
- O código abaixo permite o uso do modelo treinado com o [`Doc2VecFacil`](../README.md) para gerar uma planilha excel com o nome dos arquivos e os grupos formados pela similaridade entre eles.
  
- [`UtilAgrupamentoFacil`](../src/util_agrupamento_facil.py)  

> :bulb: <sub>Nota: existem diversas formas de agrupar vetores, como [`HDBScan`](https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html), [`DBScan`](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html), [`K-Means`](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html), dentre outros. A forma apresentada aqui é mais uma forma, simples, e que não exige identificar número de grupos e nem extrapola a similaridade por critérios de borda ou continuidade. Mas cada caso é um caso e pode-se aplicar a técnica que melhor resolva o problema.</sub>

## Para usar o código de agrupamento

- Esse é um exemplo simples de agrupamento, ele pode ser colocado em um serviço, por exemplo, e o usuário escolher os documentos de uma base do `SingleStore` ou `ElasticSearch`, os vetores já estariam gravados e seriam carregados, agrupados e o usuário receberia os grupos ou esses grupos seriam gravados em uma tabela para consulta do usuário.
- Usando o códido por linha de comando:
  - `python util_agrupamento_facil.py -modelo meu_modelo -textos meus_documentos -sim 80`

- os parâmetros de `util_agrupamento_facil.py` são:
  - `-modelo` é a pasta do modelo treinado, podendo ser a pasta com pacote de treinamento ou a pasta final só com o modelo
  - `-textos` é a pasta contendo os arquivos `.txt` que serão agrupados (se omitido, será procurada a pasta textos_grupos na pasta do pacote de treinamento)
  - `-sim` é a similaridade mínima para que um ou mais arquivos sejam agrupados, se não informado será usada a similaridade 90%
  - `-epocas` é a quantidade de épocas para a inferência do vetor de cada arquivo, se não informado será inferido com `3` épocas.
  - `-plotar` faz a redução de dimensões dos vetores agrupados para 2d e plota um gráfico. É uma apresentação interessante, mas em 2d alguns vetores aparecem juntos no gráfico mesmo estando distantes, o que dá uma falsa impressão de proximidade. Mas os vetores realmente próximos estarão representados com a mesma cor.
  - `-texto` inclui uma coluna com até 250 caracteres do texto analisado para auxiliar na análise do agrupamento
 
- será criado um arquivo com o nome da pasta de textos e a similaridade informada `agrupamento {pasta de textos} sim {similaridade}.xlsx` 
- se for usado o parâmetro `-plotar`, será criado o arquivo `agrupamento {pasta de textos} sim {similaridade}.png` também
- exemplo de resultado do agrupamento:

![exemplo de agrupamento de arquivos](../exemplos/img_agrupamento.png?raw=true "agrupamento de arquivos") 

- Exemplo do plot de um agrupamento de 50mil vetores 2d randômicos para ter uma ideia de como os grupos são formados (esquerda). Com um plot de 2mil vetores 300d reduzidos para 2d com a criação de 28 grupos (em cores) para 82 vetores e os outros ficaram órfãos (modelo de teste treinado com 100 épocas apenas). Cada cor representa um grupo formado.

![exemplo plot agrupamento](../exemplos/img_agrupamento_50k_2k.png?raw=true "Exemplo de agrupamento de 50mil vetores 2d randômicos e 2mil vetores 300d de textos")

> :bulb: <sub>Nota: o arquivo usado como principal para capturar outros arquivos similares terá a similaridade 100 na planilha, pode ser considerado o centróide do grupo.</sub>
 
## Como funciona o agrupamento
1. cada arquivo é vetorizado 
2. inicia-se o agrupamento pegando um vetor e buscando todos os com a similaridade mínima e cria-se um grupo
3. pega-se o próximo vetor sem grupo e é feita a busca dos vetores sem grupo com similaridade mínima e cria-se outro grupo
4. o item 3 é repetido até acabarem os vetores carregados
5. se um vetor não tiver vetores similares, fica no grupo `-1` que é o grupo de órfãos
6. uma última verificação é feita avaliando se algum vetor de um grupo estaria mais perto de um centroide de outro grupo criado posteriormente

> :bulb: <sub>Nota: Como toda a vetorização é feita no momento do agrupamento, o processo pode ser demorado, principalmente com muitas épocas na inferência do vetor.</sub><br>
> <sub>- Em uma situação de serviço em produção, os vetores já estariam disponíveis em um banco de dados, elasticsearch ou outra forma de armazenamento, ficando um processo muito rápido de agrupamento.</sub><br>

## Como rodar o agrupamento pelo código
- usando o código para criar a planilha de agrupamento de arquivos de uma pasta:
```python
    from util_agrupamento_facil import UtilAgrupamentoFacil
    PASTA_MODELO = 'meu_modelo'
    PASTA_TEXTOS = 'meus_textos'
    similaridade = 85
    epocas = 3
    
    UtilAgrupamentoFacil.agrupar_arquivos(pasta_modelo=PASTA_MODELO, 
                                          pasta_arquivos=PASTA_TEXTOS, 
                                          similaridade=similaridade,
                                          epocas = epocas,
                                          plotar = True)
```

- usando o código para agrupar vetores carregados do banco de dados, elasticsearch etc.
- será retornado um DataFrame com os dados dos arquivos, grupos, similaridade e centróides.
```python
    from util_agrupamento_facil import UtilAgrupamentoFacil
    similaridade = 85
    vetores, ids = carregar_vetores_do_banco(...)
    # retorna uma lista de tuplas na mesma ordem da lista recebida com [(grupo,similaridade), ...]
    grupos_sim = UtilAgrupamentoFacil.agrupar_vetores(vetores,similaridade)
    
    
    [ print(id, grupo, sim) for id, (grupo, sim) in zip(ids, grupos_sim) ]
```

- usando o código para agrupar vetores de uma lista de dicionários de dados [{'vetor': ..., 'nome': 'xxx', ...}, ...].
- é requisito ter uma coluna `vetor` com o vetor no formato list de floats.
- opcionalmente o parâmetro `dados` pode ser uma lista de vetores [[0.24, 0.55, 0.23 ...], ...]
- será retornado um DataFrame com os dados incluindo as colunas: `grupo`, `similaridade` e `centroide`.
```python
    from util_agrupamento_facil import UtilAgrupamentoFacil
    similaridade = 85
    # suponha que os dados carregados são [{'nome_documento': 'bla bla bla,'vetor': [0.23, 0.56, 0.44, ...], 'data_documento' : '2021-01-01', ...}, ..]
    dados = carregar_vetores_do_banco(...)
    util = UtilAgrupamentoFacil(dados, similaridade=similaridade)
    # arredondando os dados - só para ilustrar
    util.dados['similaridade'] = [round(s,2) for s in util.dados['similaridade']]
    # dados é um DataFrame com os dados originais recebidos do banco incluindo as novas colunas
    print(util.dados)
```

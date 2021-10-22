# Dicas de uma forma rápida de agrupar os documentos vetorizados
- Com vetores que possibilitam a comparação de similaridade de documentos, pode-se agrupar por similaridade um conjunto de arquivos, por exemplo.
- O código abaixo permite o uso do modelo treinado com o [`Doc2VecFacil`](../README.md) para gerar uma planilha excel com o nome dos arquivos e os grupos formados pela similaridade entre eles.
  
- [`UtilAgrupamentoFacil`](./src/util_agrupamento_facil.py)  

> :bulb: <sub>Nota: existem diversas formas de agrupar vetores, como [`HDBScan`](https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html), [`DBScan`](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html), [`K-Means`](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html), dentre outros. A forma apresentada aqui é mais uma forma, simples, e que não exige identificar número de grupos e nem extrapola a similaridade por critérios de borda ou continuidade. Mas cada caso é um caso e pode-se aplicar a técnica que melhor resolva o problema.</sub>

## Para usar o código de agrupamento

- Esse é um exemplo simples de agrupamento, ele pode ser colocado em um serviço, por exemplo, e o usuário escolher os documentos de uma base do `SingleStore` ou `ElasticSearch`, os vetores já estariam gravados e seriam carregados, agrupados e o usuário receberia os grupos ou esses grupos seriam gravados em uma tabela para consulta do usuário.
- Usando o códido por linha de comando:
  - `python util_agrupamento_facil.py -modelo meu_modelo -textos meus_documentos -sim 80`

- os parâmetros de `util_agrupamento_facil.py` são:
  - `-modelo` é a pasta do modelo treinado, podendo ser a pasta com pacote de treinamento ou a pasta final só com o modelo
  - `-textos` é a pasta contendo os arquivos `.txt` que serão agrupados
  - `-sim` é a similaridade mínima para que um ou mais arquivos sejam agrupados, se não informado será usada a similaridade 90%
  - `-epocas` é a quantidade de épocas para a inferência do vetor de cada arquivo, se não informado será inferido com `3` épocas.
 
- será criado um arquivo com o nome da pasta de textos e a similaridade informada `agrupamento {pasta de textos} sim {similaridade}.xlsx` 
- exemplo de resultado do agrupamento:

![exemplo de agrupamento de arquivos](../exemplos/img_agrupamento.png?raw=true "agrupamento de arquivos") 

> :bulb: <sub>Nota: o arquivo usado como principal para capturar outros arquivos similares terá a similaridade 100 na planilha.</sub>
 
## Como funciona o agrupamento
1. cada arquivo é vetorizado 
2. inicia-se o agrupamento pegando um vetor e buscando todos os com a similaridade mínima e cria-se um grupo
3. pega-se o próximo vetor sem grupo e é feita a busca dos vetores sem grupo com similaridade mínima e cria-se outro grupo
4. o item 3 é repetido até acabarem os vetores carregados
5. se um vetor não tiver vetores similares, fica no grupo `-1` que é o grupo de órfãos

> :bulb: <sub>Nota: Como toda a vetorização é feita no momento do agrupamento, o processo pode ser demorado, principalmente com muitas épocas na inferência do vetor.</sub><br>
> <sub>- Em uma situação de serviço em produção, os vetores já estariam disponíveis em um banco de dados, elasticsearch ou outra forma de armazenamento, ficando um processo muito rápido de agrupamento.</sub><br>

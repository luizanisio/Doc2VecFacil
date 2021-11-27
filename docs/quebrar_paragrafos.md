# Utilitário para quebrar parágrafos de treinamento
O doc2vec pode ser treinado com documentos completos ou com parágrafos ou textos menores. 
Esse utilitário ajuda na quebra de documentos grandes em seus parágrafos de treinamento com algumas opções como segue:
- `python util_quebrar_paragrafos.py -entrada pasta_textos -saida pasta_trechos`
- parâmetros:
  - entrada = nome da pasta com os textos originais
  - saida = nome da pasta para gravação dos trechos - será criada se não existir 
  - length = qual o menor tamanho de uma sentença para iniciar uma nova ou unir com a anterior - padrão 1
  - tokens = quantos tokens no mínimo uma sentença precisa ter para ser válida ou ser descartada - padrão 5

### Como é feita a quebra?
1. Primeiro as sentenças são quebradas pelas pontuações `.`, `?` ou `!`.
2. Depois as sentenças quebradas incorretamente por abreviações, por exemplo, são unidas
3. É verificado se cada sentença cumpre com o mínimo de caracteres estipulado em `-length`. Caso não tenho o mínimo, ela é unida com a próxima até atingir o mínimo.
4. Com a sentença cumprindo o mínimo de caracters, é verificado se possui o mínimo de tokens (uma quebra simples por espaços). Se não possuir, será excluída.
5. As sentenças que cumprirem o mínimo de tokens serão gravadas na pasta de saída com um nome sequencial ou, com a opção `-nome`, será usado o prefixo do nome e um complemento sequencial.
   - Caso a sentença já exista na pasta de saída ou nas sentenças geradas, ela será ignorada. É criado um hash de cada sentença para avaliar a duplicidade.

### Exemplos:
- quebrar em sentenças com pelo menos 5 tokens e qualquer tamanho de caracteres 
  - `python util_quebrar_paragrafos.py -entrada meu_modelo/textos_raw -saida meu_modelo/textos_treino -length 0 -tokens 5`

- quebrar em sentenças com pelo menos 10 tokens e 200 caracteres 
  - `python util_quebrar_paragrafos.py -entrada meu_modelo/textos_raw -saida meu_modelo/textos_treino -length 200 -tokens 10`

### Como isso afeta o treino?
- Com trechos pequenos, menores que os documentos, o treinamento terá muito mais documentos para treinar
- Estou treinando alguns modelos em diversos cenários usando a base [`imdb-ptbr - kaggle`](https://www.kaggle.com/luisfredgs/imdb-ptbr) e em breve vou postar as comparações
- Mas como é de se esperar, cada caso é um caso e será necessário fazer testes para avaliar o melhor cenário para os documentos

# Dicas simplificadas de como preparar e treinar um modelo em diversos cenários
Segue abaixo alguns cenários e seus passos.<br>
As dicas vão levar em conta que o seu modelo será criado na pasta `meu_modelo`, mas pode criar a pasta com o nome que quiser, basta passar esse nome no parêmtro.

## 1) Quero treinar sem preparar um vocab:
 - Crie a pasta `meu_modelo`
 - Crie uma subpasta `meu_modelo/textos_treino`
 - Rode o treinamento do modelo:
   - `python util_doc2vec_facil.py -pasta ./meu_modelo -treinar`
  > 💡 Nota: todos os tokens serão treinados, será feita apenas a limpeza simples dos textos para comparações simples já é o suficiente
  > - Após a primeira época será criado o arquivo `vocab_treino.txt` que contém os termos realmente treinados (os disponíveis no vocab e que foram encontrados nos textos de treino)

## 2) Quero usar as palavras sugeridas ou já tenho as minhas:
 - Crie a pasta `meu_modelo`
 - Crie uma subpasta `meu_modelo/textos_treino`
 - Crie uma subpasta `meu_modelo/doc2vecfacil` e coloque os arquivos `VOCAB_BASE_PTBR.txt` e, se quiser que ocorra a fragmentação de termos fora do vocab, coloque também o arquivo `VOCAB_BASE_PTBR_FRAGMENTOS.txt`. Ou use seus arquivos com ou sem esses dois sugeridos. 
 - Rode o treinamento do modelo:
   - `python util_doc2vec_facil.py -pasta ./meu_modelo -treinar`
 > 💡 Nota: serão treinados apenas os termos encontrados nos arquivos `VOCAB_BASE*.txt` ou no caso de um termo não ser encontrado, o `stemmer` e `sufixo` dele serão treinados se estiverem no vocab.
  > - Após a primeira época será criado o arquivo `vocab_treino.txt` que contém os termos realmente treinados (os disponíveis no vocab e que foram encontrados nos textos de treino)

## 3) Quero criar ngramas ou limpar o texto com termos que não devem ser treinados:
 - Além dos arquivos do cenário `2`, acrescente o arquivo `VOCAB_TRADUTOR_COMPOSTOS_PTBR.txt` e crie outros arquivos se desejar com seus ngramas (veja [NGramasFacil](readme_ngramas.md) ) ou termos compostos para remoção. Crie também um ou mais arquivos `VOCAB_REMOVIDO*.txt` com suas listas de exclusões.
 - Rode o treinamento do modelo:
   - `python util_doc2vec_facil.py -pasta ./meu_modelo -treinar`
 > 💡 Nota: os termos compostos são agrupados após a tokenização e limpeza e serão incluídos automaticamente no vocab de treino.  
  > - Após a primeira época será criado o arquivo `vocab_treino.txt` que contém os termos realmente treinados (os disponíveis no vocab e que foram encontrados nos textos de treino)

## 4) Quero criar meu vocab do zero, fazer curadoria e depois treinar:
 - Crie a pasta `meu_modelo`
 - Crie uma subpasta `meu_modelo/textos_vocab`
   - Coloque nesta pasta documentos relevantes para a criação do vocab
   - Rode: `python util_doc2vec_vocab_facil.py -pasta meu_modelo`

#### 4.1 realize o ciclo de curadoria :repeat::
 - Abra o arquivo `curadoria_planilha_vocab.txt` e avalie os termos que deseja treinar.
   - Crie um arquivo com os termos completos ex. `VOCAB_BASE meus termos.txt`, coloque nesse arquivo os termos completos que deseja treinar.
   - Crie um arquivo com os termos fragmentados, se desejar, ex. `VOCAB_BASE meus fragmentos.txt`, coloque nesse arquivo os termos da coluna QUEBRADOS que deseja treinar.
 - Crie uma subpasta `meu_modelo/textos_treino` e coloque alguns ou todos os arquivos de treino
   - Rode: `python util_doc2vec_vocab_facil.py -pasta meu_modelo -reiniciar`
 - Abra o arquivo `curadoria_planilha_vocab.txt` e avalie os termos que não foram localizados (colunas VOCAB ou VOCAB_QUEBRADOS iguais a N), atualize os seus arquivos com esses termos.
 - Siga esse ciclo até que o seu vocab esteja como deseja e inicie o treinamento :repeat:.
 - Inicie o treinamento:
    - `python util_doc2vec_facil.py -pasta ./meu_modelo -treinar`

## 5) Quero criar um vocab extra além dos termos dos arquivos de exemplo:
 - Crie a pasta `meu_modelo`
 - Crie uma subpasta `meu_modelo/textos_treino`
 - Crie uma subpasta `meu_modelo/doc2vecfacil` e coloque os arquivos `VOCAB_BASE_PTBR.txt` e, se quiser que ocorra a fragmentação de termos fora do vocab, coloque também o arquivo `VOCAB_BASE_PTBR_FRAGMENTOS.txt`. Ou use seus arquivos com ou sem esses dois sugeridos. 
 - Crie uma subpasta `meu_modelo/textos_vocab`
   - Coloque nesta pasta documentos relevantes para a criação do vocab extra
   - Rode: `python util_doc2vec_vocab_facil.py -pasta meu_modelo`
 - Siga os passos 4.1 até ter o vocab desejado
 - Inicie o treinamento:
    - `python util_doc2vec_facil.py -pasta ./meu_modelo -treinar`

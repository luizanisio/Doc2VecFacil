# Dicas simplificadas de como preparar e treinar um modelo em diversos cenários
Logo abaixo estão descritos alguns cenários de criação de vocab e treinamento e os seus passos.<br>
As dicas vão levar em conta que o seu modelo será criado na pasta `meu_modelo`, mas pode criar a pasta com o nome que quiser, basta passar esse nome no parêmtro.
  
#### Estrutura de pastas:  
:file_folder: `Pasta raiz` (informada no parâmetro da chamada - padrão = "meu_modelo")<br>
&nbsp;&nbsp;\_:file_folder: `doc2vecfacil` (pasta do modelo e dos vocabs): ao disponibilizar o modelo para uso, pode-se renomear essa pasta livremente<br>
&nbsp;&nbsp;\_:file_folder: `textos_vocab`: textos que serão usados para criar a planilha de curadoria<br>
&nbsp;&nbsp;\_:file_folder: `textos_treino`: textos que serão usados na fase de treinamento.<br>

> 💡 Nota: é difícil definir um número de épocas para o treinamento, pode ter 1000 ou 5000. Esse número depende de vários fatores, entre eles o número de termos e documentos treinados.
        
#### Estrutura de arquivos:
 Os arquivos necessários para o treino que serão usados para a tokenização são:<br>
 :file_folder: `meu_modelo` <br>
 &nbsp;&nbsp;\_:file_folder: `doc2vecfacil`<br>
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - `VOCAB_BASE_*.txt`: arquivos com termos que serão treinados <br>
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - `VOCAB_REMOVIDO*.txt`: arquivos com termos que serão ignorados (opcional)<br>
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - `VOCAB_TRADUTOR*.txt`: arquivos com termos ou frases que serão removidas ou transformadas (opcional)<br>

> 💡 Nota: baixe os [`exemplos`](./exemplos) de configurações do tokenizador. Analise, ajuste os termos, termos compostos e remoções, gere a planilha de curadoria e adapte ao seu contexto. 
> 📑 O exemplo `modelo_legislacoes` já possui alguns textos para o vocab e diversos termos, fragmentos e ngramas configurados, bastando apenas incluir seus documentos para gerar uma planilha de curadoria ou iniciar o treinamento de uma primeira versão do seu modelo. Os textos foram baixados de links públicos, os links estão na pasta de exemplo.

#### Durante o treino:
 Arquivos para acompanhar durante o treinamento:<br>
 :file_folder: `meu_modelo` <br>
 &nbsp;&nbsp;\_:file_folder: `doc2vecfacil`<br>
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - `doc2vec.log`: algumas informações sobre a última época treinada e dados do modelo.<br>
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - `comparar_termos.log`: a cada época o arquivo `comparar_termos.log` será atualizado com os termos mais similares dos termos indicados para acompanhamento, bem como a similaridade entre frases indicadas para acompanhamento.<br>
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - `termos_comparacao_treino.txt`: termos/frases desse arquivo `termos_comparacao_treino.txt` serão usadas para a geração do arquivo de acompanhamento de treinamento. Caso esse arquivo não exista, será criado com alguns termos do vocab treinado e uma frase de exemplo. Altere esse arquivo sempre que quiser.<br>
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - `vocab_treino.txt`: será criado após a primeira época e contém os termos realmente treinados (os disponíveis no vocab e que foram encontrados nos textos de treinamento após a tokenização)

## 1) Quero treinar sem preparar um vocab:
 - Crie a pasta `meu_modelo`
 - Crie uma subpasta `meu_modelo/textos_treino`
 - Rode o treinamento do modelo:
   - `python util_doc2vec_facil.py -pasta ./meu_modelo -treinar`
  > 💡 Nota: todos os tokens serão treinados, será feita apenas a limpeza simples dos textos para comparações simples já é o suficiente
  > - você pode rodar `python util_doc2vec_vocab_facil.py -pasta meu_modelo` só para ter uma ideia dos termos e suas relevâncias antes do treinamento.

## 2) Quero usar as palavras sugeridas ou já tenho as minhas:
 - Crie a pasta `meu_modelo`
 - Crie uma subpasta `meu_modelo/textos_treino`
 - Crie uma subpasta `meu_modelo/doc2vecfacil` e coloque os arquivos `VOCAB_BASE_PTBR.txt` e, se quiser que ocorra a fragmentação de termos fora do vocab, coloque também o arquivo `VOCAB_BASE_PTBR_FRAGMENTOS.txt`. Ou use seus arquivos com ou sem esses dois sugeridos. 
 - Rode o treinamento do modelo:
   - `python util_doc2vec_facil.py -pasta ./meu_modelo -treinar`
 > 💡 Nota: serão treinados apenas os termos encontrados nos arquivos `VOCAB_BASE*.txt` ou no caso de um termo não ser encontrado, o `stemmer` e `sufixo` dele serão treinados se estiverem no vocab.

## 3) Quero criar ngramas ou limpar o texto com termos que não devem ser treinados:
 - Além dos arquivos do cenário `2`, acrescente o arquivo `VOCAB_TRADUTOR_COMPOSTOS_PTBR.txt` e crie outros arquivos se desejar com seus ngramas (veja [NGramasFacil](readme_ngramas.md) ) ou termos compostos para remoção. Crie também um ou mais arquivos `VOCAB_REMOVIDO*.txt` com suas listas de exclusões.
 - Rode o treinamento do modelo:
   - `python util_doc2vec_facil.py -pasta ./meu_modelo -treinar`
 > 💡 Nota: os termos compostos são agrupados após a tokenização e limpeza e serão incluídos automaticamente no vocab de treino.  

## 4) Quero criar meu vocab do zero, fazer curadoria e depois treinar:
 - Crie a pasta `meu_modelo`
 - Crie uma subpasta `meu_modelo/textos_vocab`
   - Coloque nesta pasta documentos relevantes para a criação do vocab
   - Rode: `python util_doc2vec_vocab_facil.py -pasta meu_modelo`

 > 💡 Nota: Será criado o arquivo `curadoria_planilha_vocab.xlsx` com todos os termos encontrados nos textos da pasta `textos_vocab`, suas frequências, tfidf, tamanho, dentre outros atributos para permitir uma análise e curadoria dos termos. Esse arquivo pode ser aberto no Excel para facilitar a análise/curadoria do vocabulário que será treinado.
 > - opcionalmente pode-se usar o parâmetro `-treino` para que a planilha de curadoria seja criada com os textos da pasta `textos_treino`.

#### 4.1 realize o ciclo de curadoria :repeat::
 - Abra o arquivo `curadoria_planilha_vocab.xlsx` e avalie os termos que deseja treinar.
   - Crie um arquivo com os termos completos ex. `VOCAB_BASE meus termos.txt`, coloque nesse arquivo os termos completos que deseja treinar.
   - Crie um arquivo com os termos fragmentados, se desejar, ex. `VOCAB_BASE meus fragmentos.txt`, coloque nesse arquivo os termos da coluna QUEBRADOS que deseja treinar.
 - Crie uma subpasta `meu_modelo/textos_treino` e coloque alguns ou todos os arquivos de treino
   - ❗ lembre de fechar o arquivo de curadoria antes de rodar novamente o ciclo
   - Rode: `python util_doc2vec_vocab_facil.py -pasta meu_modelo -reiniciar`
 - Abra o arquivo `curadoria_planilha_vocab.xlsx` e avalie os termos que não foram localizados (colunas VOCAB ou VOCAB_QUEBRADOS iguais a N), atualize os seus arquivos com esses termos.
 - Siga esse ciclo até que o seu vocab esteja como deseja e inicie o treinamento :repeat:.
 - Inicie o treinamento:
    - `python util_doc2vec_facil.py -pasta ./meu_modelo -treinar`

## 5) Quero criar um vocab extra com ajuda da curadoria:
 - Crie a pasta `meu_modelo`
 - Crie uma subpasta `meu_modelo/textos_treino`
 - Crie uma subpasta `meu_modelo/doc2vecfacil` e coloque os arquivos `VOCAB_BASE_PTBR.txt` e, se quiser que ocorra a fragmentação de termos fora do vocab, coloque também o arquivo `VOCAB_BASE_PTBR_FRAGMENTOS.txt`. Ou use seus arquivos com ou sem esses dois sugeridos, ou quantos arquivos desejar para compor o `VOCAB_BASE. 
 - Crie uma subpasta `meu_modelo/textos_vocab`
   - Coloque nesta pasta documentos relevantes para a criação do vocab extra
   - Rode: `python util_doc2vec_vocab_facil.py -pasta meu_modelo`
 - Siga o passo `4.1` até ter o vocab desejado
 - Inicie o treinamento:
    - `python util_doc2vec_facil.py -pasta ./meu_modelo -treinar`

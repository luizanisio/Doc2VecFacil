# Doc2VecFacil

Componente python que simplifica o processo de criação de um modelo `Doc2Vec` (gensim) com facilitadores para geração de um vocab personalizado e com a geração de arquivos de curadoria.
- se você não sabe o que é um modelo de similaridade, em resumo é um algoritmo não supervisionado para transformar frases ou documentos em vetores matemáticos que podem ser comparados retornando um valor que representa a similaridade semântica entre dois ou mais documentos. Nesse contexto a máquina 'aprende' o vocabulário treinado e o contexto em que as palavras aparecem, permitindo identificar a similaridade entre os termos, as frases e os documentos.
- Com essa comparação vetorial, é possível encontrar documentos semelhantes a um indicado, agrupar documentos semelhantes de uma lista de documentos e monitorar documentos que entram na base ao compará-los com os documentos marcados como importantes. 

### O componente `Doc2VecFacil` trabalha em duas etapas:
 - criação de um vocab personalizado ao processar textos considerados importantes para o modelo que será treinado
   - `python util_doc2vec_vocab_facil.py -pasta ./meu_modelo`
 - treinamento do modelo usando a estrutura de tokenização criada a partir do vocab personalizado
   - `python util_doc2vec_facil.py -pasta ./meu_modelo` -treinar

- <b>Códigos</b>: 
  - [`Criação de vocab`](util_doc2vec_vocab_facil.py)
  - [`UtilDoc2VecFacil`](util_doc2vec_facil.py) e [`UtilDoc2VecFacil_Treinamento`](util_doc2vec_facil.py) 
  - [`TradutorTermos`](util_tradutor_termos.py)

Logo abaixo estão algumas dicas de como criar um modelo personalizado com esse código, como ele funciona e como usar o seu modelo para pesquisas de documentos semelhantes semanticamente ou textualmente, como realizar agrupamento de documentos por similaridade para auxiliar na organização de documentos usando o ElasticSearch e a pesquisa vetorial.

`EM BREVE`: Será disponibilizado um serviço exemplo em conjunto com o componente [PesquisaElasticFacil](https://github.com/luizanisio/PesquisaElasticFacil) para criação de modelos de similaridade textual, agregando valor às pesquisas do ElasticSearch de forma simples com um modelo treinado no corpus específico de cada projeto.

### Criação do vocab personalizado

O arquivo `util_doc2vec_vocab_facil.py` é complementar à classe `Doc2VecFacil` e serve para facilitar a criação de arquivos que configuram o `TokenizadorInteligente`. A ideia é trabalhar com termos importantes para o modelo, adicionados a termos complementares compostos por fragmentos de termos `stemmer` + `sufixo`. Com isso novos documentos que possuam termos fora do vocab principal podem ter o stemmer e o sufixo dentro do vocab do modelo, criando um vocab mais flexível e menor. É possível também transformar termos ou conjunto de termos durante o processamento, como criar n-gramas, reduzir nomes de organizações em siglas etc.

### São criados dois arquivos de vocabulários, um principal e um complementar
 - o dicionário principal é composto pelos termos completos encontrados nos textos da pasta `textos_vocab`
   - coloque aqui textos que contenham boas palavras, limpas de preferência. Podem ser listas retiradas de algum documento, não importa o contexto delas, apenas as palavras nesse momento. Então listas de palavras e documentos como gramáticas de dicionários de português digitais parecem uma boa opção. Coloque também documentos com palavras relacionadas ao corpus desejado (psicologia, medicina, legislação, administração, etc).
 - o dicionário complementar é composto pelos termos quebrados encontrados nos textos da pasta `textos_vocab_complementar`. 
   - coloque aqui listas de palavras não tão importantes para o contexto do corpus, mas que são importantes na composição desses textos. Podem ser textos maiores, a ideia aqui é construir fragmentos de palavras menos importantes que as principais, no formato (stemmer + sufixos). Aos termos não encontrados no dicionário criado pelos textos principais será aplicado o stemmer e será incluído o sufixo do termo complementando o stemmer, criando um conjunto extra de termos que possibilitam uma flexibilidade de novas combinações não conhecidas durante o treino.
 - Essa combinação de termos completos e fragmentos (stemmer + sufixo) possibilita criar palavras por combinação ao submeter um documento novo ao modelo.
   
 - <b>Exemplo</b>: `engloba` pode ser composta por `englob` `#a`, e outras formações podem ocorrer como `englob` `#ada`, `englob` `#adamente`, `englob` `#adas` caso esses  fragmentos estejam disponíveis no vocabulário de treinamento.
   - O vocab de treinamento não precisa do `#` antes do sufixo, apenas dos fragmentos. Mas durante o treinamento os fragmentos usados como
     sufixo iniciarão com `#` para facilitar sua identificação e diferenciar dos termos principais no modelo final.
 - <b>Exemplo de tokenização</b>: 
   ```python
   # para carregar o modelo, indique a pasta criada no treinamento
   dv = UtilDoc2VecFacil(pasta_modelo='./meu_modelo_treinado')
   print(dv.tokens_sentenca('ATENDIAM A TESTEMUNHA SEU DEPOIMENTO APESAR DE TRAZER ALGUMAS IMPRECISÕES SOBRE OS FATOS ATENDO-SE OS JURADOS ÀS PROVAS PRODUZIDAS EM PLENÁRIOS'))
   ```
   ```
   ['atendiam_testemunha', 'seu', 'depoimento', 'apesar', 'de', 'trazer', 'algumas', 'impreciso', '#es', 'sobre', 'os', 'fatos', 'atend', '#o', 'se', 'os', 'jurados', 'as', 'provas', 'produzidas', 'em', 'plenari', '#os']
   ```

Junto com a criação dos dicionários é criado um arquivo `curadoria_vocab.txt` com cada termo, sua frequência, tfidf, tamanho, dentre outros atributos para permitir uma análise e curadoria dos termos. Esse arquivo pode ser aberto no Excel para facilitar a análise/curadoria do vocabulário que será treinado.
- Opcionalmente pode-se editar o arquivo do dicionário principal com base nos dados dessa planilha e remover o dicionário complementar e rodar o código novamente para que o dicionário complementar seja recriado aproveitando o dicionário base ajustado manualmente, sem um novo processamento do dicionário principal.
- Opcionamente pode-se criar quantos dicionários quiser com o prefixo `vocab_base*.txt`, os termos desses dicionários não farão parte do dicionário principal criado, nem do complementar, mas serão carregados para treinamento do modelo. 
- Opcionamente também é possível criar arquivos de termos excluídos do treinamento `vocab_removido*.txt`
- Opcionamente também é possível criar arquivos de frases/termos para tradução/transformação como substituir `processo penal` por `processo_penal`, compondo termos compostos para serem tratados como termos únicos no modelo. Pode-se também remover termos ao não indicar a transformação. Podem ser criados arquivos no formato `vocab_tradutor*.txt`. E a configuração dos termos é por linha `processo penal => processo_penal`, ou `remover esse trecho =>`.
  - Um exemplo de uso é incluir nomes de pessoas ou empresas que não há interesse em compor o treinamento ou reduzir termos compostos em termos únicos para a vetorização posterior de novos documentos. Removendo ruídos conhecidos e reduzindo o vocabulário principal. Importante ressaltar que quanto maior o número de termos, maior o tempo de processamento, mesmo usando recursos otimizados para essa transformação veja a classe `TradutorTermos` no arquivo `util_tradutor_termos`.

### Exemplo de arquivo `curadoria_vocab.txt` de curadoria de termos:
| TERMO       | TFIDF       | TAMANHO |  QTD  | QTD_DOCS | VOCAB | VOCAB_QUEBRADOS |
|-------------|-------------|:-------:|:-----:|:--------:|:-----:|:---------------:|
| acessorias  | 0,301051057 |    10   |   91  |    28    |   S   |        N        |
| calculo     | 0,490002279 |    7    |  1736 |    810   |   S   |        N        |
| cinco       | 0,471974082 |    5    | 68661 |   6846   |   S   |        N        |
| custas      | 0,41286071  |    6    |  740  |    417   |   S   |        N        |

## Definição de pastas:
 A estrutura de pastas é pré-definida para facilitar o uso dos componentes. <br>
 O único parâmetro informado é a pasta raiz que vai conter as outras pastas. <br>
 - `\Pasta raiz` (informada no parâmetro da chamada - padrão = "meu_modelo")
   - `\doc2vecfacil` (pasta do modelo): ao disponibilizar o modelo para uso, pode-se renomear essa pasta livremente
   - `\textos_vocab`: textos que serão tokenizados para criação do vocab principal
   - `\textos_complementares`: textos que seráo tokenizados para criação do dicionário complementar de fragmentos dos termos não encontrados no vocab principal.

 Ao final são gerados os arquivos de dicionários que podem ser alterados manualmente antes do treinamento do modelo, desde que mantidos os nomes dos arquivos que são padronizados e definirão a tokenização para treinamento e a tokenização durante o uso do modelo final.
 
 - `\Pasta raiz`
   - `doc2vecfacil/VOCAB_BASE*.txt` são arquivos carregados em conjunto para formar o vocab de tokenização
   - `doc2vecfacil/VOCAB_REMOVIDO*.txt` são arquivos carregados em conjunto para excluir termos do vocab de tokenização
   - `doc2vecfacil/VOCAB_TRADUTOR*.txt` são arquivos com tradutores no estilo `termo1 => termo2`, podendo conter termos compostos nas duas pontas ou não conter `=>` que indica que o termo ou os termos da esquerda serão removidos. A avaliação é feita após a limpeza, lowercase e remoção de acentos. Um arquivo `vocab_tradutor_termos.log` é criado indicando como as transformações serão realizadas.
 - São gerados alguns arquivos para auxílio na curadoria que podem ser apagados depois. Pode-se abrir os arquivos `curadoria_planilha_vocab.txt` e `curadoria_planilha_vocab (OOV).txt` no Excel para análise mais apurada dos termos (use os filtros do excel para analisar, remover ou incluir termos novos).
    'curadoria_planilha_vocab.txt` - planilha de curadoria dos termos inteiros ou fragmentados onde pelo menos o stemmer estava no dicionário, com vários atributos como: tamanho, tfidf, frequência, etc. Poderá ser aberto no excel para análise e refinamento do vocabulário final de treinamento
   - `curadoria_planilha_vocab (OOV).txt` - planilha de curadoria dos termos que não foram encontrados nos dicionários mesmo após fragmentados
   - `curadoria termos inteiros OOV.txt` - são os termos inteiros que não foram encontrados no dicionário principal (antes de serem fragmentados)
   - `curadoria termos OOV TOKENS QUEBRADOS.txt` - são os fragmentos dos termos que não foram encontrados nos dicionários 
   - `curadoria termos stemmer OOV TOKENS INTEIROS.txt` - são os termos inteiros que mesmo após fragmentados não foram encontrados nos dicionários 
   - `curadoria termos VOCAB TOKENS QUEBRADOS.txt` - são os termos que não foram encontrados inteiros no vocab mas seus fragmentos forma encontrados
   - `curadoria termos VOCAB FINAL PRÉ-TREINO.txt` - são os termos e fragmentos que vão compor o vocab de treinamento. 

 - Pode-se gerar um vocab e utilizá-lo para treinar diversos conjuntos diferentes de arquivos em modelos diferentes. 
 - Por isso o arquivo `vocab_treino`, gerado durante o treinamento, pode ser menor que o arquivo `curadoria termos VOCAB FINAL PRÉ-TREINO`, já que vai conter apenas os termos encontrados durante o treinamento. Os arquivos criados são sugestões e podem ser alterados livremente antes de iniciar o treinamento do modelo.
 - Por esse motivo, ao iniciar o treinamento o cache de pré-processamento é reconstruído para evitar que ajustes manuais não estejam contemplados no treinamento.

## Passo a passo para criar o vocab de treino: 
 1) Criar as pastas:
    - `meu_modelo`
    - `meu_modelo\textos_vocab`: colocar um conjunto de textos importantes para o corpus
    - `meu_modelo\textos_vocab_complementar`: colocar um conjunto de textos complementares (tokens serão quebrados)
    - `meu_modelo\textos_treino`: colocar os arquivos que serão usados no treinamento
 2) Rodar: `python util_doc2vec_vocab_facil.py -pasta./meu_modelo`
    - para forcar recriar os arquivos se já existirem, basta colocar o parâmetro `-reiniciar`
    - ao chamar uma segunda vez, o código vai apenas atualizar o arquivo de curadoria
    - o arquivo de curadoria será criado considerando os textos da pasta `textos_treino` também
 3) Opcional: abrir o arquivo ./meu_modelo/doc2vecfacil/curadoria_vocab.txt no excel e analisar os termos
    - alterar os arquivos `VOCAB_BASE*` e `VOCAB_REMOVIDO*` com base na curadoria
    - alterar o arquivo `termos_comparacao_treino.txt` com termos importantes para acompanhar a evolução do modelo
 4) Opcional: arquivos de exclusão e de transformação de termos
    - arquivos no formato `VOCAB_TRADUTOR*.txt` com transformações, ngramas etc, e um ou mais arquivos .
    - arquivos no roamto `VOCAB_REMOVIDO*.txt' com termos que serão excluídos do vocab final (a diferença entre o arquivo de transformação é que trabalha com termos únicos do vocab).

## Criando o vocab manualmente (opcional):
 Os arquivos necessários para o treino que serão usados para a tokenização são:
    - `meu_modelo\doc2vecfacil\VOCAB_BASE_*.txt`: arquivos com termos que serão treinados 
 Opcionais:
    - `meu_modelo\doc2vecfacil\VOCAB_REMOVIDO*.txt`: arquivos com termos que serão ignorados
    - `meu_modelo\doc2vecfacil\VOCAB_TRADUTOR*.txt`: arquivos com termos ou frases que serão removidas ou transformadas
 - Pode-se criar os arquivos manualmente com os termos desejados, ou aproveitar os arquivos de outro treino. Ou Ajustar os arquivos criados automaticamente incluindo ou retirando termos.
 - Pode-se criar arquivos de transformação automaticamente usando o código `util_tradutor_termos.py-pasta = meu_modelo´. Ele vai carregar os arquivos e utilizar o `Phrases` do gensim para sugerir bigramas, trigramas e quadrigramas que poderam ser analisados e incorporados a um arquivo de transformação como `meu_modelo\doc2vecfacil\VOCAB_TRADUTOR_NGRAMAS.txt` por exemplo. Caso queira saber mais sobre a criação de ngramas usando esse componente, veja aqui: [nGramasFacil](ngramas.md)

## Conferindo o processamento dos textos
- Pode-se conferir os arquivos `.clr` criados nas pastas `textos*` pois eles são o resultado do processamento dos textos originais com o `TokenizadorInteligente`.
- Nesse arquivo é possível identificar os fragmentos e os tokens principais e verificar se a tokenização está de acordo com o esperado. O treinamento do modelo será feito com esse arquivo. 
- No início do treinamento os arquivos `.clr` serão atualizados para garantir que novos termos incluídos ou alterados manualmente sejam refletidos na tokenização.
- Os arquivos `.clr` são necessários durante todo o treinamento e serão recriados se não forem encontrados, isso acelera o treinamento para não haver necessidade de reprocessar o texto cada vez que o treinamento passar por ele.

## Passo a passo para treinar o modelo doc2vec: 
 Com os arquivos de vocab prontos, criados automaticamente ou manualmente, pode-se treinar o modelo.
 1) Conferir a pasta de texto e arquivos do vocab (o case dos nomes não importa, caixa alta é para facilitar a identificação):
    - `meu_modelo\textos_treino\`: colocar os arquivos que serão usados no treinamento
    - `meu_modelo\doc2vecfacil\VOCAB_BASE*.txt`: arquivos com termos que serão treinados 
    - `meu_modelo\doc2vecfacil\VOCAB_REMOVIDO*.txt`: arquivos com termos que serão ignorados
    - `meu_modelo\doc2vecfacil\VOCAB_TRADUTOR*.txt`: arquivos com termos que serão ignorados
 2) Rodar: `python util_doc2vec_facil.py -pasta ./meu_modelo`.
    - se já existir o modelo, o treinamento será continuado.
    - sugere-se aguardar no mínimo 1000 épocas, se possível umas 5000
    - pode-se acompanhar a evolução do modelo criando ou alterando o arquivo `termos_comparacao_treino.txt` que contém uma lista de termos para geração do arquivo `termos_comparacao.log` onde para cada termo será apresentada uma lista de termos mais semelhantes, permitindo uma avaliação do modelo em treinamento. Esse arquivo não interfere no treino e pode ser modificado a qualquer momento.

## Parâmetros
 - `python util_doc2vec_facil.py`
    - `-pasta` - nome da pasta de treinamento que contém a pasta do modelo e as pastas de textos, o padrão é `meu_modelo` se não for informada.
    - `-treinar`' - iniciar o treinamento do modelo
    - `-reiniciar sim` - remove o modeo atual, se existir, e inicia um novo treinamento
    - `-testar` - carrega o modelo atual, se existir, e atualiza o arquivo `comparar_termos.log` com os termos encontrados no arquivo `termos_comparacao_treino.txt`
    - `-epocas` - define o número de épocas que serão treinadas, o padrão é 5000 e pode ser interrompido ou acrescido a qualquer momento.
    - `-dimensoes` - define o número de dimensões dos vetores de treinamento.
    - `-workers` - número de threads de treinamento, padrão 100

 - `python util_doc2vec_vocab_facil.py`
    - `-pasta` - nome da pasta de treinamento que contém as pastas de textos, o padrão é `meu_modelo` se não for informada.
    - `-reiniciar` - remove os arquivos de vocab automáticos, se existirem, e reinicia a criação deles.
    - `-teste` - carrega o `TokenizadorInteligente` para verificar se os arquivos que serão usados para o processamento no treino estão ok.

## Dicas:
 Ao rodar o código para criar o vocab:
 - caso exista um ou mais arquivos do dicionário principal e não existam arquivos do secundário, será criado apenas o dicionário secundário com base nos textos complementares das pastas `textos_vocab_complementar`. Isso facilita criar novos modelos mantendo um dicionário base padrão.
 - caso existam todos os dicionários, será criado/atualizado o arquivo de curadoria, sem modificar os dicionários existentes. Usando o parâmetros `-reiniciar`, o vocab automático e automático complementar serão recriados.
 - se durante a preparação do dicionário existirem documetnos na pasta `textos_treino`, o arquivo de curadoria será criado usando os textos de treinamento, facilitando identificar os termos que não serão treinados, bem como outros atributos de todos os termos. Como o processo pode demorar, para um corpus muito grande pode-se deixar apenas alguns textos na pasta treino para a criação do vocab e colocar todos no momento de treinamento propriamente dito.
 - no arquivo de curadoria, a coluna `VOCAB` S/N indica se o termo está contido inteiro no vocab e a coluna `VOCAB_QUEBRADOS` S/N indica se o termo foi incluído após ser fragmentado. Caso as duas colunas sejam N, isso indica que o termo não será treinado, nem inteiro e nem o seu formato `stemmer`+`#sufixo`.

### Termos comparados para acompanhar a evolução do modelo:
Exemplo de saída do arquivo `comparar_termos.log` atualizado a cada época.
Esse log é gerado com os termos disponíveis no arquivo `termos_comparacao_treino.txt` que é carregado no início do treino e pode ser alterado sempre que desejado.
```
artigo               |  art (77%)                |  artigos (55%)            |  arts (53%)              
cobrados             |  pagos (54%)             
cogitar              |  falar (52%)             
compromisso          |  promessa (57%)          
comprovadas          |  demonstradas (58%)      
entendimento         |  posicionamento (52%)    
entorpecentes        |  drogas (64%)            
julga                |  julgou (71%)             |  julgando (53%)          
parcelas             |  prestacoes (60%)         |  despesas (55%)           |  quantias (52%)          
termo                |  peticao (64%)            |  inepcia (63%)           
```

## Usando o modelo:
 O modelo pode ser carregado facilmente:
 ```python 
 from util_doc2vec_facil import UtilDoc2VecFacil
 dv = UtilDoc2VecFacil(pasta_modelo=PASTA_MODELO)

frase1 = 'EXECUÇÃO POR TÍTULO EXTRAJUDICIAL DE HONORÁRIO ADVOCATÍCIO EMBARGOS ADJUDICAÇÃO PENHORAS'
frase2 = 'EMENTA SEGUROs de VIDA COBRANÇA CUMULADA C PRETENSÃO INDENIZATÓRIA PRESCRIÇÃO RECONHECIDA'
frase3 = 'DE HONORÁRIOS ADVOCATÍCIOS EMBARGOS ADJUDICAÇÃO PENHORA EXECUÇÃO POR TÍTULO EXTRAJUDICIAL '
print('Frase1: ', frase1, '\nFrase2: ', frase2, '\n\t - Similaridade: ', dv.similaridade(frase1,frase2))
print('Frase1: ', frase1, '\nFrase3: ', frase3, '\n\t - Similaridade: ', dv.similaridade(frase1,frase3))        
print('\nTokens frase 1: ', dv.tokens_sentenca(frase1))
 ```
 
 <b>Resultado:</b>
 ```
Frase1:  EXECUÇÃO POR TÍTULO EXTRAJUDICIAL DE HONORÁRIO ADVOCATÍCIO EMBARGOS ADJUDICAÇÃO PENHORAS
Frase2:  EMENTA SEGUROs de VIDA COBRANÇA CUMULADA C PRETENSÃO INDENIZATÓRIA PRESCRIÇÃO RECONHECIDA
         - Similaridade:  0.43190062046051025
Frase1:  EXECUÇÃO POR TÍTULO EXTRAJUDICIAL DE HONORÁRIO ADVOCATÍCIO EMBARGOS ADJUDICAÇÃO PENHORAS
Frase3:  DE HONORÁRIOS ADVOCATÍCIOS EMBARGOS ADJUDICAÇÃO PENHORA EXECUÇÃO POR TÍTULO EXTRAJUDICIAL
         - Similaridade:  0.46588313579559326

Tokens frase 1:  ['execucao', 'por', 'titulo', 'extrajudicial', 'de', 'honorario', 'advocaticio', 'embargos', 'adjudicacao', 'penhoras'] 
 ```

## Dicas de uso:
- gravar os vetores, textos e metadados dos documentos no ElasticSearch
- fazer pesquisas com More Like This, vetoriais e por proximidade de termos como disponibilizado no componente [PesquisaElasticFacil](https://github.com/luizanisio/PesquisaElasticFacil) em breve.


# Doc2VecFacil

Componente python que simplifica o processo de criação de um modelo `Doc2Vec` (gensim) com facilitadores para geração de um vocab personalizado e com a geração de arquivos de curadoria.

O componente trabalha em duas etapas:
 - criação de um vocab personalizado ao processar textos considerados importantes para o modelo que será treinado
   - `python util_doc2vec_vocab_facil.py -pasta ./meu_modelo`
 - treinamento do modelo usando a estrutura de tokenização criada a partir do vocab personalizado
   - `python util_doc2vec_facil.py -pasta ./meu_modelo` -treinar

Logo abaixo estão algumas dicas de como criar um modelo personalizado com esse código, como ele funciona e como usar o seu modelo para pesquisas de documentos semelhantes semanticamente ou textualmente, como realizar agrupamento de documentos por similaridade para auxiliar na organização de documentos usando o ElasticSearch e a pesquisa vetorial.

`EM BREVE`: Será disponibilizado um serviço exemplo em conjunto com o componente [PesquisaElasticFacil](https://github.com/luizanisio/PesquisaElasticFacil) para criação de modelos de similaridade textual, agregando valor às pesquisas do ElasticSearch de forma simples com um modelo treinado no corpus específico de cada projeto.

`FINALIZANDO TESTES antes de disponibilizar`

## Criação do vocab personalizado

O arquivo `util_doc2vec_vocab_facil.py` é complementar à classe `Doc2VecFacil` e serve para facilitar a criação de arquivos que configuram o `TokenizadorInteligente`. A ideia é trabalhar com termos importantes para o modelo, adicionados a termos complementares compostos por fragmentos de termos `stemmer` + `sufixo`. Com isso novos documentos que possuam termos fora do vocab principal podem ter o stemmer e o sufixo dentro do vocab do modelo, criando um vocab mais flexível e menor.

São criados dois arquivos de vocabulários, um principal e um complementar
 - o dicionário principal é composto pelos termos completos encontrados nos textos da pasta `textos_vocab`
 - o dicionário complementar é composto pelos termos quebrados encontrados nos textos da pasta `textos_vocab_complementar'`
 - aos termos não encontrados no dicionário será aplicado o stemmer e será incluído o sufixo do termo complementando o stemmer, criando um conjunto extra de termos que possibilitam uma flexibilidade de novas combinações não conhecidas durante o treino.
   - Isso possibilita criar palavras por combinação `prefixo` `#sufixo` ao submeter um documento novo ao modelo.
   
   <b>Exemplo</b>: `engloba` pode ser composta por `englob` `#a`, e outras formações podem ocorrer como `englob` `#ada`, `englob` `#adamente`, `englob` `#adas` caso esses  fragmentos estejam disponíveis no vocabulário de treinamento.
   - O vocab de treinamento não precisa do `#` antes do sufixo, apenas dos fragmentos. Mas durante o treinamento os fragmentos usados como
     sufixo iniciarão com `#` para facilitar sua identificação e diferenciar dos termos principais.

Junto com a criação dos dicionários é criado um arquivo com cada termo, sua frequência, tfidf, tamanho, dentre outros atributos para permitir uma análise e curadoria dos termos. Esse arquivo pode ser aberto no Excel para facilitar a análise/curadoria do vocabulário que será treinado.
- Opcionalmente pode-se editar o arquivo do dicionário principal com base nos dados dessa planilha e remover o dicionário complementar e rodar o código novamente para que o dicionário complementar seja recriado aproveitando o dicionário base ajustado manualmente, sem um novo processamento do dicionário principal.
- Opcionamente pode-se criar quantos dicionários quiser com o prefixo `vocab_base*.txt`, os termos desses dicionários não farão parte do dicionário principal criado, nem do complementar, mas serão carregados para treinamento do modelo. 
- Opcionamente também é possível criar arquivos de termos excluídos do treinamento `vocab_removido*.txt`

## Definição de pastas:
 A estrutura de pastas é pré-definida para facilitar o uso dos componentes. <br>
 O único parâmetro informado é a pasta raiz que vai conter as outras pastas. <br>
 - `Pasta raiz` (informada no parâmetro da chamada)
   - `doc2vecfacil` (pasta do modelo): ao disponibilizar o modelo para uso, pode-se renomear essa pasta livremente
   - `textos_vocab`: textos que serão tokenizados para criação do vocab principal
   - `textos_complementares`: textos que seráo tokenizados para criação do dicionário complementar de fragmentos dos termos não encontrados no vocab principal.

 Ao final são gerados os arquivos de dicionários que podem ser alterados manualmente antes do treinamento do modelo, desde que mantidos os nomes dos arquivos que são padronizados e definirão a tokenização para treinamento e a tokenização durante o uso do modelo final.
 
 - `Pasta raiz`
   - `doc2vecfacil/VOCAB_BASE*.txt` são arquivos carregados em conjunto para formar o vocab de tokenização
   - `doc2vecfacil/VOCAB_REMOVIDO*.txt` são arquivos carregados em conjunto para excluir termos do vocab de tokenização
   - `doc2vecfacil/vocab_final.txt` arquivo informativo sobre os tokens aceitos para treinamento (compilado dos dicionários)
   - `doc2vecfacil/vocab_fora_quebrados.txt` arquivo informativo sobre os tokens quebrados fora do vocab de treino
   - `doc2vecfacil/vocab_ok_quebrados.txt` arquivo informativo sobre os tokens quebrados incluídos no vocab de treino
   - `doc2vecfacil/vocab_treino.txt` arquivo gerado ao realizar o treinamento, contendo os tokens treinados (pode ser menor ou igual ao vocab_final.txt)
   - `doc2vecfacil/termos_comparacao_treino.txt` arquivo com alguns termos para acompanhamento da evolução do modelo
   - `doc2vecfacil/curadoria_vocab.txt` arquivo com os tokens inteiros e quebrados e vários atributos como: tamanho, tfidf, frequência, etc. Poderá ser aberto no excel para análise e refinamento do dicionário

 - Pode-se gerar um vocab e utilizá-lo para treinar diversos conjuntos diferentes de arquivos em modelos diferentes. 
 - Por isso o arquivo `vocab_treino` pode ser menor que o arquivo `vocab_final`, já que vai conter apenas os termos encontrados durante o treino. Os arquivos criados são sugestões e podem ser alterados livremente antes de iniciar o treinamento do modelo.

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

## Criando o vocab manualmente (opcional):
 Os arquivos necessários para o treino que serão usados para a tokenização são:
    - `meu_modelo\doc2vecfacil\VOCAB_BASE_*.txt`: arquivos com termos que serão treinados 
    - `meu_modelo\doc2vecfacil\VOCAB_REMOVIDO_*.txt`: arquivos com termos que serão ignorados
 - Pode-se criar os arquivos manualmente com os termos desejados, ou aproveitar os arquivos de outro treino. Ou Ajustar os arquivos criados automaticamente.
 - Pode-se conferir os arquivos `.clr` criados nas pastas `textos*` pois eles são o resultado do processamento dos textos originais com o `TokenizadorInteligente`

## Passo a passo para treinar o modelo doc2vec: 
 4) Com os arquivos de vocab prontos, criados automaticamente ou manualmente, pode-se treinar o modelo.
 5) Conferir a pasta de texto e arquivos do vocab:
    - `meu_modelo\textos_treino\`: colocar os arquivos que serão usados no treinamento
    - `meu_modelo\doc2vecfacil\VOCAB_BASE_*.txt`: arquivos com termos que serão treinados 
    - `meu_modelo\doc2vecfacil\VOCAB_REMOVIDO_*.txt`: arquivos com termos que serão ignorados
 5) Rodar: `python util_doc2vec_facil.py -pasta ./meu_modelo`.
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
 - caso existam todos os dicionários, será criado/atualizado o arquivo de curadoria, sem modificar os dicionários existentes.
 - se durante a preparação do dicionário já existirem documetnos na pasta  `textos_treino`, o arquivo de curadoria será criado usando os textos de treinamento, facilitando identificar os termos que não serão treinados, bem como outros atributos de todos os termos.
 - no arquivo de curadoria, a coluna `VOCAB` S/N indica se o termo está contido inteiro no vocab e a coluna `VOCAB_QUEBRADOS` S/N indica se o termo foi incluído após ser fragmentado. Caso as duas colunas sejam N, isso indica que o termo não será treinado, nem inteiro e nem o seu stemmer #sufixo.

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
- fazer pesquisas com More Like This, vetoriais e por proximidade de termos como disponibilizado no componente `PesquisaElasticFacil` em breve.


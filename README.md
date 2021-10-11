# Doc2VecFacil

Componente python que simplifica o processo de criação de um modelo `Doc2Vec` [`Gensim 4.0.1`](https://radimrehurek.com/gensim/) com facilitadores para geração de um vocab personalizado e com a geração de arquivos de curadoria.
- se você não sabe o que é um modelo de similaridade, em resumo é um algoritmo não supervisionado para transformar frases ou documentos em vetores matemáticos que podem ser comparados retornando um valor que representa a similaridade semântica entre dois ou mais documentos. Nesse contexto a máquina 'aprende' o vocabulário treinado e o contexto em que as palavras aparecem, permitindo identificar a similaridade entre os termos, as frases e os documentos.
- Com essa comparação vetorial, é possível encontrar documentos semelhantes a um indicado, agrupar documentos semelhantes de uma lista de documentos e monitorar documentos que entram na base ao compará-los com os documentos marcados como importantes para monitoramento. 
- Esse é um repositório de estudos, analise, ajuste, corrija e use os códigos como desejar.

### Esse componente `Doc2VecFacil` trabalha em duas etapas:
 - criação de um vocab personalizado ao processar textos considerados importantes para o modelo que será treinado
   - `python util_doc2vec_vocab_facil.py -pasta ./meu_modelo`
 - treinamento do modelo usando a estrutura de tokenização criada manualmente ou a partir do código acima
   - `python util_doc2vec_facil.py -pasta ./meu_modelo` -treinar

 - Aqui tem um passo a passo simplificado: [`Passo a Passo`](passo_a_passo_facil.md)
 - Logo abaixo estão as explicações detalhadas de como ele funciona e como usar o seu modelo para pesquisas de documentos semelhantes semanticamente ou textualmente, como realizar agrupamento de documentos por similaridade para auxiliar na organização de documentos usando o ElasticSearch e a pesquisa vetorial.

- :page_with_curl: <b>Códigos</b>: 
  - [`Criação de vocab`](./src/util_doc2vec_vocab_facil.py)
  - [`UtilDoc2VecFacil`](./src/util_doc2vec_facil.py) e [`UtilDoc2VecFacil_Treinamento`](./src/util_doc2vec_facil.py) 
  - [`TradutorTermos`](./src/util_tradutor_termos.py)
  - [`Criação de ngramas`](./src/util_ngramas_facil.py) dicas aqui [NGramasFacil](readme_ngramas.md)

`EM BREVE`: Será disponibilizado um serviço exemplo em conjunto com o componente [PesquisaElasticFacil](https://github.com/luizanisio/PesquisaElasticFacil) para criação de modelos de similaridade textual, agregando valor às pesquisas do ElasticSearch de forma simples com um modelo treinado no corpus específico de cada projeto.

## Criação do vocab personalizado

O arquivo `util_doc2vec_vocab_facil.py` é complementar à classe `Doc2VecFacil` e serve para facilitar a criação de arquivos que configuram o `TokenizadorInteligente`. A ideia é trabalhar com termos importantes para o modelo, adicionados a termos complementares compostos por fragmentos de termos `stemmer` + `sufixo`. Com isso novos documentos que possuam termos fora do vocab principal podem ter o stemmer e o sufixo dentro do vocab do modelo, criando um vocab mais flexível e menor. É possível também transformar termos ou conjunto de termos durante o processamento, como criar n-gramas, reduzir nomes de organizações em sigla, remover termos simples ou compostos etc.

### Arquivo de curadoria para criação do vocab
 - Será criado um arquivo de curadoria `curadoria_planilha_vocab.xlsx` com os termos encontrados nos textos da pasta `textos_vocab`. 
   - coloque aqui textos que contenham boas palavras, limpas de preferência. Podem ser listas retiradas de algum documento, não importa o contexto delas, apenas as palavras nessa primeira etapa. Então listas de palavras e documentos como gramáticas e dicionários de português digitais parecem uma boa opção. Coloque também documentos com palavras relacionadas ao corpus desejado (psicologia, medicina, legislação, administração, etc). Esse site permite uma análise muito boa de termos e suas características [Dicio](https://www.dicio.com.br/).
 - Alguns termos podem não ser tão importantes para o domínio escolhido, mas podem ser importantes para o contexto. Esses termos podem compor o dicionário em forma de `stemmer` + `sufixo`. Aos termos não encontrados no dicionário durante a tokenização para treinamento, será aplicado o stemmer com o sufixo após o stemmer. Caso o stemmer esteja no vocab de treinamento, este será usado. O sufixo é opcional e será incluído se estiver no vocab de treinamento também.
 - Essa combinação de termos completos e fragmentos (stemmer + sufixo) possibilita criar palavras por combinação ao submeter um documento novo ao modelo que contenha termos fora do vocam de treinamento.
   
 - <b>Exemplo</b>: `engloba` pode ser composta por `englob` `#a`, e outras formações podem ocorrer como `englob` `#ada`, `englob` `#adamente`, `englob` `#adas` caso esses  fragmentos estejam disponíveis no vocabulário de treinamento.
   - O vocab de treinamento não precisa do `#` antes do sufixo, apenas dos fragmentos. Mas durante o treinamento os fragmentos usados como sufixo iniciarão com `#` para facilitar sua identificação e diferenciar dos termos principais no modelo final.
 - <b>Exemplo de tokenização</b>: 
   ```python
   # para carregar o modelo, indique a pasta criada no treinamento
   dv = UtilDoc2VecFacil(pasta_modelo='./meu_modelo_treinado')
   print(dv.tokens_sentenca('ATENDIAM A TESTEMUNHA SEU DEPOIMENTO APESAR DE TRAZER ALGUMAS IMPRECISÕES SOBRE OS FATOS ATENDO-SE OS JURADOS ÀS PROVAS PRODUZIDAS EM PLENÁRIOS'))
   ```
   ```
   ['atendiam_testemunha', 'seu', 'depoimento', 'apesar', 'de', 'trazer', 'algumas', 'impreciso', '#es', 'sobre', 'os', 'fatos', 'atend', '#o', 'se', 'os', 'jurados', 'as', 'provas', 'produzidas', 'em', 'plenari', '#os']
   ```
- Veja o [`passo a passo`](passo_a_passo_facil.md) para criar o vocabulário de treinamento de acordo com o cenário desejado.
- O arquivo `curadoria_planilha_vocab.xlsx` tem todos os termos encontrados nos textos da pasta `textos_vocab`, suas frequências, tfidf, tamanho, dentre outros atributos para permitir uma análise e curadoria dos termos. Esse arquivo pode ser aberto no Excel para facilitar a análise/curadoria do vocabulário que será treinado.

- Como funciona o TokenizadorInteligente:
  - Ao ser instanciado, o tokenizador busca os termos do vocab de treinamento contidos nos arquivos com padrão `VOCAB_BASE*.txt` (não importa o case).
  - Podem existir listas de termos que serão excluídos do treinamento, basta esterem em arquivos com o padrão `VOCAB_REMOVIDO*.txt`.
  - Podem existir transformadores de termos nos arquivos com o padrão `VOCAB_TRADUTOR*.txt` que podem conter termos simples ou compostos que será convertidos em outros termos simples ou compostos, como ngramas por exempo. Veja [`NGramasFacil`](readme_ngramas.md) para mais detalhes.
  - Os tradutores funcionam após a limpeza do texto e transformam termos de acordo com a configuração no arquivo:
    - `termo1 => termo2` - converte o `termo1` em `termo2` quando encontrado no texto (ex. `min => ministro`)
    - `termo1 termo2 => termo1_termo2` - converte o termo composto `termo1 termo2` em um termo único `termo1_termo2` (Ex. `processo penal => processo_penal`)
    - `termo1 termo2` - remove o termo composto `termo1 termo2` (Ex. `documento digital => ` ou `documento digital`)
  - Os tradutores podem ser usados para converter nomes de organizações em suas siglas, termos compostos em um termo únicos (ngramas) e até termos conhecidos como idênticos em sua forma mais usual.
> 💡 A ideia de criar vários arquivos é para organizar por domínios. Pode-se, por exemplo, criar um arquivo `VOCAB_BASE portugues.txt` com termos que farão parte de vários modelos, um arquivo `VOCAB_BASE direito.txt` com termos do direito que serão somados ao primeiro no treinamento, um arquivo `VOCAB_BASE direito fragmentos.txt` com fragmentos (`stemmer` + `sufixos`) de termos do direito, e assim por diante. Facilitando evoluções futuras dos vocabulários.

- É importante ressaltar que quanto maior o número de termos, maior o tempo de processamento, mesmo usando recursos otimizados para essa transformação (veja a classe `TradutorTermos` no arquivo [`util_tradutor_termos.py`](./src/util_tradutor_termos.py) ). 
  - Está disponível um gerador de bigramas e quadrigramas aqui [`NGramasFacil`](readme_ngramas.md) para gerar sugestões automáticas de termos que podem ser unificados.
> 💡 A ideia de criar vários arquivos é para organizar por domínios. Pode-se, por exemplo, criar um arquivo VOCAB_BASE_portugues.txt com termos que farão parte de vários modelos, um arquivo VOCAB_BASE_direito.txt com termos do direito que serão somados ao primeiro no treinamento, um arquivo VOCAB_COMPLEMENTAR_direito.txt com fragmentos (`stemmer` + `sufixos`) de termos do direito, e assim por diante.

### Exemplo de arquivo `curadoria_planilha_vocab.xlsx` de curadoria de termos:
| TERMO                  | QUEBRADO         | TFIDF   | TAMANHO |  QTD  | QTD_DOCS | COMPOSTO | VOCAB | VOCAB_QUEBRADOS | ESTRANHO |
|------------------------|------------------|:-------:|:-------:|:-----:|:--------:|:--------:|:------|:---------------:|:--------:|
| acao_penal             |                  | 0,37127 |   	30	  |  178  |   	44	   |    S     |  	S   |        N        |    N     |
| adaptacao              | adaptaca o       | 0,30105 |    10   |   91  |    28    |    N     |   S   |        N        |    N     |
| advogado               | advog ado        | 0,49000 |    7    |  1736 |    810   |    N     |   S   |        N        |    N     |
| custas                 | cust as          | 0,41286 |    6    |  740  |    417   |    N     |   S   |        N        |    N     |
| materia_constitucional	|                  | 0,20749 |   	22	  |   8   |    	2	   |    S	    |   S   |       	N        |    N     |

> 💡 Notas sobre as colunas: 
> - `TFIDF` - contém o maior peso que o termo teve dentre os pesos que teve nos documentos - [Saiba mais sobre `TFIDF`](https://www.ti-enxame.com/pt/python/interpretar-um-resumo-das-pontuacoes-das-palavras-do-tf-idf-nos-documentos/829990829/)
> - `TAMANHO` - é o tamanho do termo
> - `QTD` - é a quantidade de vezes que o termo apareceu no corpus
> - `QTD_DOCS` - é a quantidade de documentos onde o termo apareceu
> - `COMPOSTO` Sim / Não - indica se o termo é composto 
> - `VOCAB` Sim / Não - indica se o termo está presente no vocab principal
> - `VOCAB_QUEBRADOS` Sim / Não - indica se pelo menos o stemmer do termo está presente no vocab principal
> - `ESTRANHO` Sim / Não - termos sem vogais ou com consoantes com várias repetições

## Definição de pastas:
 A estrutura de pastas é pré-definida para facilitar o uso dos componentes. <br>
 O único parâmetro informado é a pasta raiz que vai conter as outras pastas. <br>
 - :file_folder: `Pasta raiz` (informada no parâmetro da chamada - padrão = "meu_modelo")
   - :file_folder: `doc2vecfacil` (pasta do modelo): ao disponibilizar o modelo para uso, pode-se renomear essa pasta livremente
   - :file_folder: `textos_vocab`: textos que serão tokenizados para criação do vocab principal
   - :file_folder: `textos_complementares`: textos que seráo tokenizados para criação do dicionário complementar de fragmentos dos termos não encontrados no vocab principal.


## Passo a passo para criar o vocab de treino: 
 1) Criar as pastas:
    - :file_folder: `meu_modelo`
      - :file_folder: `textos_vocab`: colocar um conjunto de textos importantes para o corpus
      - :file_folder: `textos_vocab_complementar`: colocar um conjunto de textos complementares (tokens serão quebrados)
      - :file_folder: `textos_treino`: colocar os arquivos que serão usados no treinamento
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
 - Pode-se criar arquivos de transformação automaticamente usando o código `util_tradutor_termos.py-pasta = meu_modelo`. Ele vai carregar os arquivos da pasta `textos_vocab` e utilizar o `Phrases` do gensim para sugerir bigramas, trigramas e quadrigramas que poderam ser analisados e incorporados a um arquivo de transformação como `meu_modelo\doc2vecfacil\VOCAB_TRADUTOR_NGRAMAS.txt` por exemplo. Caso queira saber mais sobre a criação de ngramas usando esse componente, veja aqui: [`nGramasFacil`](readme_ngramas.md)

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
    - `-dimensoes` - define o número de dimensões dos vetores de treinamento (não pode ser alterado depois de iniciado o treinamento).
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
- Exemplo de saída do arquivo `comparar_termos.log` atualizado a cada época.
- Esse log é gerado com os termos ou frases disponíveis no arquivo `termos_comparacao_treino.txt` que é carregado no início do treino e pode ser alterado sempre que desejado.
  - o arquivo contém termos linha a linha e frases que podem ser comparadas.
Exemplo do arquivo `termos_comparacao_treino.txt`:
```
apresentada para o réu a decisão sobre o processo = apresentada para o acusado a sentença sobre o processo
artigo
cobrados
cogitar 
compromisso
comprovadas
entendimento
lei
entorpecentes
julga
parcelas
termo
```
> 💡 Nota: na primeira linha temos duas frases que serão comparadas ao longo do treino. Nas outras linhas temos termos soltos que serão apresentados os termos mais parecidos durante o treino. 
> O resultado do arquivo `comparar_termos.log` é esse:
```
apresentada para o réu a decisão sobre o processo | apresentada para o acusado a sentença sobre o processo (65%)
artigo               |  art (77%)                |  artigos (55%)            |  arts (53%)              
cobrados             |  pagos (54%)             
cogitar              |  falar (52%)             
compromisso          |  promessa (57%)          
comprovadas          |  demonstradas (58%)      
entendimento         |  posicionamento (52%)    
lei                  |  lei_federal (75%)        |  lei_complementar (72%)   |  cpp (63%) 
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
 
- O que precisa ser disponibilizado para o modelo funcionar:
  - `VOCAB_BASE*.txt` - arquivo com termos e fragmentos que compõem o vocab
  - `VOCAB_TRADUTOR*.txt` - arquivo de transformações do tokenizados
  - `VOCAB_REMOVER*.txt` - arquivo de exclusões do tokenizados
  - `doc2vec*` - arquivos do modelo treinado

## Dicas de uso:
- gravar os vetores, textos e metadados dos documentos no ElasticSearch
- fazer pesquisas com More Like This, vetoriais e por proximidade de termos como disponibilizado no componente [PesquisaElasticFacil](https://github.com/luizanisio/PesquisaElasticFacil) em breve.


# Doc2VecFacil
Componente python que simplifica o processo de criação de um modelo `Doc2Vec` [`Gensim 4.0.1`](https://radimrehurek.com/gensim/models/doc2vec.html) com facilitadores para geração de um vocab personalizado e com a geração de arquivos de curadoria. Dicas de agrupamento de documentos similares, uso de `ElasticSearch` e `SingleStore`.
- se você não sabe o que é um modelo de similaridade, em resumo é um algoritmo não supervisionado para criar um modelo que transforma frases ou documentos em vetores matemáticos que podem ser comparados retornando um valor equivalente à similaridade semântica de documentos do mesmo contexto/domínio dos documentos usados no treinamento do modelo (doc2vec). Nesse cenário a máquina 'aprende' o vocabulário treinado e o contexto em que as palavras aparecem (word2vec), permitindo identificar a similaridade entre os termos, as frases e os documentos. O doc2vec amplia o treinamento do word2vec para frases ou documentos.
- alguns links para saber mais:
  - [`Paragraph Vector 2014`](https://cs.stanford.edu/~quocle/paragraph_vector.pdf) - a origem do Doc2Vec
  - [`Gensim 4.0.1 Doc2Vec`](https://radimrehurek.com/gensim/auto_examples/tutorials/run_doc2vec_lee.html) - documentação do framework
  - [`me Amilar 2018`](https://repositorio.idp.edu.br/handle/123456789/2635) - Dissertação de Mestrado IDP - Doc2Vec em documentos jurídicos
  - [`Representação Semântica Vetorial`](https://sol.sbc.org.br/index.php/erigo/article/view/7125) - Artigo UFG
  - [`Word2Vec Explained`](https://www.youtube.com/watch?v=yFFp9RYpOb0) - vídeo no Youtube
  - [`Word Embedding Explained and Visualized`](https://www.youtube.com/watch?v=D-ekE-Wlcds) - vídeo no Youtube
  - [`ti-exame`](https://www.ti-enxame.com/pt/python/como-calcular-similaridade-de-sentenca-usando-o-modelo-word2vec-de-gensim-com-python/1045257495/) - Word2Vec
  - [`Tomas Mikolov paper`](https://arxiv.org/pdf/1301.3781.pdf) - mais um artigo sobre o Doc2Vec

- Com essa comparação vetorial, é possível encontrar documentos semelhantes a um indicado ou [`agrupar documentos semelhantes`](./docs/agrupamento.md) entre si de uma lista de documentos. Pode-se armazenar os vetores no `SingleStore` ou `ElasticSearch` para permitir uma pesquisa vetorial rápida e combinada com metadados dos documentos, como nas dicas [aqui](#dicas).

- O uso da similaridade permite também um sistema sugerir rótulos para documentos novos muito similares a documentos rotulados previamente – como uma classificação rápida, desde que o rótulo esteja relacionado ao conteúdo geral do documento e não a informações externas a ele. Rotulação por informações muito específicas do documento pode não funcionar muito bem, pois detalhes do documento podem não ser ressaltados na similaridade semântica. Outra possibilidade seria o sistema sugerir revisão de rotulação/classificação quando dois documentos possuem similaridades muito altas, mas rótulos distintos, ou rótulos iguais para similaridades muito baixas (não é necessariamente um erro, mas sugere-se conferência nesses casos). Ou o sistema pode auxiliar o usuário a identificar rótulos que precisam ser revistos, quando rótulos diferentes são encontrados para documentos muito semelhantes e os rótulos poderiam ser unidos em um único rótulo, por exemplo. Essas são apenas algumas das possibilidades de uso da similaridade. 

- Em um recorte do espaço vetorial criado pelo treinamento do modelo, pode-se perceber que documentos semelhantes ficam próximos enquanto documentos diferentes ficam distantes entre si. Então agrupar ou buscar documentos semelhantes é uma questão de identificar a distância vetorial dos documentos após o treinamento. Armazenando os vetores dos documentos no `ElasticSearch` ou `SingleStore`, essa tarefa é simplificada, permitindo construir sistemas de busca semântica com um esforço pequeno. Uma técnica parecida pode ser usada para treinar e armazenar vetores de imagens para encontrar imagens semelhantes, mas isso fica para outro projeto.

![exemplo recorte espaço vetorial](./exemplos/img_recorte_espaco_vetorial.png?raw=true "Exemplo recorte de espaço vetorial")

> :bulb: Uma dica para conjuntos de documentos com pouca atualização, é fazer o cálculo da similaridade dos documentos e armazenar em um banco transacional qualquer para busca simples pelos metadados da similaridade. Por exemplo uma tabela com as colunas `seq_doc_1`, `seq_doc_2` e `sim` para todos os documentos que possuam similaridade acima de nn% a ser avaliado de acordo com o projeto. Depois basta fazer uma consulta simples para indicar documentos similares ao que o usuário está acessando, por exemplo.

- O core desse componente é o uso de um Tokenizador Inteligente que usa as configurações dos arquivos contidos na pasta do modelo para tokenizar os arquivos de treinamento e os arquivos novos para comparação no futuro (toda a configuração do tokenizador é opcional).
- Esse é um repositório de estudos. Analise, ajuste, corrija e use os códigos como desejar.
> :thumbsup: <sub> Agradecimentos especiais ao Miguel Angelo Neto do Paraná por vários feedbacks contribuindo para a correção de bugs e a melhoria da documentação.</sub><br>

> :warning: <sub>A quantidade de termos treinados e de épocas de treinamento são valores que dependem do objetivo e do tipo de texto de cada projeto. Quanto mais termos, mais detalhes e mais diferenças serão destacadas entre os textos e o modelo vai realçar as particularidades da escrita. Escolhendo menos termos mais relacionados com o domínio analisado, e compondo ngramas, há uma chance maior do modelo realçar a temática geral dos textos.</sub>

### As etapas de um treinamento são simples:
1) reservar um volume de documentos que represente a semântica que será treinada. Então o primeiro passo é extrair e separar em uma pasta os documentos que serão usados no treinamento. É interessante que sejam documentos “texto puro” (não ocerizados), mas não impede que sejam usados documentos ocerizados na falta de documentos “texto puro”. Com textos com muito ruído, como em textos ocerizados, torna-se mais importante a curadoria de termos como será descrito mais abaixo.
2) preparar o ambiente python caso ainda não tenha feito isso: [`anaconda`](https://www.anaconda.com/) + [`requirements`](./src/requirements.txt)
3) baixar os arquivos do [`projeto`](./src/) 
4) baixar um [`modelo`](./exemplos/) ou criar a sua estrutura de pastas
5) rodar o treinamento, a curadoria, o agrupamento e explorar os recursos que o uso do espaço vetorial permite
> :bulb: <sub> Nota: Esse é o [tutorial oficial](https://radimrehurek.com/gensim/auto_examples/tutorials/run_doc2vec_lee.html#introducing-paragraph-vector), a ideia do componetne é simplificar a geração e uso do modelo treinado, mas não há nada muito especial se comparado aos códigos da documentação. </sub>

<hr>

## Esse componente `Doc2VecFacil` trabalha em duas etapas:
 - criação/configuração e curadoria de um [vocab](#vocab) personalizado para o [`TokenizadorInteligente`](./docs/vocab_e_tokenizador.md).
   - `python util_doc2vec_vocab_facil.py -pasta ./meu_modelo`
 - treinamento do modelo usando a estrutura de tokenização criada 
   - `python util_doc2vec_facil.py -pasta ./meu_modelo -treinar`
> 💡 <sub>Nota: para interromper o treino sem correr o risco corromper o modelo durante a gravação, basta criar um arquivo `meu_modelo/doc2vecfacil/parar.txt` na pasta do modelo que o treinamento será interrompido ao final da iteração em andamento.</sub>

 - Aqui tem um [`passo a passo simplificado`](./docs/passo_a_passo_facil.md) para alguns cenários de treinamento com a estrutura de pastas e arquivos, bem como [`exemplos`](./exemplos) de configurações de treinamento.
 - Alguns dos cenários do passo a passo:
   - [Treino simples sem preparação de vocab](./docs/passo_a_passo_facil.md#1-quero-treinar-sem-preparar-um-vocab)
   - [Treino com ngramas](./docs/passo_a_passo_facil.md#3-quero-criar-ngramas-ou-limpar-o-texto-com-termos-que-n%C3%A3o-devem-ser-treinados)
   - [Treino com curadoria de termos](./docs/passo_a_passo_facil.md#4-quero-criar-meu-vocab-do-zero-fazer-curadoria-e-depois-treinar)
 
 - Logo abaixo estão as explicações detalhadas de como funciona o componente e como usar o seu modelo treinado para pesquisas de documentos semelhantes semanticamente (por vetores) e/ou textualmente (por termos), como realizar agrupamento de documentos por similaridade para auxiliar na organização de documentos usando o `ElasticSearch` ou o `SingleStore` com a pesquisa vetorial.

- :page_with_curl: <b>Códigos</b>: 
  - [`Criação de vocab`](./src/util_doc2vec_vocab_facil.py)
  - [`UtilDoc2VecFacil`](./src/util_doc2vec_facil.py) e [`UtilDoc2VecFacil_Treinamento`](./src/util_doc2vec_facil.py) 
  - [`TradutorTermos`](./src/util_tradutor_termos.py)
  - [`Criação de ngramas`](./src/util_ngramas_facil.py) dicas aqui -> [NGramasFacil](./docs/readme_ngramas.md)
  - [`UtilAgrupamentoFacil`](./src/util_agrupamento_facil.py) dicas aqui -> [agrupamento](./docs/agrupamento.md)
  - [`Quebra de parágrafos`](./src/util_quebrar_paragrafos.py) dicas aqui -> [parágrafos para treinamento](./docs/quebrar_paragrafos.md)

`EM BREVE`: Será disponibilizado um serviço exemplo em conjunto com o componente [PesquisaElasticFacil](https://github.com/luizanisio/PesquisaElasticFacil) para criação de modelos de similaridade textual, agregando valor às pesquisas do ElasticSearch de forma simples com um modelo treinado no corpus específico de cada projeto.

## Criação do vocab personalizado <a name="vocab">

O arquivo `util_doc2vec_vocab_facil.py` é complementar à classe `Doc2VecFacil` e serve para facilitar a criação de arquivos que configuram o `TokenizadorInteligente`. A ideia é trabalhar com termos importantes para o modelo, adicionados a termos complementares compostos por fragmentos de termos `stemmer` + `sufixo`. Com isso novos documentos que possuam termos fora do vocab principal podem ter o stemmer e o sufixo dentro do vocab do modelo, criando um vocab mais flexível e menor. É possível também transformar termos ou conjunto de termos durante o processamento, como criar n-gramas, reduzir nomes de organizações em sigla, remover termos simples ou compostos etc.
- veja como funciona o [`TokenizadorInteligente`](./docs/vocab_e_tokenizador.md) e como analisar e interferir no treinamento ajustando os termos dos arquivos de configuração.

## Conferindo o processamento dos textos
- Durante o treinamento são criados arquivos com a extensão `.clr` para cada arquivo `.txt` da pasta `texto_treino`, eles são o resultado do processamento dos textos originais com o `TokenizadorInteligente`.
- Nesses arquivos é possível identificar os fragmentos, os tokens principais e os termos compostos, e verificar se a tokenização está de acordo com o esperado. O treinamento do modelo será feito com esse arquivo e o vocab de treinamento é o resultado da análise desses termos. O objetivo do `TokenizadorInteligente` é preparar o texto que será treinado e fazer o mesmo processo em textos novos para vetorização usando o modelo no futuro. 
- No início do treinamento os arquivos `.clr` serão atualizados para garantir que novos termos incluídos ou alterados manualmente sejam refletidos na tokenização.
- Os arquivos `.clr` são necessários durante todo o treinamento e serão recriados se não forem encontrados, isso acelera o treinamento para não haver necessidade de reprocessar os textos cada vez que o treinamento passar por eles.

## Treinando o modelo doc2vec: 
 Com os arquivos de vocab prontos, criados automaticamente ou manualmente, pode-se treinar o modelo.<br>
 Siga os passos do cenário que atende à sua necessidade: [`passo a passo`](./docs/passo_a_passo_facil.md)

### Parâmetros
 - `python util_doc2vec_facil.py`
    - `-pasta` - nome da pasta de treinamento que contém a pasta do modelo e as pastas de textos, o padrão é `meu_modelo` se não for informada.
    - `-treinar`' - iniciar o treinamento do modelo
    - `-reiniciar sim` - remove o modeo atual, se existir, e inicia um novo treinamento (o sim é para garantir que se quer apagar o modelo e reiniciar). Pode-se também apagar manualmente os arquivos `doc2vec.*` para reiniciar o modelo.
    - `-testar` - carrega o modelo atual, se existir, e apresenta no console a comparação de termos encontrados no arquivo `termos_comparacao_treino.txt` e o agrupamento de textos da pasta `textos_teste`.
    - `-epocas` - define o número de épocas que serão treinadas, o padrão é 5000 e pode ser interrompido ou acrescido a qualquer momento.
    - `-dimensoes` - define o número de dimensões dos vetores de treinamento (não pode ser alterado depois de iniciado o treinamento). Não sabe como escolher use o padrão 300.
    - `-workers` - número de threads de treinamento, padrão 100
    - `-bloco` - número de treinos realizados para cada gravação e testes do modelo - padrão 5

 - `python util_doc2vec_vocab_facil.py`
    - `-pasta` - nome da pasta de treinamento que contém as pastas de textos, o padrão é `meu_modelo` se não for informada.
    - `-treino` - cria a planilha de curadoria com a pasta `textos_treino`, sem esse parâmtro a planilha é criada com a pasta `textos_vocab`.
    - `-teste` - cria a planilha de curadoria com a pasta `textos_teste`, sem esse parâmtro a planilha é criada com a pasta `textos_vocab`.
  > :bulb: <sub>Nota: são muitas opções de uso da curadoria para criação de um bom vocab. Se não sabe por onde começar, copie todos os termos listados na planilha para um arquivo `VOCAB_BASE.txt` e crie um arquivo `VOCAB_REMOVIDO.txt` com stopwords.</sub><br>
  <sub>Caso os seus documentos sejam limpos (não sejam ocerizados), preocupe-se inicialmente apenas com o arquivo `VOCAB_REMOVIDO.txt` para um treino mais simples sem prefixos e sufixos treinando todo o vocabulário e removendo apenas stopwords.</sub>
  
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
> 💡 <sub>Nota: na primeira linha temos duas frases que serão comparadas ao longo do treino. Nas outras linhas temos termos soltos que serão apresentados os termos mais parecidos durante o treino. Coloque quantos termos ou frases desejar. Aparecerão os termos que tiverem similares com mais de 50% de similaridade.</sub><br>
  > <sub>O resultado do arquivo `comparar_termos.log` é esse:</sub>
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

### Arquivos comparados para acompanhar a evolução do modelo:
- Semelhante a comparação de termos, pode-se criar uma pasta `textos_teste` com alguns arquivos para comparação durante o treinamento. 
- Uma sugestão de avaliação do modelo é colocar arquivos com indicadores de grupos para avaliar pelos nomes se estão sendo agrupados como esperado.
- Por exemplo, colocando no nome do arquivo `grupo1`, `grupo2` etc, pode-se avaliar se os arquivos estão sendo agrupados por similaridade da forma que é desejado. Com isso pode-se ajustar os parâmetros do modelo para avaliar se está se aproximando mais ou menos do que se espera. 
- O resultado de comparação será colocado no arquivo `comparar_arquivos.log` como no exemplo abaixo.
- Um dos arquivos é escolhido para ser comparado com ele mesmo para ter mais uma informação para avaliação do modelo. Como informado na documentação do Doc2Vec, chamadas subsequentes de um mesmo conteúdo podem inferir vetores diferentes. Essa diferença diminui aumentando o número de épocas na inferência do vetor. 
- Como a comparação de arquivos pode exigir mais processamento, ela aguarda pelo menos 5 minutos para ser atualizada. Se o arquivo for excluído, ela será realizada após o final da próxima época. A comparação de arquivos também será realizada ao usar o parâmetro `-testar` para testar o modelo.
![exemplo arquivo comparar_arquivos.log](./exemplos/img_comparar_arquivos.png?raw=true "comparar arquivos.log")
> :bulb: <sub>Nota: Essa rotulação não faz o treinamento ser supervisionado, apenas auxilia a avaliação do modelo, já que os rótulos não são levados em consideração no treinamento.</sub><br>
  > <sub>No [`passo a passo`](./docs/passo_a_passo_facil.md) tem dicas de como rotular os documentos para que o treinamento busque aproximar os vetores dos documentos com mesmos rótulos.</sub>  

## Usando o modelo:
O que precisa ser disponibilizado para o modelo funcionar:<br>
 :file_folder: `modelo_teste` <br>
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - `VOCAB_BASE*.txt` - arquivo com termos e fragmentos que compõem o vocab <br>
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - `VOCAB_TRADUTOR*.txt` - arquivo de transformações do tokenizados <br>
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - `VOCAB_REMOVER*.txt` - arquivo de exclusões do tokenizados (stopwords por exemplo) <br>
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - `doc2vec*` - arquivos do modelo treinado <br>
 
 Coloque tudo em uma pasta com o nome que desejar e pronto. Esse é um pacote do seu modelo. <br>

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
print('\nVetor frase 1: ', dv.vetor_sentenca(frase1, normalizado = True, epocas = 3):  
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
Vetor frase 1: [-0.09982843697071075, 0.05516746640205383, 0.09597551822662354, -0.03438882157206535, ...]
 ```
 
## Dicas de uso: <a name="dicas">
- gravar os vetores, textos e metadados dos documentos no [`ElasticSearch`](https://www.elastic.co/pt/), e usar os recursos de pesquisas: More Like This, vetoriais e por proximidade de termos como disponibilizado no componente [`PesquisaElasticFacil`](https://github.com/luizanisio/PesquisaElasticFacil) ou criar sua própria estrutura de dados com [`essas dicas`](https://github.com/luizanisio/PesquisaElasticFacil/blob/main/docs/ElasticQueries.md).
- gravar os vetores, textos e metadados no [`SingleStore`](https://www.singlestore.com/) e criar views de similaridade para consulta em tempo real dos documentos inseridos na base, incluindo filtros de metadados e textuais como nos exemplos disponíveis aqui: [`dicas SingleStore`](./docs/readme_dicas.md).

# Doc2VecFacil
Componente python que simplifica o processo de criação de um modelo `Doc2Vec` [`Gensim 4.0.1`](https://radimrehurek.com/gensim/) com facilitadores para geração de um vocab personalizado e com a geração de arquivos de curadoria.
- se você não sabe o que é um modelo de similaridade, em resumo é um algoritmo não supervisionado para criar um modelo que transforma frases ou documentos em vetores matemáticos que podem ser comparados retornando um valor equivalente à similaridade semântica dos documentos. Nesse contexto a máquina 'aprende' o vocabulário treinado e o contexto em que as palavras aparecem, permitindo identificar a similaridade entre os termos, as frases e os documentos. 
  - alguns links para saber mais: [`me Amilar 2018`](https://repositorio.idp.edu.br/handle/123456789/2635), [`Gensim 4.0.1 Doc2Vec`](https://radimrehurek.com/gensim/auto_examples/tutorials/run_doc2vec_lee.html), [`ti-exame`](https://www.ti-enxame.com/pt/python/como-calcular-similaridade-de-sentenca-usando-o-modelo-word2vec-de-gensim-com-python/1045257495/)
- Com essa comparação vetorial, é possível encontrar documentos semelhantes a um indicado, agrupar documentos semelhantes entre si de uma lista de documentos, e também monitorar documentos que entram na base ao compará-los com os documentos rotulados para monitoramento, como uma classificação rápida.
- Esse é um repositório de estudos, analise, ajuste, corrija e use os códigos como desejar.
- O core desse componente é o uso de um Tokenizador Inteligente que usa as configurações dos arquivos contidos na pasta do modelo para tokenizar os arquivos de treinamento e os arquivos novos para comparação no futuro.
> :warning: A quantidade de termos treinados e de épocas de treinamento são valores que dependem do objetivo e do tipo de texto de cada projeto. Quanto mais termos, mais detalhes e mais diferenças serão destacadas entre os textos e o modelo vai realçar as particularidades da escrita. Escohendo menos termos mais relacionados com o domínio analisado, e compondo ngramas, há uma chance maior do modelo realçar a temática geral dos textos.

### Esse componente `Doc2VecFacil` trabalha em duas etapas:
 - criação/configuração de um vocab personalizado para o Tokenizador Inteligente.
   - `python util_doc2vec_vocab_facil.py -pasta ./meu_modelo`
 - treinamento do modelo usando a estrutura de tokenização criada 
   - `python util_doc2vec_facil.py -pasta ./meu_modelo` -treinar

 - Aqui tem um passo a passo simplificado para criação/configuração do vocab e realização do treinamento: [`Passo a Passo`](passo_a_passo_facil.md)
 
 - Logo abaixo estão as explicações detalhadas de como ele funciona e como usar o seu modelo para pesquisas de documentos semelhantes semanticamente (por vetores) e/ou textualmente (por termos), como realizar agrupamento de documentos por similaridade para auxiliar na organização de documentos usando o ElasticSearch e a pesquisa vetorial.

- :page_with_curl: <b>Códigos</b>: 
  - [`Criação de vocab`](./src/util_doc2vec_vocab_facil.py)
  - [`UtilDoc2VecFacil`](./src/util_doc2vec_facil.py) e [`UtilDoc2VecFacil_Treinamento`](./src/util_doc2vec_facil.py) 
  - [`TradutorTermos`](./src/util_tradutor_termos.py)
  - [`Criação de ngramas`](./src/util_ngramas_facil.py) dicas aqui [NGramasFacil](readme_ngramas.md)

`EM BREVE`: Será disponibilizado um serviço exemplo em conjunto com o componente [PesquisaElasticFacil](https://github.com/luizanisio/PesquisaElasticFacil) para criação de modelos de similaridade textual, agregando valor às pesquisas do ElasticSearch de forma simples com um modelo treinado no corpus específico de cada projeto.

## Criação do vocab personalizado

O arquivo `util_doc2vec_vocab_facil.py` é complementar à classe `Doc2VecFacil` e serve para facilitar a criação de arquivos que configuram o `TokenizadorInteligente`. A ideia é trabalhar com termos importantes para o modelo, adicionados a termos complementares compostos por fragmentos de termos `stemmer` + `sufixo`. Com isso novos documentos que possuam termos fora do vocab principal podem ter o stemmer e o sufixo dentro do vocab do modelo, criando um vocab mais flexível e menor. É possível também transformar termos ou conjunto de termos durante o processamento, como criar n-gramas, reduzir nomes de organizações em sigla, remover termos simples ou compostos etc.

## Como funciona o Tokenizador Inteligente
  - Ao ser instanciado, o tokenizador busca os termos do vocab de treinamento contidos nos arquivos com padrão `VOCAB_BASE*.txt` (não importa o case).
  - Você pode criar listas de termos que serão excluídos do treinamento, basta esteram em arquivos com o padrão `VOCAB_REMOVIDO*.txt`.
  - Há um singularizador de termos automático. Dependendo da terminação do termo, ele será singularizado por regras simples de singularização e caso o resultado desse processamento esteja no vocab, o termo ficará no singular. A ideia é reduzir um pouco mais o vocab nesse passo. Exemplo: o termo `humanos` será convertido em `humano` se o termo `humano` estiver no vocab.
  - Podem existir transformadores de termos nos arquivos com o padrão `VOCAB_TRADUTOR*.txt` que podem conter termos simples ou compostos que serão convertidos em outros termos simples ou compostos, como ngramas por exempo. Veja [`NGramasFacil`](readme_ngramas.md) para mais detalhes.
  - Os tradutores funcionam após a limpeza do texto e transformam termos de acordo com a configuração no arquivo:
    - `termo1 => termo2` - converte o `termo1` em `termo2` quando encontrado no texto (ex. `min => ministro`)
    - `termo1 termo2 => termo1_termo2` - converte o termo composto `termo1 termo2` em um termo único `termo1_termo2` (Ex. `processo penal => processo_penal`)
    - `termo1 termo2` - remove o termo composto `termo1 termo2` (Ex. `documento digital => ` ou `documento digital`)
  - Os tradutores podem ser usados para converter nomes de organizações em suas siglas, termos compostos em um termo únicos (ngramas) e até termos conhecidos como idênticos em sua forma mais usual. É importante ressaltar que quanto maior o número de termos para transformação, maior o tempo de processamento, mesmo usando recursos otimizados para essa transformação (veja a classe `TradutorTermos` no arquivo [`util_tradutor_termos.py`](./src/util_tradutor_termos.py) ). 
    - Está disponível um gerador de bigramas e quadrigramas aqui [`NGramasFacil`](readme_ngramas.md) para gerar sugestões automáticas de termos que podem ser unificados.
  - Os tradutores são processados antes da singularização, então termos transformados em termos compostos não serão afetados pela singularização.

> 💡 A ideia de criar vários arquivos é para organizar por domínios. Pode-se, por exemplo, criar um arquivo `VOCAB_BASE portugues.txt` com termos que farão parte de vários modelos, um arquivo `VOCAB_BASE direito.txt` com termos do direito que serão somados ao primeiro no treinamento, um arquivo `VOCAB_BASE direito fragmentos.txt` com fragmentos (`stemmer` + `sufixos`) de termos do direito, e assim por diante. Facilitando evoluções futuras dos vocabulários.

### Arquivo de curadoria para criação do vocab
 Ao rodar o código `python util_doc2vec_vocab_facil.py -pasta ./meu_modelo`, será criado um arquivo de curadoria de termos `curadoria_planilha_vocab.xlsx` com os termos encontrados nos textos da pasta `textos_vocab`. 
 - coloque na pasta `textos_vocab` textos que contenham boas palavras, limpas de preferência. Podem ser listas retiradas de algum documento, não importa o contexto delas, apenas as palavras nessa primeira etapa. Então listas de palavras e documentos como gramáticas e dicionários de português digitais parecem uma boa opção. Coloque também documentos com palavras relacionadas ao corpus desejado (psicologia, medicina, legislação, administração, etc). Esse site permite uma análise muito boa de termos e suas características, bem como encontrar novos termos derivados de outros termos [`Dicio`](https://www.dicio.com.br/).
 - Alguns termos podem não ser tão importantes para o domínio escolhido, mas podem ser importantes para o contexto. Esses termos podem compor o dicionário em forma de `stemmer` + `sufixo`. Aos termos não encontrados no dicionário durante a tokenização para treinamento, será aplicado o stemmer com o sufixo após o stemmer. Caso o stemmer esteja no vocab de treinamento, este será usado. O sufixo é opcional e será incluído se estiver no vocab de treinamento também.
 - Essa combinação de termos completos e fragmentos (stemmer + sufixo) possibilita criar palavras por combinação ao submeter um documento novo ao modelo que contenha termos fora do vocam de treinamento.
 - Opcionalmente pode-se usar o parâmetro `-treino` para gerar o arquivo de curadoria com base nos arquivos da pasta `texto_treino`.
   
 - <b>Exemplo</b>: `engloba` pode ser composta por `englob` `#a`, e outras formações podem ocorrer como `englob` `#ada`, `englob` `#adamente`, `englob` `#adas` caso esses  fragmentos estejam disponíveis no vocabulário de treinamento.
   - O vocab de treinamento não precisa do `#` antes do sufixo, apenas dos fragmentos. Mas durante o treinamento os fragmentos usados como sufixo iniciarão com `#` para facilitar sua identificação e diferenciar dos termos principais no modelo final.
 - <b>Exemplo de tokenização com termos simples, compostor e fragmentos</b>: 
   ```
   ['atendiam_testemunha', 'seu', 'depoimento', 'apesar', 'de', 'trazer', 'algumas', 'impreciso', '#es', 'sobre', 'os', 'fatos', 'atend', '#o', 'se', 'os', 'jurados', 'as', 'provas', 'produzidas', 'em', 'plenari', '#os']
   ```

- Veja o [`passo a passo`](passo_a_passo_facil.md) para criar o vocabulário de treinamento de acordo com o cenário desejado e realizar o treinamento propriamente dito.

### Exemplo de arquivo `curadoria_planilha_vocab.xlsx` de curadoria de termos:
![recorte curadoria_planilha_vocab.xlsx](./exemplos/img_corte_plan_curadoria.png?raw=true "Recorte planilha curadoria")

> 💡 Notas sobre as colunas: 
> - `TFIDF` - contém a média dos pesos que o termo teve nos documentos - [Saiba mais sobre `TFIDF`](https://www.ti-enxame.com/pt/python/interpretar-um-resumo-das-pontuacoes-das-palavras-do-tf-idf-nos-documentos/829990829/)
> - `TAMANHO` - é o tamanho do termo
> - `QTD` - é a quantidade de vezes que o termo apareceu no corpus
> - `QTD_DOCS` - é a quantidade de documentos onde o termo apareceu
> - `COMPOSTO` Sim / Não - indica se o termo é composto 
> - `VOCAB` - indica se o termo está presente no vocab principal, se es´ta presente quando fragmentado, se é composto ou se não está presente.
> - `ESTRANHO` Sim / Não - termos sem vogais ou com consoantes/vogais com várias repetições
> - Ao final são incluídas algumas colunas indicando a posição das colunas `TFIDF`, `TAMANHO`, `QTD` e `QTD_DOCS` no BoxPlot de cada coluna.

## Conferindo o processamento dos textos
- Pode-se conferir os arquivos `.clr` criados nas pastas `texto_treino` pois eles são o resultado do processamento dos textos originais com o `TokenizadorInteligente`.
- Nesse arquivo é possível identificar os fragmentos, os tokens principais e os termos compostos, e verificar se a tokenização está de acordo com o esperado. O treinamento do modelo será feito com esse arquivo. 
- No início do treinamento os arquivos `.clr` serão atualizados para garantir que novos termos incluídos ou alterados manualmente sejam refletidos na tokenização.
- Os arquivos `.clr` são necessários durante todo o treinamento e serão recriados se não forem encontrados, isso acelera o treinamento para não haver necessidade de reprocessar o texto cada vez que o treinamento passar por ele.

## Treinando o modelo doc2vec: 
 Com os arquivos de vocab prontos, criados automaticamente ou manualmente, pode-se treinar o modelo.<br>
 Siga os passos do cenário que atende à sua necessidade: [`passo a passo`](passo_a_passo_facil.md)

### Parâmetros
 - `python util_doc2vec_facil.py`
    - `-pasta` - nome da pasta de treinamento que contém a pasta do modelo e as pastas de textos, o padrão é `meu_modelo` se não for informada.
    - `-treinar`' - iniciar o treinamento do modelo
    - `-reiniciar sim` - remove o modeo atual, se existir, e inicia um novo treinamento (o sim é para garantir que se quer apagar o modelo e reiniciar)
    - `-testar` - carrega o modelo atual, se existir, e atualiza o arquivo `comparar_termos.log` com os termos encontrados no arquivo `termos_comparacao_treino.txt`
    - `-epocas` - define o número de épocas que serão treinadas, o padrão é 5000 e pode ser interrompido ou acrescido a qualquer momento.
    - `-dimensoes` - define o número de dimensões dos vetores de treinamento (não pode ser alterado depois de iniciado o treinamento).
    - `-workers` - número de threads de treinamento, padrão 100

 - `python util_doc2vec_vocab_facil.py`
    - `-pasta` - nome da pasta de treinamento que contém as pastas de textos, o padrão é `meu_modelo` se não for informada.
    - `-treino` - cria a planilha de curadoria com a pasta `textos_treino`, sem esse parâmtro a planilha é criada com a pasta `textos_vocab`.
    - `-teste` - carrega o `TokenizadorInteligente` para verificar se os arquivos que serão usados para o processamento no treino estão ok.

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
> 💡 Nota: na primeira linha temos duas frases que serão comparadas ao longo do treino. Nas outras linhas temos termos soltos que serão apresentados os termos mais parecidos durante o treino. Coloque quantos termos ou frases desejar. Aparecerão os termos que tiverem similares com mais de 50% de similaridade.<br>
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

### Arquivos comparados para acompanhar a evolução do modelo:
- Semelhante a comparação de termos, pode-se criar uma pasta `textos_teste` com alguns arquivos para comparação durante o treinamento. 
- Uma sugestão de avaliação do modelo é colocar arquivos com indicadores de grupos para avaliar pelos nomes se estão sendo agrupados como esperado.
- Por exemplo, colocando no nome do arquivo `grupo1`, `grupo2` etc, pode-se avaliar se os arquivos estão sendo agrupados por similaridade da forma que é desejado. Com isso pode-se ajustar os parâmetros do modelo para avaliar se está se aproximando mais ou menos do que se espera. 
- O resultado de comparação a cada época será colocado no arquivo `comparar_arquivos.log` como no exemplo abaixo.
![exemplo arquivo comparar_arquivos.log](./exemplos/img_comparar_arquivos.png?raw=true "comparar arquivos.log")
> :bulb: <sub>Nota: Essa rotulação não faz o treinamento ser supervisionado, apenas auxilia a avaliação do modelo, já que os rótulos não são levados em consideração no treinamento.</sub>

## Usando o modelo:
O que precisa ser disponibilizado para o modelo funcionar:<br>
 :file_folder: `modelo_teste` <br>
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - `VOCAB_BASE*.txt` - arquivo com termos e fragmentos que compõem o vocab <br>
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - `VOCAB_TRADUTOR*.txt` - arquivo de transformações do tokenizados <br>
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - `VOCAB_REMOVER*.txt` - arquivo de exclusões do tokenizados <br>
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


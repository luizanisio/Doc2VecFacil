
## Como funciona o Tokenizador Inteligente <a name="tokenizador">
  - Ao ser instanciado, o tokenizador busca os termos do vocab de treinamento contidos nos arquivos com padrão `VOCAB_BASE*.txt` (não importa o case). Caso não seja encontrado nenhum arquivo de termos, todos os termos dos documentos serão considerados.
  - Você pode criar listas de termos que serão excluídos do treinamento, basta estarem em arquivos com o padrão `VOCAB_REMOVIDO*.txt` - aqui geralmente ficam 'stopwords' ou termos que não ajudam na diferenciação dos documentos, bem como termos considerados "ruídos" dos textos com OCR, por exemplo. Os termos contidos na lista de termos removidos não são fragmentados durante a tokenização.
  - Há um singularizador automático de termos. Dependendo da terminação do termo, ele será singularizado por regras simples de singularização - método `singularizar( )` - e caso o resultado desse processamento esteja no vocab, o termo ficará no singular. Funciona para termos compostos também. A ideia é reduzir um pouco mais o vocab nesse passo. 
    - Exemplo: o termo `humanos` será convertido em `humano` se o termo `humano` estiver no vocab. Termos no singular constantes no `VOCAB_REMOVIDO` fazem com que os termos do plural seram removidos também. O resultado de todos os termos removidos pode ser conferido em `vocab_removido.log` gerado ao carregar o tokenizador ou o modelo.
  - Podem existir transformadores de termos nos arquivos com o padrão `VOCAB_TRADUTOR*.txt` que podem conter termos simples ou compostos que serão convertidos em outros termos simples ou compostos, como ngramas por exempo. Veja [`NGramasFacil`](readme_ngramas.md) para mais detalhes.
  - Os tradutores funcionam antes da verificação dos termos do vocab. Todos os conjuntos de transformação são incluídos automaticamente no vocab. 
  - Exemplo de configuração:
    - `termo1 => termo2` - converte o `termo1` em `termo2` quando encontrado no texto (ex. `min => ministro`)
    - `termo1 termo2 => termo1_termo2` - converte o termo composto `termo1 termo2` em um termo único `termo1_termo2` (Ex. `processo penal => processo_penal`)
    - `termo1 termo2` - remove o termo composto `termo1 termo2` (Ex. `documento digital => ` ou `documento digital`)
  - Os tradutores podem ser usados para converter nomes de organizações em suas siglas, termos compostos em um termo únicos (ngramas), correções ortográficas de termos importantes, e até termos conhecidos como idênticos convertendo para a sua forma mais usual. É importante ressaltar que quanto maior o número de termos para transformação, maior o tempo de pré-processamento. A transformação e limpeza ocorrem apenas uma vez e um arquivo `.clr` é criado para cada arquivo `.txt` com o documento pronto para treinamento.
    - Está disponível um gerador de bigramas e quadrigramas aqui [`NGramasFacil`](readme_ngramas.md) para gerar sugestões automáticas de termos que podem ser unificados.
  
> 💡 <sub>A ideia de criar vários arquivos é para organizar por domínios. Pode-se, por exemplo, criar um arquivo `VOCAB_BASE portugues.txt` com termos que farão parte de vários modelos, um arquivo `VOCAB_BASE direito.txt` com termos do direito que serão somados ao primeiro no treinamento, um arquivo `VOCAB_BASE direito fragmentos.txt` com fragmentos (`stemmer` + `sufixos`) de termos do direito, e assim por diante. Facilitando evoluções futuras dos vocabulários.</sub><br>
  <sub>Pode-se realizar o treinamento sem nenhuma configuração de vocab, então todos os termos serão treinados. A ideia de criar um vocab é poder ter algum controle do que será treinado e limpar textos ocerizados que possuem muitos erros e podem aumentar muito o vocab de treinamento com ruídos.</sub>

## Arquivo de curadoria para criação do vocab
 Ao rodar o código `python util_doc2vec_vocab_facil.py -pasta ./meu_modelo`, será criado um arquivo de curadoria de termos `curadoria_planilha_vocab.xlsx` com os termos encontrados nos textos da pasta `textos_vocab` ou, se essa pasta não existir, será usada a pasta `textos_treino`. 
 - o objetivo do arquivo de curadoria é permitir identificar os termos que serão treinados e a importância deles nos documentos (TFIDF, tamanho, frequência etc). Pode-se interferir no treinamento retirando termos ou indicando os termos que serão treinados. Pode-se realizar um treinamento com todos os termos dos documentos, nesse caso não há necessidade de identificar termos que deverão entrar ou sair do treinamento (basta treinar o modelo sem configurar os arquivos de vocab).
 - você pode criar a planilha de curadoria com todos os textos que serão treinados - `textos-treino`, ou com uma amostra dos textos para fazer uma análise dos termos mais relevantes de forma mais rápida - `textos-vocab`. 
 - outra forma de criar uma lista de termos para configurar o vocab de treinamento é colocar diversos documentos relevantes na pasta `textos-vocab`, incluindo gramáticas, lista de termos e textos técnicos com palavras relacionadas ao corpus desejado (psicologia, medicina, legislação, administração, etc). Esse site permite uma análise muito boa de termos e suas características, bem como encontrar novos termos derivados de outros termos [`Dicio`](https://www.dicio.com.br/), aqui temos [`Leis`](http://www4.planalto.gov.br/legislacao/portal-legis/legislacao-1/codigos-1) e documentos Jurídicos nos sites do [`STF`](https://www.stf.jus.br) ou [`STJ`](https://www.stj.jus.br).
 - depois de criar a lista de termos relevantes para o corpus, pode-se incluir eles em um arquivo `VOCAB_BASE termos técnicos.txt` e rodar novamente a planilha de curadoria para a pasta de treino - `textos-treino`. Com isso vai aparecer na planilha se o termo encontrado está ou não no vocab configurado até o momento. É mais uma informação para análise dos termos para criação do vocab final de treinamento.
 - Alguns termos podem não ser tão importantes para o domínio escolhido, mas podem ser importantes para o contexto. Esses termos podem compor o dicionário em forma de `stemmer` + `sufixo`. Aos termos não encontrados no dicionário durante a tokenização para treinamento, será aplicado o [`stemmer`](https://www.nltk.org/_modules/nltk/stem/snowball.html)  com o sufixo após o stemmer. Caso o stemmer esteja no vocab de treinamento, este será usado. O sufixo é opcional e será incluído se estiver no vocab e se o prefixo estiver no vocab de treinamento também.
 - Essa combinação de termos completos e fragmentos (stemmer + sufixo) possibilita criar palavras por combinação ao submeter um documento novo ao modelo que contenha termos fora do vocam de treinamento. Pode ser interessante manter o stemmer apenas dos termos mais relevantes para o contexto do treinamento.
 - Opcionalmente pode-se usar o parâmetro `-treino` para gerar o arquivo de curadoria com base nos arquivos da pasta `texto_treino`, ou o parâmetro `-teste` para os arquivos da pasta `texto_teste`. 
 - o parâmetro `-vocab_treino` pode ser útil para comparar os termos dos textos com os termos realmente treinados no vocab após o início do treinamento. Pode-se combinar os parâmetros `-treino -vocab_treino` ou `-teste -vocab_treino`. Com isso é possível avaliar rapidamente o comportamento do vocabulário selecionado para tokenização até o vocabulário utilizado pelo `Gensim` no treinamento. O arquivo `vocab_treino.txt` é criado após a primeira época de treinamento.
> :bulb: <sub> Nota: é importante ressaltar que após modelo ser gerado, apenas os termos contidos no vocabulário do modelo serão usados na vetorização. Os termos realmente treinados podem ser visualizados no arquivo `vocab_treino.txt` gerado após a primeira época de treinamento.</sub>  
   
 - <b>Exemplo</b>: `engloba` pode ser composta por `englob` `#a`, e outras formações podem ocorrer como `englob` `#ada`, `englob` `#adamente`, `englob` `#adas` caso esses  fragmentos estejam disponíveis no vocabulário de treinamento.
   - O vocab de treinamento não precisa do `#` antes do sufixo, apenas dos fragmentos. Mas durante o treinamento os fragmentos usados como sufixo iniciarão com `#` para facilitar sua identificação e diferenciar dos termos principais no modelo final.
 - <b>Exemplo de tokenização com termos simples, compostos e fragmentos</b>: 
   ```
   ['atendiam_testemunha', 'seu', 'depoimento', 'apesar', 'de', 'trazer', 'algumas', 'impreciso', '#es', 'sobre', 'os', 'fatos', 'atend', '#o', 'se', 'os', 'jurados', 'as', 'provas', 'produzidas', 'em', 'plenari', '#os']
   ```

- Veja o [`passo a passo`](passo_a_passo_facil.md) para criar o vocabulário de treinamento de acordo com o cenário desejado e realizar o treinamento propriamente dito.

### Exemplo de arquivo `curadoria_planilha_vocab.xlsx` de curadoria de termos:
![recorte curadoria_planilha_vocab.xlsx](../exemplos/img_corte_plan_curadoria.png?raw=true "Recorte planilha curadoria")

> 💡 Notas sobre as colunas: 
> - `TFIDF` - contém a média dos pesos que o termo teve nos documentos - Saiba mais sobre [`TFIDF`](https://www.ti-enxame.com/pt/python/interpretar-um-resumo-das-pontuacoes-das-palavras-do-tf-idf-nos-documentos/829990829/), ou [`TFIDF - beatles`](https://iyzico.engineering/how-to-calculate-tf-idf-term-frequency-inverse-document-frequency-from-the-beatles-biography-in-c4c3cd968296).
> - `TAMANHO` - é o tamanho do termo
> - `QTD` - é a quantidade de vezes que o termo apareceu no corpus
> - `QTD_DOCS` - é a quantidade de documentos onde o termo apareceu
> - `COMPOSTO` Sim / Não - indica se o termo é composto 
> - `VOCAB` - indica se o termo está presente no vocab principal, se está presente quando fragmentado, se é composto, se é removido do vocab ou se não está presente no vocab.
> - `ESTRANHO` Sim / Não - termos sem vogais ou com consoantes/vogais com várias repetições
> - Ao final são incluídas algumas colunas indicando a posição das colunas `TFIDF`, `TAMANHO`, `QTD` e `QTD_DOCS` no BoxPlot de cada coluna.


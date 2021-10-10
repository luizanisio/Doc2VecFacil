### Geração de ngramas com curadoria e lista de exclusões

- A ideia desse código é permitir a criação de modelos de bigramas, trigramas e quadrigramas de forma simples e com a facilidade de intervir de alguma forma no modelo gerado pelo Phrases do gensim.
- Dada uma pasta de textos ou uma pasta de modelo para treinamento com Doc2VecFacil, pode-se gerar um modelo de ngramas e usar a classe NGramasFacil() para carregar e usar o modelo quando conveniente.
- Para saber mais sobre a geração de bigramas pelo Gensim, clique aqui: [Gensim Phrases](https://radimrehurek.com/gensim/models/phrases.html)
- A ideia geral é combinar termos que normalmente aparecem juntos em um único token para treinamento de modelos.

## Gerando os modelos de bigramas e nGramas:
- rodar o código `python util_ngramas_facil.py -pasta ./meu_modelo -min_count 20 -threshold 50
- Serão gerados os arquivos:
  - `bigramas.bin` - modelo de bigramas 
  - `bigramas.log` - lista de bigramas gerados no formato texto
  - `quadrigramas.bin` - modelo de trigramas ou quadrigramas
  - `quadrigramas.log` - lista de quadrigramas gerados no formato texto
  - `vocab_sub_bigramas_sugerido.txt` - arquivo com transformações sugeridas para usar no [`Doc2VecFacil`](https://github.com/luizanisio/Doc2VecFacil)
  - `ngramas_remover.txt` - lista de termos (um por linha) que não devem comport bigramas ou quadrigramas. 
    - Exemplo:
    ```
    fls
    documento
    pagina_digitalizada
    ```
  - Esses termos são removidos da análise do Phrases criando uma quebra de linha nos documentos tokenizados durante a análise.
  - Sempre que essa lista for alterada, deve-se rodar novamente o código `util_ngramas_facil.py` para geração de novos ngramas sem esses termos.
  - No exemplo acima, `fls` e `documento` não formarão bigramas, enquanto que `pagina_digitalizada` se formar bigrama, não formará trigrama ou quadrigrama.
  - Essa configuração evitaria bigramas como `penal_fls`, e evitaria `pagina_digitalizada_agravo` caso `pagina_digitalizada` fosse um bigrama formado.
  - A classe já contém uma lista de stop words em português para serem evitadas nos ngramas. 
  ```
  STOP_BR = {"a", "ao", "aos", "aquela", "aquelas", "aquele", "aqueles", "aquilo", "as", "ate", "ate", "com", "como", "da", "das", "de", "dela", "delas", "dele", "deles", "depois", "di", "do", "dos", "du", "e", "e-stj", "e_stj", "ela", "elas", "ele", "eles", "em", "entre", "era", "eram", "eramos", "essa", "essas", "esse", "esses", "esta", "estamos", "estao", "estas", "estava", "estavam", "estavamos", "este", "esteja", "estejam", "estejamos", "estes", "esteve", "estive", "estivemos", "estiver", "estivera", "estiveram", "estiveramos", "estiverem", "estivermos", "estivesse", "estivessem", "estivessemos", "estiveramos", "estivessemos", "estou", "esta", "estavamos", "estao", "eu", "fl", "foi", "fomos", "for", "fora", "foram", "foramos", "forem", "formos", "fosse", "fossem", "fossemos", "fui", "foramos", "fossemos", "ha", "haja", "hajam", "hajamos", "hao", "havemos", "hei", "houve", "houvemos", "houver", "houvera", "houveram", "houveramos", "houverao", "houverei", "houverem", "houveremos", "houveria", "houveriam", "houveriamos", "houvermos", "houvera", "houverao", "houveriamos", "houvesse", "houvessem", "houvessemos", "houveramos", "houvessemos", "ha", "hao", "i", "i-stj", "i_stj", "isso", "isto", "ja", "ja", "lhe", "lhes", "mais", "mas", "me", "mesmo", "meu", "meus", "minha", "minhas", "muito", "na", "nao", "nas", "nem", "no", "nos", "nossa", "nossas", "nosso", "nossos", "num", "numa", "nao", "nos", "o", "os", "ou", "para", "pela", "pelas", "pelo", "pelos", "por", "pra", "qual", "quando", "que", "quem", "sao", "se", "seja", "sejam", "sejamos", "sem", "sera", "serao", "serei", "seremos", "seria", "seriam", "seriamos", "sera", "serao", "seriamos", "seu", "seus", "so", "somos", "sou", "sua", "suas", "sao", "so", "tambem", "tambem", "te", "tem", "temos", "tenha", "tenham", "tenhamos", "tenho", "tera", "terao", "terei", "teremos", "teria", "teriam", "teriamos", "tera", "terao", "teriamos", "teu", "teus", "teve", "tinha", "tinham", "tinhamos", "tive", "tivemos", "tiver", "tivera", "tiveram", "tiveramos", "tiverem", "tivermos", "tivesse", "tivessem", "tivessemos", "tiveramos", "tivessemos", "tu", "tua", "tuas", "tem", "tinhamos", "u", "uaos", "um", "uma", "voce", "voces", "voce", "voces", "vos", "a", "as", "eramos"}
  ```

- Resultado: ao rodar o código, serão mostrados alguns exemplos de bigramas e quadrigramas gerados e o caminho dos arquivos gerados.
```
Iniciando...
Lista de exclusões carregada com 20 termos
Analisando bigramas de 34 arquivos com min_count = 20 e threshold = 50

[..] dados do processamento, exemplos etc [..]

=========================================================================================
nGrams gerados com min_count = 20 e threshold = 50
Sugestões e modelos de bigramas e quadrigramas criados em:  ./meu_modelo\analise_ngramas
=========================================================================================
```

### Usando os ngramas
- O arquivo [`python util_ngramas_facil.py`](./src/util_ngramas_facil.py) tem a classe `TransformarNGramas()` que permite usar os arquivos bigramas.bin e quadrigramas.bin em um conjunto de tokens de um arquivo.
```python
gerador = TransformarNGramas(pasta_ngramas)
texto = 'Conforme Agravo de Instrumento apreciado pelo primeiro grau'
tokens = pre_processamento(texto)

tokens = gerador.transformar( tokens )
print(' '.join(tokens))
```
Resultado:
```
conforme agravo_de_instrumento apreciado pelo primeiro_grau
```


### Geração de ngramas com curadoria e lista de exclusões

- A ideia desse código é facilitar a criação de modelos de bigramas, trigramas e quadrigramas de forma simples e com a facilidade de intervir de alguma forma no modelo gerado pelo Phrases do gensim.
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
- Pode-se analisar as sugestões, incluir outras manualmente e criar um arquivo `VOCAB_TRADUTOR ngramas.txt` para o [`Doc2VecFacil`](https://github.com/luizanisio/Doc2VecFacil).
- Resultado: ao rodar o código, serão mostrados alguns exemplos de bigramas e quadrigramas gerados e o caminho dos arquivos gerados.
```
Iniciando...
Lista de exclusões carregada com 20 termos
Analisando bigramas de 34 arquivos com min_count = 20 e threshold = 50
 - gravando modelo bigramas:  ./meu_modelo\analise_ngramas\bigramas.bin
 - gravando log de Bigramas gerados: bigramas.log
 - número de Bigramas gerados: 38
 - alguns Bigramas: 
   - constrangimento_ilegal
   - organizacao_criminosa

 - gravando modelo Quadrigramas:  ./meu_modelo\analise_ngramas\quadrigramas.bin
 - gravando log de Quadrigramas gerados: quadrigramas.log
 - número de Quadrigramas gerados: 69
 - alguns Quadrigramas:
   - condicoes_pessoais_favoraveis
   - aplicacao_medidas_cautelares

Modelos finalizados:  ./meu_modelo\analise_ngramas\bigramas.bin ./meu_modelo\analise_ngramas\quadrigramas.bin

[..]

=========================================================================================
nGrams gerados com min_count = 20 e threshold = 50
Sugestões e modelos de bigramas e quadrigramas criados em:  ./meu_modelo\analise_ngramas
=========================================================================================
```

### Usando os ngramas
- O arquivo [`python util_ngramas_facil.py`](./src/python util_ngramas_facil.py) tem a classe `TransformarNGramas()` que permite usar os arquivos bigramas.bin e quadrigramas.bin em um conjunto de tokens de um arquivo.
```python
gerador = TransformarNGramas(pasta_ngramas)
texto = 'Conforme Agravo de Instrumento apreciado pelo primeiro grau'
print(' - Aplicanto ngramas em: ', texto)
tokens = gerador.transformar( pre_processamento(texto) )
print(' '.join(tokens))
```
Resultado:
```
conforme agravo_de_instrumento apreciado pelo primeiro_grau
```


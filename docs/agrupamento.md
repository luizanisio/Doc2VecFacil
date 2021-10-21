# Dicas de uma forma rápida de agrupar os documentos vetorizados
- Com vetores que possibilitam a comparação de similaridade de documentos, pode-se agrupar por similaridade um conjunto de arquivos, por exemplo.
- O código abaixo permite o uso do modelo treinado para gerar uma planilha excel com o nome dos arquivos e os grupos formados pela similaridade entre eles.
  
- [`UtilAgrupamentoFacil`](./src/util_agrupamento_facil.py)  

## Para usar o código de agrupamento

- Esse é um exemplo simples de agrupamento, ele pode ser colocado em um serviço, por exemplo, e o usuário escolher os documentos de uma base do `SingleStore` ou `ElasticSearch`, os vetores já estariam gravados e seriam carregados, agrupados e o usuário receberia os grupos ou esses grupos seriam gravados em uma tabela para consulta do usuário.
- Usando o códido por linha de comando:
  - `python util_agrupamento_facil.py -modelo meu_modelo -textos meus_documentos -sim 80`

- os parâmetros de `util_agrupamento_facil.py` são:
  - `-modelo` é a pasta do modelo treinado, podendo ser a pasta com pacote de treinamento ou a pasta final só com o modelo
  - `-textos` é a pasta contendo os arquivos `.txt` que serão agrupados
  - `-sim` é a similaridade mínima para que um ou mais arquivos sejam agrupados
 
- será criado um arquivo com o nome da pasta de textos e a similaridade informada `agrupamento {pasta de textos} sim {similaridade}.xlsx` 
- exemplo de resultado do agrupamento:<br>
![exemplo de agrupamento de arquivos](../exemplos/img_agrupamento.png?raw=true "agrupamento de arquivos") 
 
 
## Como funciona o agrupamento
1. cada arquivo é vetorizado 
2. inicia-se o agrupamento pegando um vetor e buscando todos os com a similaridade mínima e cria-se um grupo
3. pega-se o próximo vetor sem grupo e busca os vetores sem grupo com similaridade mínima e cria-se outro grupo
4. se um vetor não tiver vetores similares, fica no grupo `-1` que é o grupo de órfãos


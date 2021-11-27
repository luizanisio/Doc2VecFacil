# -*- coding: utf-8 -*-

#######################################################################
# Código complementar ao Doc2VecFacil para criar parágrafos de textos para treinamento
# Esse código, dicas de uso e outras informações: 
#   -> https://github.com/luizanisio/Doc2VecFacil/
# Luiz Anísio 
# 25/11/2021 - disponibilizado no GitHub  
#######################################################################

'''
Chamada: 
  - python util_quebrar_paragrafos.py -entrada pasta_textos -saida pasta_trechos -length 200 -tokens 5
  - length = qual o menor tamanho de uma sentença para iniciar uma nova ou unir com a anterior
  - tokens = quantos tokens no mínimo uma sentença precisa ter para ser válida ou ser descartada
  - entrada = nome da pasta com os textos originais
  - saida = nome da pasta para gravação dos trechos 

Primeiro as sentenças são unidas até chegar no mínimo de caracteres. 
Depois os tokens são contados e ela é descartada se não possuir o mínimo de tokens.

  Exemplos:
    - quebrar em sentenças com pelo menos 5 tokens e qualquer tamanho de caracteres
      - python util_quebrar_paragrafos.py -entrada pasta_textos -saida pasta_trechos -length 0 -tokens 5

    - quebrar em sentenças com pelo menos 10 tokens e 200 caracteres
      - python util_quebrar_paragrafos.py -entrada pasta_textos -saida pasta_trechos -length 200 -tokens 5
'''


from util_doc2vec_facil import listar_arquivos, carregar_arquivo, map_thread
from util_agrupamento_facil import progress_bar
import re
import os

MIN_LENGTH_SENTENCA = 1
MIN_TOKENS_SENTENCA = 5

ABREVIACOES = ['sra?s?', 'exm[ao]s?', 'ns?', 'nos?', 'doc', 'ac', 'publ', 'ex', 'lv', 'vlr?', 'vls?',
               'exmo\(a\)', 'ilmo\(a\)', 'av', 'of', 'min', 'livr?', 'co?ls?', 'univ', 'resp', 'cli', 'lb',
               'dra?s?', '[a-z]+r\(as?\)', 'ed', 'pa?g', 'cod', 'prof', 'op', 'plan', 'edf?', 'func', 'ch',
               'arts?', 'artigs?', 'artg', 'pars?', 'rel', 'tel', 'res', '[a-z]', 'vls?', 'gab', 'bel',
               'ilm[oa]', 'parc', 'proc', 'adv', 'vols?', 'cels?', 'pp', 'ex[ao]', 'eg', 'pl', 'ref',
               'reg', 'f[ilí]s?', 'inc', 'par', 'alin', 'fts', 'publ?', 'ex', 'v. em', 'v.rev',
               'des', 'des\(a\)', 'desemb']
#print('REGEX: ',r'(?:\b{})\.\s*$'.format(r'|\b'.join(ABREVIACOES)) )
ABREVIACOES_RGX = re.compile(r'(?:\b{})\.\s*$'.format(r'|\b'.join(ABREVIACOES)), re.IGNORECASE)
PONTUACAO_FINAL = re.compile(r'([\.\?\!]\s+)')
PONTUACAO_FINAL_LISTA = {'.','?','!'}
RE_NUMEROPONTO = re.compile(r'(\d+)\.(?=\d)')

def unir_paragrafos_quebrados(texto):
    lista = texto if type(texto) is list else texto.split('\n')
    res = []
    def _final_pontuacao(_t):
        if len(_t.strip()) == 0:
            return False
        return _t.strip()[-1] in PONTUACAO_FINAL_LISTA
    for i, linha in enumerate(lista):
        _ant = lista[i-1] if i>0 else ""
        #print(f'Lista: [{_ant}]' )
        #print('linha {}: |{}| '.format(i,linha.strip()), _final_pontuacao(linha), ABREVIACOES_RGX.search(lista[i-1]) if i>0 else False)
        if i==0:
            res.append(linha)
        elif (not _final_pontuacao(lista[i-1])) or \
            (_final_pontuacao(lista[i-1]) and (ABREVIACOES_RGX.search(lista[i-1]))):
            # print('juntar: ', lista[i-1].strip(), linha.strip())
            if len(res) ==0: res =['']
            res[len(res)-1] = res[-1].strip() + ' '+ linha
        else:
            res.append(linha)
    return res

def sentencas_arquivo(arq, texto = ''):
    texto = carregar_arquivo(arq, juntar_linhas=False) if arq else [texto]
    sentencas = []
    # quebra as possíveis sentenças pelos pontos finais
    for linha in texto:
        # pontuação ? e ! são garantidas, ponto final pode ser de abreviações
        linha = RE_NUMEROPONTO.sub(r'\1',linha) # números com ponto são unidos para não quebrar como sentença
        linha = linha.replace('? ','\n').replace('! ','\n').replace('.','.\n')
        sents = [f'{_}' for _ in linha.split('\n') if _]
        sentencas.extend(sents)
    # une as sentenças se foram quebradas em abreviações
    sentencas = unir_paragrafos_quebrados(sentencas)
    sentencas = [_.strip() for _ in sentencas if _ ]
    # vai unir as sentenças até o mínimo estipulado
    res = []
    ultima = ''
    for sent in sentencas:
        ultima += f' {sent}'
        if len(ultima) >= MIN_LENGTH_SENTENCA:
            if MIN_TOKENS_SENTENCA<1 or len(ultima.strip().split(' ')) >= MIN_TOKENS_SENTENCA:
                res.append(ultima)
            ultima = ''
    # o resto vai unir com a anterior
    if ultima:
        if any(res):
           res[-1] = f'{res[-1]} {ultima}'
        else:
           if MIN_TOKENS_SENTENCA<1 or len(ultima.strip().split(' ')) >= MIN_TOKENS_SENTENCA:
              res.append(ultima)
    return res

HASHS_INCLUIDOS = set() # inclui apenas as sentenças diferentes
ID_SENT = 0
ID_ARQ = 0
QTD_NOVAS = 0
QTD_REPETIDAS = 0
import unicodedata
def gravar_sentencas(arq):
    global ID_SENT, ID_ARQ, HASHS_INCLUIDOS, QTD_NOVAS, QTD_REPETIDAS
    sentencas = sentencas_arquivo(arq)
    if not any(sentencas):
       return
    ID_ARQ += 1
    for sent in sentencas:
        nm = os.path.split(arq)[1]
        nm = os.path.splitext(nm)[0]
        ID_SENT +=1
        if manter_nome_arquivo:
            nm = os.path.join(pasta_saida, f'{nm} {ID_ARQ}p{ID_SENT}.txt')
        else:
            nm =  os.path.join(pasta_saida, f'arq_{str(ID_ARQ).rjust(9,"0")} p{str(ID_SENT).rjust(9,"0")}.txt')
            sent = unicodedata.normalize('NFD', sent.strip())
            _hash = hash(sent)
            if not _hash in (HASHS_INCLUIDOS):
                with open(nm,'w',encoding='utf8') as f:
                        f.write(sent)
                HASHS_INCLUIDOS.add(_hash)
                QTD_NOVAS +=1
            else:
                QTD_REPETIDAS += 1

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Pasta do modelo')
    parser.add_argument('-entrada', help='pasta contendo os textos que serão quebrados', required=False)
    parser.add_argument('-saida', help='pasta contendo os textos de saída', required=False)
    parser.add_argument('-nome', help='deve manter o nome dos arquivos e suas tags', required=False, action='store_const', const=1)
    parser.add_argument('-length', help='mínimo de caracteres para uma sentença - padrão 200 ', required=False)
    parser.add_argument('-tokens', help='mínimo de tokens para uma sentença - padrão 5 ', required=False)

    args = parser.parse_args()

    pasta_entrada = args.entrada or './textos_entrada'
    pasta_saida = args.saida or './textos_saida'
    manter_nome_arquivo = args.nome
    MIN_LENGTH_SENTENCA = max( int(args.length or MIN_LENGTH_SENTENCA), 1)
    MIN_TOKENS_SENTENCA = max( int(args.tokens or MIN_TOKENS_SENTENCA), 1)

    print('-'*80)
    print(f'Quebrando textos com sentenças de no mínimo {MIN_TOKENS_SENTENCA} tokens e {MIN_LENGTH_SENTENCA} caracteres.')
    print('-'*80)

    '''
    teste = 'em nosso ordenamento jurídico (art. 5º, LXI, LXV e LXVI, da CF). Conforme art.\n105 da CF\nde 1988.\n\n Próximo parágrafo.\n Outro texto qualquer.'
    teste = 'AGRAVO INTERNO. AGRAVO EM RECURSO ESPECIAL. PROCESSO CIVIL (CPC/2015). AÇÃO DE COBRANÇA. NEGATIVA DE PRESTAÇÃO JURISPRUDENCIAL. NÃO OCORRÊNCIA. DISPOSITIVOS SUPOSTAMENTE VIOLADOS. MATÉRIA NÃO PREQUESTIONADA. SÚMULA N. 282/STF. CLÁUSULA DE CONSUMO MÍNIMO. INTERPRETAÇÃO. SÚMULA N. 5/STJ. ADEQUAÇÃO DA DECISÃO AGRAVADA.\nAGRAVO INTERNO DESPROVIDO.'
    teste = sentencas_arquivo(arq='',texto=teste)
    print('Teste: ', teste)
    print(unir_paragrafos_quebrados(teste))
    exit()
    '''

    # se a pasta não tive o modelo dentro, verifica se ele está na subpasta doc2vecfacil
    if not os.path.isdir(pasta_entrada ):
        print(f'ERRO: não foi encontrada a pasta de entrada dos documentos "{pasta_entrada}"')
        exit()
    
    os.makedirs(pasta_saida,exist_ok=True)

    # varre os arquivos da pasta de saída para guardar os hashs e não repetir os conteúdos
    arquivos = listar_arquivos(pasta=pasta_saida)
    if len(arquivos)>0:
        print('Carregando arquivos da pasta de saída para ignorar repetições de conteúdo')
        print(f'- {len(arquivos)} arquivos encontrados')
        print(f'- carregando hashs ... ')
        for i, arq in enumerate(arquivos):
            progress_bar(i+1,len(arquivos),f'Arq {i}')
            texto = carregar_arquivo(arq,juntar_linhas=True)
            HASHS_INCLUIDOS.add(hash(texto))
        ID_ARQ = len(arquivos)
        ID_SENT = len(arquivos)
        print(f'- {len(HASHS_INCLUIDOS)} hashs carregados')

    # varre os arquivos e cria novos arquivos por parágrafo
    qtd = 0
    print('Carregando arquivos da pasta de entrada para ignorar repetições de conteúdo')
    print(f'- {len(arquivos)} arquivos encontrados na pasta "{pasta_entrada}"')
    arquivos = listar_arquivos(pasta=pasta_entrada)
    for i, arq in enumerate(arquivos):
        progress_bar(i+1,len(arquivos),f'Arq {ID_ARQ} - Novas {QTD_NOVAS} - Repet. {QTD_REPETIDAS}          ')
        gravar_sentencas(arq)

    print('-'*80)
    print(f'Quebra de sentenças finalizada com {QTD_NOVAS} sentenças de {len(arquivos)} arquivos.')
    print(f'- {QTD_REPETIDAS} repetidas foram ignoradas')
    print('-'*80)


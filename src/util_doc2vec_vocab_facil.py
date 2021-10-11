# -*- coding: utf-8 -*-

#######################################################################
# Código complementar ao Doc2VecFacil para criar os vocabs de um conjuntos de textos em pastas padronizadas
# Esse código, dicas de uso e outras informações: 
#   -> https://github.com/luizanisio/Doc2VecFacil/
# Luiz Anísio 
# 03/10/2021 - disponibilizado no GitHub  
#######################################################################

from gensim.models import TfidfModel
from gensim.corpora import Dictionary

from util_doc2vec_facil import Documentos, TokenizadorInteligente, carregar_arquivo, UtilDoc2VecFacil
from collections import Counter
import os
import re
import pandas as pd

# TFIDF: https://rockcontent.com/br/blog/tf-idf/

#tokenizador padrão sem dicionário
TOKENIZADOR = TokenizadorInteligente('',registrar_oov=False)
# tamanho máximo de token para registrar na saída
TAMANHO_MAXIMO_TOKEN = 50

from multiprocessing.dummy import Pool as ThreadPool
def map_thread(func, lista, n_threads=5):
    # print('Iniciando {} threads'.format(n_threads))
    pool = ThreadPool(n_threads)
    pool.map(func, lista)
    pool.close()
    pool.join()  

def progress_bar(current_value, total, msg=''):
    increments = 50
    percentual = int((current_value / total) * 100)
    i = int(percentual // (100 / increments))
    text = "\r[{0: <{1}}] {2:.2f}%".format('=' * i, increments, percentual)
    print('{} {}           '.format(text, msg), end="\n" if percentual == 100 else "")

# gera os arquivos e retorna uma lista dos tokens (token, estava no vocab, estava no vocab quebrado)
def criar_arquivo_curadoria_termos(pasta_textos, pasta_vocab = None):
    # definição dos arquivos de curadoria
    arquivo_saida = os.path.join(pasta_vocab,'curadoria_planilha_vocab.xlsx')
    ####################################################################################################
    print('=============================================================')
    print(f'Análise iniciada: ', pasta_textos, ' para ',arquivo_saida)

    print('\t - carregando vocabulário base se disponível')
    if os.path.isdir(pasta_vocab):
       tokenizador_base = TokenizadorInteligente(pasta_vocab=pasta_vocab)
       vocab_base = set(tokenizador_base.vocab)
    else:
        vocab_base = []

    print('\t - carregando documentos para criar vocabulário')

    docs = Documentos(pasta_textos=pasta_textos, retornar_tokens=True, tokenizar_tudo=True,
                      ignorar_cache=True) #, registrar_oov=True) #recria o cache
    # recupera a lista de tokens que entraram depois de quebrados para incluir na análise
    print(f'\t - processando documentos e compilando dicionário')
    dicionario = Dictionary(docs)
    ####################################################################################################
    print('\t - calculando pesos dos termos em cada documento')
    docs = Documentos(pasta_textos=pasta_textos, retornar_tokens=True, ignorar_cache=False)
    print('\t - calculando modelo TFIDF')
    modelo_tfidf = TfidfModel((dicionario.doc2bow(d) for d in docs), normalize = True)
    print('\t - calculando quantidades e maior peso de cada termo nos documentos')
    docs = Documentos(pasta_textos=pasta_textos, retornar_tokens=True, ignorar_cache=False)
    pesos_maximos = dict({})
    contadores = Counter()
    contadores_docs = Counter()
    if pasta_vocab is not None:
        temp_tokenizador = TokenizadorInteligente(pasta_vocab=pasta_vocab)

    i, qtd = 1, docs.high
    for d in docs:
        if i % 100 == 0 or i == qtd:
            progress_bar(i,qtd,f'calculando pesos {i}/{qtd}')
        i += 1
        pesos = pesos_documento(modelo_tfidf, dicionario, d)
        contadores.update(Counter(d))
        contadores_docs.update(Counter(set(d)))
        for c,v in pesos.items():
            pesos_maximos[c] = max(pesos_maximos.get(c,0), v)

    for t in vocab_base:
        if not t in pesos_maximos:
           pesos_maximos[t] = 0
           contadores_docs[t] = 0
           contadores[t] = 0

    re_estranho = re.compile(r'([^rsaei])\1{2,}')
    re_estranho_consoantes = re.compile(r'[^rsaeiounl_\n]{3,}')
    re_vogais = re.compile(r'[aeiouáéíóú]')
    def _estranho(termo):
        # não tem vogais ou tem letras repetidas
        if re_estranho.search(termo) or re_estranho_consoantes.search(termo):
            return 'S'
        if not re_vogais.search(termo):
            return 'S'
        if termo and (termo[0] == '_') or termo[-1]=='_':
            return 'S'
        return 'N'
        
    dados_saida = []    
    for c,v in pesos_maximos.items():
        if len(c) > TAMANHO_MAXIMO_TOKEN:
            continue
        v = round(v,5)
        c = c.replace('#','')
        ok_vocab = 'S' if c in vocab_base else 'N'
        composto = 'S' if c.find('_')>=0 else 'N'
        quebrado = TOKENIZADOR.quebrar_token_simples(c) if composto !='S' else ''
        quebrado = '' if quebrado == c else quebrado
        estranho = 'N' if ok_vocab=='S' else _estranho(c)
        ok_vocab_quebrado = 'S' if ok_vocab == 'N' and quebrado.split(' ')[0] in vocab_base else 'N'
        linha = {'TERMO': c, 'QUEBRADO':quebrado, 'TFIDF': v, 'TAMANHO':len(c), 
                 'QTD' :contadores[c], 
                 'QTD_DOCS' : contadores_docs[c],
                 'COMPOSTO' : composto, 'VOCAB': ok_vocab,  'VOCAB_QUEBRADOS': ok_vocab_quebrado, 'ESTRANHO': estranho}
        dados_saida.append(linha)
    dados_saida = pd.DataFrame(dados_saida)
    dados_saida.to_excel(arquivo_saida,sheet_name=f'Curadoria de vocab para treino')

    print('Análise concluída: ', pasta_textos, ' para ',arquivo_saida)
    print('=============================================================')

def pesos_documento(modelo_tfidf, dicionario, texto):
  tks = TOKENIZADOR.tokenizar(texto) if type(texto) is str else texto
  vec = modelo_tfidf[dicionario.doc2bow(tks)]  
  return {dicionario[pair[0]]: pair[1] for pair in vec}

# faz a leitura de um arquivo de termos e quebra os tokens gerando um arquivo "nome_arquivo (QUEBRADOS).txt"
def quebrar_tokens_arquivo(arquivo):
    if not os.path.isfile(arquivo):
        print('quebrar_tokens_arquivo: aquivo não encontrado: ',arquivo)  
    print('Quebrando tokens do arquivo: ', arquivo)
    arquivo_saida_quebrados = arquivo[:-4] if arquivo.lower().endswith('.txt') else arquivo.replace('.','_')
    arquivo_saida_quebrados = f'{arquivo_saida_quebrados} (QUEBRADOS).txt'
    vocab = carregar_arquivo(arquivo,juntar_linhas=True)
    vocab = TokenizadorInteligente.remover_acentos( vocab.replace('\n',' ').lower() )
    vocab = TokenizadorInteligente.retornar_vocab_texto(vocab)
    print(f'\t - gravando {len(vocab)} tokens criados')
    with open(arquivo_saida_quebrados,'w') as f:
         [f.write(f'{_}\n') for _ in sorted(vocab)]

def testar_tokenizador(texto='', pasta_vocab='./meu_modelo/doc2vecfacil/'):
    txt = texto if texto else 'Esse é um pequeno teste de tokenização de texto'
    tokenizador = TokenizadorInteligente(pasta_vocab=pasta_vocab)
    print(f'Tokenizando o texto: {txt}')
    tokens = tokenizador.tokenizar(txt)
    print(f'Foram retornados {len(tokens)} tokens')
    print(f'Tokens: ', ' '.join(tokens))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Pasta do modelo')
    parser.add_argument('-pasta', help='pasta raiz contendo as pastas do modelo e dos textos', required=False)
    parser.add_argument('-teste', help='realiza a carga do tokenizador para teste - não processa documentos', required=False, action='store_const', const=1)
    parser.add_argument('-treino', help='gera a curadoria com os textos de', required=False, action='store_const', const=1)
    args = parser.parse_args()

    PASTA_BASE = args.pasta if args.pasta else './meu_modelo'
    teste_tokenizar = args.teste
    pasta_treino = args.treino

    if not os.path.isdir(PASTA_BASE):
        msg = f'Não foi encontrada a pasta "{PASTA_BASE}" para criação dos vocabulários'
        raise Exception(msg)

    if teste_tokenizar:
        pasta_base = PASTA_BASE
        if not os.path.isdir(pasta_base):
            if os.path.isdir(os.path.join(pasta_base,'doc2vecfacil')):
                pasta_base = os.path.join(pasta_base,'doc2vecfacil')

        if not os.path.isdir(pasta_base):
            msg = f'Não foi encontrada a pasta "{pasta_base}" para teste do tokenizador'
            raise Exception(msg)
        
        print('###########################################################')
        print('# Carregando tokenizador para testes                      #')
        print('###########################################################')
        testar_tokenizador(pasta_vocab=pasta_base)
        exit()

    PASTA_TEXTOS_VOCAB = os.path.join(PASTA_BASE,'textos_vocab')  
    PASTA_TEXTOS_TREINO = os.path.join(PASTA_BASE,'textos_treino')  
    PASTA_MODELO = os.path.join(PASTA_BASE,'doc2vecfacil')  
    os.makedirs(PASTA_MODELO, exist_ok=True)

    print('###########################################################')
    print('# Criando arquivo de curadoria                            #')
    
    pasta = PASTA_TEXTOS_TREINO if pasta_treino else PASTA_TEXTOS_VOCAB
    print('# >> incluindo textos da pasta "{pasta}" para a curadoria')
    criar_arquivo_curadoria_termos(pasta_textos = pasta, 
                                   pasta_vocab=PASTA_MODELO)
    print('#---------------------------------------------------------#')
    print('# Arquivos de curadoria criados                           #')
    print('# - pasta analisada: ', pasta)
    print('# Abra o arquivo curadoria_planilha_vocab.xlsx no Excel   #')
    print('###########################################################')

    
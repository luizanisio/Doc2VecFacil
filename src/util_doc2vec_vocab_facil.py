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
import random

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
def criar_arquivo_curadoria_termos(pasta_textos, pasta_vocab = None, gerar_curadoria = True):
    arquivo_saida_oov = None
    arquivo_saida_oov_quebrados = None
    arquivo_saida_ok_quebrados = None
    arquivo_saida_vocab = None
    # definição dos arquivos de curadoria
    arquivo_saida = os.path.join(pasta_vocab,'curadoria_planilha_vocab.txt')
    arquivo_saida_oov = os.path.join(pasta_vocab,'curadoria termos inteiros OOV.txt')
    arquivo_saida_oov_quebrados = os.path.join(pasta_vocab,'curadoria termos OOV TOKENS QUEBRADOS.txt')
    arquivo_saida_oov_quebrados_tokens = os.path.join(pasta_vocab,'curadoria termos stemmer OOV TOKENS INTEIROS.txt')
    arquivo_saida_ok_quebrados = os.path.join(pasta_vocab,'curadoria termos VOCAB TOKENS QUEBRADOS.txt')
    arquivo_saida_vocab = os.path.join(pasta_vocab,'curadoria termos VOCAB FINAL PRÉ-TREINO.txt')
    ####################################################################################################
    print('=============================================================')
    print(f'Análise iniciada: ', pasta_textos, ' para ',arquivo_saida)
    print('\t - carregando documentos para criar vocabulário')
    docs = Documentos(pasta_textos=pasta_textos, retornar_tokens=True, tokenizar_tudo=True,
                      pasta_vocab=pasta_vocab,
                      ignorar_cache=True, registrar_oov=arquivo_saida_oov) #recria o cache
    # recupera a lista de tokens que entraram depois de quebrados para incluir na análise
    vocab_base_quebrados = set(docs.tokenizer.oov_quebrados_tokens) 
    tokenizador_inicial = docs.tokenizer
    print('\t - processando documentos e compilando dicionário de termos')
    dicionario = Dictionary(docs)
    if not gerar_curadoria:
        print(f'\t - curadoria ignorada - retornando tokenizador com {len(tokenizador_inicial.vocab_final)} termos no vocab')
        return tokenizador_inicial
    ####################################################################################################
    # ao compilar o dicionário, o docs vai conter os tokens quebrados e inteiros como oov
    # pois nenhum vocabulário foi passado para o tokenizador
    if gerar_curadoria:
        print(f'\t - gravando {len(docs.tokenizer.vocab)} tokens no arquivo {arquivo_saida_vocab}')
        with open(arquivo_saida_vocab,'w') as f:
             [f.write(f'{_}\n') for _ in sorted(docs.tokenizer.vocab_final)]
        print(f'\t - gravando {len(docs.tokenizer.oov)} tokens no arquivo {arquivo_saida_oov}')
        with open(arquivo_saida_oov,'w') as f:
             [f.write(f'{_}\n') for _ in sorted(docs.tokenizer.oov)]
        print(f'\t - gravando {len(docs.tokenizer.oov_quebrados)} tokens no arquivo {arquivo_saida_oov_quebrados}')
        with open(arquivo_saida_oov_quebrados,'w') as f:
             [f.write(f'{_}\n') for _ in sorted(docs.tokenizer.oov_quebrados)]
        print(f'\t - gravando {len(docs.tokenizer.oov_quebrados_tokens)} tokens no arquivo {arquivo_saida_oov_quebrados_tokens}')
        with open(arquivo_saida_oov_quebrados_tokens,'w') as f:
             [f.write(f'{_}\n') for _ in sorted(docs.tokenizer.oov_quebrados_tokens)]
        print(f'\t - gravando {len(docs.tokenizer.ok_quebrados)} tokens no arquivo {arquivo_saida_ok_quebrados}')
        with open(arquivo_saida_ok_quebrados,'w') as f:
             [f.write(f'{_}\n') for _ in sorted(docs.tokenizer.ok_quebrados)]
    ####################################################################################################
    print('\t - calculando peso dos termos')
    docs = Documentos(pasta_textos=pasta_textos, retornar_tokens=True, ignorar_cache=False)
    print('\t - calculando modelo TFIDF')
    modelo_tfidf = TfidfModel((dicionario.doc2bow(d) for d in docs), normalize = True)
    print('\t - calculando quantidades e maior peso de cada termo nos documentos')
    docs = Documentos(pasta_textos=pasta_textos, retornar_tokens=True, ignorar_cache=False)
    pesos_maximos = dict({})
    contadores = Counter()
    contadores_docs = Counter()
    vocab_base = set()
    if pasta_vocab is not None:
        temp_tokenizador = TokenizadorInteligente(pasta_vocab=pasta_vocab)
        vocab_base = set(temp_tokenizador.vocab)
        print(f'\t - vocab base para análise de termos carregado com {len(vocab_base)} tokens')
        print(f'\t - vocab quebrado para análise de termos carregado com {len(vocab_base_quebrados)} tokens')
        print(f'\t - vocab termos de tradução carregados com {len(temp_tokenizador.vocab_tradutor_termos)} linhas')
    #termos_completos = set({})
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

    with open(arquivo_saida,'w') as f:
        f.write(f'TERMO\tTFIDF\tTAMANHO\tQTD\tQTD_DOCS\tCOMPOSTO\tVOCAB\tVOCAB_QUEBRADOS\n')
        for c,v in pesos_maximos.items():
            if len(c) > TAMANHO_MAXIMO_TOKEN:
                continue
            v = str(v).replace('.',',') # para uso no excel
            c = c.replace('#','')
            ok_vocab = 'S' if c in vocab_base else 'N'
            ok_vocab_quebrados = 'S' if c in vocab_base_quebrados else 'N'
            composto = 'S' if c.find('_')>=0 else 'N'
            # registra os novos tokens para criar um novo vocab
            f.write(f'{c}\t{v}\t{len(c)}\t{contadores[c]}\t{contadores_docs[c]}\t{composto}\t{ok_vocab}\t{ok_vocab_quebrados}\n')

    print('Análise concluída: ', pasta_textos, ' para ',arquivo_saida)
    print('=============================================================')
    return tokenizador_inicial

def pesos_documento(modelo_tfidf, dicionario, texto):
  tks = TOKENIZADOR.tokenizar(texto) if type(texto) is str else texto
  vec = modelo_tfidf[dicionario.doc2bow(tks)]  
  return {dicionario[pair[0]]: pair[1] for pair in vec}

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
    parser.add_argument('-reiniciar', help='reprocessa os arquivos de vocab mesmo existindo', required=False, action='store_const', const=1)
    args = parser.parse_args()

    PASTA_BASE = args.pasta if args.pasta else './meu_modelo'
    teste_tokenizar = args.teste
    forcar_processamento = args.reiniciar

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

    PASTA_TEXTOS = os.path.join(PASTA_BASE,'textos_vocab')  
    PASTA_TEXTOS_TREINO = os.path.join(PASTA_BASE,'textos_treino')  
    PASTA_TEXTOS_COMPLEMENTARES = os.path.join(PASTA_BASE,'textos_vocab_complementar')  
    PASTA_MODELO = os.path.join(PASTA_BASE,'doc2vecfacil')  
    ARQUIVO_COMPARACAO = os.path.join(PASTA_MODELO,'termos_comparacao_treino.txt')  
    os.makedirs(PASTA_MODELO, exist_ok=True)

    ARQUIVO_VOCAB_AUTOMATICO = os.path.join(PASTA_MODELO,'VOCAB_BASE_AUTOMATICO.txt')
    ARQUIVO_VOCAB_AUTOMATICO_C = os.path.join(PASTA_MODELO,'VOCAB_BASE_AUTOMATICO (fragmentos).txt')

    principal_ok = False 
    complementar_ok = False 

    # reprocessa mesmo existindo os arquivos
    if forcar_processamento:
        forcar_processamento = os.path.isfile(ARQUIVO_VOCAB_AUTOMATICO) or os.path.isfile(ARQUIVO_VOCAB_AUTOMATICO_C)
        if os.path.isfile(ARQUIVO_VOCAB_AUTOMATICO):
            os.remove(ARQUIVO_VOCAB_AUTOMATICO)
        if os.path.isfile(ARQUIVO_VOCAB_AUTOMATICO_C):
            os.remove(ARQUIVO_VOCAB_AUTOMATICO_C)
        # remove o modelo caso já exista uma versão pois mudando o vocab tem que treinar novamente
        lista = UtilDoc2VecFacil.get_arquivos_modelo(PASTA_MODELO)
        for arq in lista:
            if os.path.isfile(arq):
               os.remove(arq)

    # não encontrou arquivos de vocab principais, cria tudo
    if  (not os.path.isfile(ARQUIVO_VOCAB_AUTOMATICO)):
        print('###########################################################')
        if forcar_processamento:
            print('# Reiciando processamento do vocabulário principal:       #')
        else:
            print('# Iniciando processamento do vocabulário principal:       #')
        tokenizador = criar_arquivo_curadoria_termos(pasta_textos = PASTA_TEXTOS,
                                                     pasta_vocab=PASTA_MODELO,
                                                     gerar_curadoria=False)
        # todos os tokens processados serão incluídos
        with open(ARQUIVO_VOCAB_AUTOMATICO,'w') as f:
            [f.write(f'{_}\n') for _ in tokenizador.oov]
        # cria arquivo de comparação de termos se não existir
        if not os.path.isfile(ARQUIVO_COMPARACAO):
            termos = [_ for _ in tokenizador.vocab_final if len(_)>3 and _[0] !='#']
            random.shuffle(termos)
            termos = termos[:100]
            with open(ARQUIVO_COMPARACAO,'w') as f:
                [f.write(f'{_}\n') for _ in termos]
        # remove os complementares para gerar novamente
        if os.path.isfile(ARQUIVO_VOCAB_AUTOMATICO_C):
            os.remove(ARQUIVO_VOCAB_AUTOMATICO_C)
        print('# Arquivo de curadoria principal criado                   #')
    else:
        principal_ok = True
        
    # não encontrou arquivos de vocab complementares e tem textos complementares, cria a curadoria com eles
    if (not os.path.isfile(ARQUIVO_VOCAB_AUTOMATICO_C)) \
        and os.path.isdir(PASTA_TEXTOS_COMPLEMENTARES):
        print('###########################################################')
        if forcar_processamento:
            print('# Reiciando processamento do vocabulário complementar:    #')
        else:
            print('# Iniciando processamento do vocabulário complementar:    #')
        pastas = [PASTA_TEXTOS, PASTA_TEXTOS_COMPLEMENTARES]
        tokenizador = criar_arquivo_curadoria_termos(pasta_textos = pastas, 
                                                     pasta_vocab=PASTA_MODELO,
                                                     gerar_curadoria=False)
        # todos os tokens fora do vocab serão incluídos
        with open(ARQUIVO_VOCAB_AUTOMATICO_C,'w') as f:
            [f.write(f'{_}\n') for _ in tokenizador.oov_quebrados]
        print('# Arquivo de curadoria complementar criado                #')
    else:
        complementar_ok = True

    print('###########################################################')
    if principal_ok:
        print('# Vocabulário principal encontrado                        #')
    else:
        print('# Vocabulário principal criado                            #')
    if complementar_ok:
        print('# Vocabulário complementar encontrado                     #')
    else:
        print('# Vocabulário complementar criado                         #')
    print('###########################################################')

    print('###########################################################')
    print('# Criando arquivo de curadoria final                      #')
    pastas = [PASTA_TEXTOS, PASTA_TEXTOS_COMPLEMENTARES]
    if os.path.isdir(PASTA_TEXTOS_TREINO):
        pastas.append(PASTA_TEXTOS_TREINO)
        print('# >> incluindo textos de treinamento na curadoria         #')
    criar_arquivo_curadoria_termos(pasta_textos = pastas, 
                                   pasta_vocab=PASTA_MODELO, 
                                   gerar_curadoria=True)
    print('#---------------------------------------------------------#')
    print('# Arquivos de curadoria criados                           #')
    print('# Abra o arquivo curadoria_vocab.txt no Excel             #')
    print('###########################################################')

    
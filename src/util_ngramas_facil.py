# -*- coding: utf-8 -*-
#######################################################################
# Código complementar ao Doc2VecFacil para criar nGramas para o documento VOCAB_TRADUTOR.txt 
# Esse código, dicas de uso e outras informações: 
#   -> https://github.com/luizanisio/Doc2VecFacil/
# Luiz Anísio 
# 09/10/2021 - disponibilizado no GitHub  
#######################################################################

import random
import glob
import os
from time import sleep
from gensim.models.phrases import Phrases, Phraser
import re
from util_doc2vec_facil import TokenizadorInteligente, carregar_arquivo
from timeit import default_timer as timer

TOKENIZADOR = TokenizadorInteligente('',registrar_oov=False, tokenizar_tudo=True, fragmentar=False)

# exemplos de conjuntos de stop_words para os ngramas
STOP_MINI = {'de', 'da', 'do', 'para', 'ao', 'a', 'e', 'i', 'o', 'u', 'pra', 'di', 'du', 'por', 'em', 'num', 'ao','aos','no', 'na' }
STOP_BR_ACENTUADO = {"a", "ao", "aos", "aquela", "aquelas", "aquele", "aqueles", "aquilo", "as", "ate", "até", "com", "como", "da", "das", "de", "dela", "delas", "dele", "deles", "depois", "di", "do", "dos", "du", "e", "e-stj", "e_stj", "ela", "elas", "ele", "eles", "em", "entre", "era", "eram", "eramos", "essa", "essas", "esse", "esses", "esta", "estamos", "estao", "estas", "estava", "estavam", "estavamos", "este", "esteja", "estejam", "estejamos", "estes", "esteve", "estive", "estivemos", "estiver", "estivera", "estiveram", "estiveramos", "estiverem", "estivermos", "estivesse", "estivessem", "estivessemos", "estivéramos", "estivéssemos", "estou", "está", "estávamos", "estão", "eu", "fl", "foi", "fomos", "for", "fora", "foram", "foramos", "forem", "formos", "fosse", "fossem", "fossemos", "fui", "fôramos", "fôssemos", "ha", "haja", "hajam", "hajamos", "hao", "havemos", "hei", "houve", "houvemos", "houver", "houvera", "houveram", "houveramos", "houverao", "houverei", "houverem", "houveremos", "houveria", "houveriam", "houveriamos", "houvermos", "houverá", "houverão", "houveríamos", "houvesse", "houvessem", "houvessemos", "houvéramos", "houvéssemos", "há", "hão", "i", "i-stj", "i_stj", "isso", "isto", "ja", "já", "lhe", "lhes", "mais", "mas", "me", "mesmo", "meu", "meus", "minha", "minhas", "muito", "na", "nao", "nas", "nem", "no", "nos", "nossa", "nossas", "nosso", "nossos", "num", "numa", "não", "nós", "o", "os", "ou", "para", "pela", "pelas", "pelo", "pelos", "por", "pra", "qual", "quando", "que", "quem", "sao", "se", "seja", "sejam", "sejamos", "sem", "sera", "serao", "serei", "seremos", "seria", "seriam", "seriamos", "será", "serão", "seríamos", "seu", "seus", "so", "somos", "sou", "sua", "suas", "são", "só", "tambem", "também", "te", "tem", "temos", "tenha", "tenham", "tenhamos", "tenho", "tera", "terao", "terei", "teremos", "teria", "teriam", "teriamos", "terá", "terão", "teríamos", "teu", "teus", "teve", "tinha", "tinham", "tinhamos", "tive", "tivemos", "tiver", "tivera", "tiveram", "tiveramos", "tiverem", "tivermos", "tivesse", "tivessem", "tivessemos", "tivéramos", "tivéssemos", "tu", "tua", "tuas", "tém", "tínhamos", "u", "uaos", "um", "uma", "voce", "voces", "você", "vocês", "vos", "à", "às", "éramos"}
STOP_BR = {"a", "ao", "aos", "aquela", "aquelas", "aquele", "aqueles", "aquilo", "as", "ate", "ate", "com", "como", "da", "das", "de", "dela", "delas", "dele", "deles", "depois", "di", "do", "dos", "du", "e", "e-stj", "e_stj", "ela", "elas", "ele", "eles", "em", "entre", "era", "eram", "eramos", "essa", "essas", "esse", "esses", "esta", "estamos", "estao", "estas", "estava", "estavam", "estavamos", "este", "esteja", "estejam", "estejamos", "estes", "esteve", "estive", "estivemos", "estiver", "estivera", "estiveram", "estiveramos", "estiverem", "estivermos", "estivesse", "estivessem", "estivessemos", "estiveramos", "estivessemos", "estou", "esta", "estavamos", "estao", "eu", "fl", "foi", "fomos", "for", "fora", "foram", "foramos", "forem", "formos", "fosse", "fossem", "fossemos", "fui", "foramos", "fossemos", "ha", "haja", "hajam", "hajamos", "hao", "havemos", "hei", "houve", "houvemos", "houver", "houvera", "houveram", "houveramos", "houverao", "houverei", "houverem", "houveremos", "houveria", "houveriam", "houveriamos", "houvermos", "houvera", "houverao", "houveriamos", "houvesse", "houvessem", "houvessemos", "houveramos", "houvessemos", "ha", "hao", "i", "i-stj", "i_stj", "isso", "isto", "ja", "ja", "lhe", "lhes", "mais", "mas", "me", "mesmo", "meu", "meus", "minha", "minhas", "muito", "na", "nao", "nas", "nem", "no", "nos", "nossa", "nossas", "nosso", "nossos", "num", "numa", "nao", "nos", "o", "os", "ou", "para", "pela", "pelas", "pelo", "pelos", "por", "pra", "qual", "quando", "que", "quem", "sao", "se", "seja", "sejam", "sejamos", "sem", "sera", "serao", "serei", "seremos", "seria", "seriam", "seriamos", "sera", "serao", "seriamos", "seu", "seus", "so", "somos", "sou", "sua", "suas", "sao", "so", "tambem", "tambem", "te", "tem", "temos", "tenha", "tenham", "tenhamos", "tenho", "tera", "terao", "terei", "teremos", "teria", "teriam", "teriamos", "tera", "terao", "teriamos", "teu", "teus", "teve", "tinha", "tinham", "tinhamos", "tive", "tivemos", "tiver", "tivera", "tiveram", "tiveramos", "tiverem", "tivermos", "tivesse", "tivessem", "tivessemos", "tiveramos", "tivessemos", "tu", "tua", "tuas", "tem", "tinhamos", "u", "uaos", "um", "uma", "voce", "voces", "voce", "voces", "vos", "a", "as", "eramos"}

# define o conjunto usado nos ngramas - não são só as stop-words, são também termos do 
# texto mas que não são interessantes para compor ngramas
STOP_NGRAMAS = STOP_BR
# alguns termos extras podem comport essa lista para não formar ngramas 
#STOP_NGRAMAS.update({'pauta','pastadigital','poder','outro','informe','um','dois','três','quatro','cinco','seis','sete','oito','nove','dez','dias','listview','seam'})
# alguns termos compostos podem compor essa lista para não formar trigramas ou quadrigramas
#STOP_NGRAMAS.update({'assinado_eletronicamente'})

def pre_processamento(texto, realizar_split = True):
    # realiza a tokenização rápida com ou sem split
    rapido = 'str' if not realizar_split else True
    return TOKENIZADOR.tokenizar(texto, rapido=rapido)

def conteudo_arquivo(arquivo, realizar_split = True):
    linhas = carregar_arquivo(arquivo, False)
    #with open(arquivo,'r', encoding='utf-8') as f:
    #  linhas = [pre_processamento(_) for _ in  f.readlines() if _.strip()]
    linhas = [pre_processamento(_, realizar_split=realizar_split) for _ in  linhas if _.strip()]
    linhas = [_ for _ in linhas if _]
    #print(linhas)
    return linhas 

def listar_arquivos(pasta, mascara):
    return [f for f in glob.glob(os.path.join(pasta,mascara))]

# retorna [(nome, conteudo), ....]
def IterarArquivos(pasta, mascara, realizar_split = True):
    current = 0
    arquivos = listar_arquivos(pasta=pasta, mascara=mascara)
    while current < len(arquivos)-1:
        nome = os.path.split(arquivos[current])[1]
        conteudo = conteudo_arquivo(arquivos[current], realizar_split=realizar_split)
        yield (nome,conteudo)
        current += 1

# quebra as sentenças nos termos não interessantes para compor ngramas 
# o uso do parâmetro common_terms=frozenset(STOP_BIGRAMAS) não teve um resultado muito bom
# a ideia é enganar o gerador de ngramas já que esses termos serão omitidos das sentenças apenas na geração de ngramas
# se eles fossem só apagados, outros ngramas estranhos poderiam ser formados
# aproveita e remove sentenças de um token
# recebe uma lista de sentenças que são listas de tokens [['termo1','termo2','stop','termo3','termo4'],['termo1','termo7','termo9']]
# retorna uma nova lista de sentenças quebradas nos stops [['termo1','termo2'],['termo3','termo4'],['termo1','termo7','termo9']]
def quebrar_sentencas_stop_ngramas(lista_de_sentencas):
    novas_sentencas = []
    for tokens in lista_de_sentencas:
        nova_sentenca = [ ]
        for token in tokens:
            if token in STOP_NGRAMAS:
                if len(nova_sentenca)>1:
                   novas_sentencas.append(nova_sentenca)
                   nova_sentenca=[]
            else:
                nova_sentenca.append(token)
        # o que sobrou da sentença analisada entra como nova
        if len(nova_sentenca)>1:
          novas_sentencas.append(nova_sentenca)
    #print('antes: ', lista_de_sentencas)
    #print('depois: ', novas_sentencas)
    return novas_sentencas

# gera modelo de bigramas
def gerar_bigramas_preprocessados(pasta_textos=r'.\textos', pasta_saida_ngramas=None, min_count=10, threshold = 20):
    # looping para ngramas
    iterarq_1 = IterarArquivos(pasta = pasta_textos, mascara = '*.txt')
    iterarq_2 = IterarArquivos(pasta = pasta_textos, mascara = '*.txt')
    qtd_arquivos = len(listar_arquivos(pasta = pasta_textos, mascara = '*.txt'))
    pasta_saida = pasta_saida_ngramas if pasta_saida_ngramas else pasta_textos 
    arquivo_bigrama = os.path.join(pasta_saida,'bigramas.bin')
    arquivo_quadrigrama = os.path.join(pasta_saida,'quadrigramas.bin')
    lst_vocab_sub = []
    # verifica se existe o arquivo ngramas_remover.txt com termos que não devem compor nGramas
    arquivo_remover = os.path.join(pasta_saida,'ngramas_remover.txt')
    if os.path.isfile(arquivo_remover):
        lista = carregar_arquivo(arquivo_remover, juntar_linhas=False)
        lista = [pre_processamento(_, realizar_split=False).strip() for _ in lista if _ and _[0] !='#']
        if any(lista):
            global STOP_NGRAMAS
            STOP_NGRAMAS.update(lista)
            print(f'Lista de exclusões carregada com {len(lista)} termos')
            #print('- ', list(STOP_NGRAMAS))
    else:
        with open(arquivo_remover,'w') as f:
            f.write('### lista de termos para remoção ###\n')

    phrases = Phrases(min_count=min_count, threshold=threshold)
    print(f'Analisando bigramas de {qtd_arquivos} arquivos com min_count = {min_count} e threshold = {threshold}')
    # adiciona as frases ao Phrases para análise
    for _, conteudo in iterarq_1:
        conteudo = quebrar_sentencas_stop_ngramas(conteudo)
        phrases.add_vocab(conteudo)
    gerador = Phraser(phrases)
    print(' - gravando modelo bigramas: ', arquivo_bigrama)
    gerador.save(arquivo_bigrama)
    #print(list(gerador.phrasegrams))
    #lst = [f'{g[0].decode()}_{g[1].decode()}' for g in gerador.phrasegrams]
    lst = list(gerador.phrasegrams)
    lst.sort(key= lambda k:len(k), reverse=True)
    lst_vocab_sub.extend(lst)
    print(' - gravando log de Bigramas gerados: bigramas.log')
    with open(os.path.join(pasta_saida,'bigramas.log'),'w') as f:
         f.writelines('\n'.join(lst))
    print(' - número de Bigramas gerados:', len(lst))
    print(' - alguns Bigramas: ')
    lst.sort(key= lambda k:len(k), reverse=True)
    [print(f'   - {_}') for _ in lst[:10]]
    ############ QUADRIGRAMAS
    q=0
    phrases_qdr = Phrases(min_count=min_count, threshold=10)
    print('')
    # adiciona as frases já com os bigramas ao Phrases para análise de quadrigramas
    # aqui podem ser formados trigramas ou quadrigramas (token+bigrama ou bigrama+bigrama)
    for _, conteudo in iterarq_2:
        sents_qdr = [gerador[tokens] for tokens in conteudo]
        sents_qdr = quebrar_sentencas_stop_ngramas(sents_qdr)
        phrases_qdr.add_vocab(sents_qdr)
    gerador_qrd = Phraser(phrases_qdr)
    print(' - gravando modelo Quadrigramas: ', arquivo_quadrigrama)
    gerador_qrd.save(arquivo_quadrigrama)
    #lst = [f'{g[0].decode()}_{g[1].decode()}' for g in gerador_qrd.phrasegrams]
    lst = list(gerador_qrd.phrasegrams)
    lst.sort(key= lambda k:len(k), reverse=True)
    lst_vocab_sub.extend(lst)
    print(' - gravando log de Quadrigramas gerados: quadrigramas.log')
    with open(os.path.join(pasta_saida,'quadrigramas.log'),'w') as f:
        f.writelines('\n'.join(lst))
    print(' - número de Quadrigramas gerados:', len(lst))
    print(' - alguns Quadrigramas: ')
    [print(f'   - {_}') for _ in lst[:10]]

    print('Modelos finalizados: ', arquivo_bigrama, arquivo_quadrigrama)

    print(' - gravando vocab de substituição')
    with open(os.path.join(pasta_saida,'VOCAB_TRADUTOR ngramas sugeridos.txt'),'w') as f:
        for linha in lst_vocab_sub:
            de, por = linha.strip().replace('_',' '), linha.strip()
            if len(de)>1 and len(por)>1:
                f.write(f'{de} => {por}\n')

def testar_ngramas(pasta_textos=r'.\textos', pasta_saida_ngramas=None, texto='', maximo_linhas=1000):
    pasta_ngramas = pasta_saida_ngramas if pasta_saida_ngramas else pasta_textos 
    print('Carregando modelos de Bigramas e Quadrigramas:')
    transformador = TransformarNGramas(pasta_ngramas)
    if texto:
       print(' - Aplicanto ngramas em: ', texto)
       tokens = transformador.transformar( pre_processamento(texto) )
       print('-----------------------------------------')
       print(' '.join(tokens))
       print('-----------------------------------------')
    elif pasta_textos:
      iterarq_1 = IterarArquivos(pasta = pasta_textos, mascara = '*.txt')
      q = 0
      inicio = timer()
      q_arq = 0
      for _, conteudo in iterarq_1:
          sents_qdr = [transformador.transformar(tokens) for tokens in conteudo]
          q_arq +=1
          for sent in sents_qdr:
              sent = [_.upper() if _.find('_')>=0 else _ for _ in sent]
              _sent = ' '.join(sent)
              # caso seja para mostrar algumas linhas, mostra as mais relevantes
              if maximo_linhas==0 or _sent.find('_') >=0 and len(_sent)>10:
                print(f'({round(timer()-inicio,3)}s) - ', _sent[:200])
                q += 1
              if maximo_linhas>0 and q==maximo_linhas:
                break
          if q==maximo_linhas:
            break

class TransformarNGramas():
    def __init__(self, pasta_ngramas='./textos'):
        self.pasta_ngramas = pasta_ngramas
        self.arquivo_bigramas = os.path.join(self.pasta_ngramas,'bigramas.bin')
        self.arquivo_quadrigramas = os.path.join(self.pasta_ngramas,'quadrigamas.bin')
        self.gerador_bigramas = None
        self.gerador_quadrigramas = None
        print(f'Iniciando TransformarNGramas("{pasta_ngramas}")')
        if os.path.isfile(self.arquivo_bigramas):
            self.gerador_bigramas = self.carregar_ngramas(self.arquivo_bigramas)
            print(' - modelo de bigramas carregado')
        if os.path.isfile(self.arquivo_quadrigramas):
            self.gerador_bigramas = self.carregar_ngramas(self.arquivo_quadrigramas)
            print(' - modelo de quadrigramas carregado')
        if self.gerador_bigramas is None and self.gerador_quadrigramas is None :
            print(' - ATENÇÃO: modelos de bigramas e quadrigramas não encontrados')

    @classmethod
    def carregar_ngramas(self, arquivo):
        if not os.path.isfile(arquivo):
            raise Exception(str(f'carregar_ngramas: Arquivo {arquivo} não encontrado'))
        gerador = Phraser.load(arquivo)
        return gerador
    
    def transformar(self, tokens):
        if self.gerador_bigramas and self.gerador_quadrigramas:
            return self.gerador_quadrigramas[self.gerador_bigramas[tokens]]
        elif self.gerador_bigramas:
            return self.gerador_bigramas[tokens]
        return tokens 
        
    def transformar_texto(self, texto):
        tokens = texto.split(' ') if type(texto) is str else texto
        tokens = self.transformar(tokens)
        return ' '.join(tokens) if type(texto) is str else tokens


############################################


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Pasta do modelo')
    parser.add_argument('-pasta', help='pasta raiz contendo as pastas do modelo e dos textos', required=False)
    parser.add_argument('-epocas', help='número de épocas para treinamento - padrão 5000 ', required=False, action='store_const', const=5000)
    parser.add_argument('-min_count', help='número de ocorrências para usar um termo - padrão 10 ', required=False)
    parser.add_argument('-threshold', help='threshold para aceitar termos como compostos - padrão 20 ', required=False)
    parser.add_argument('-treino', help='gera a curadoria com os textos de', required=False, action='store_const', const=1)

    args = parser.parse_args()
    min_count = int(args.min_count) if args.min_count else 10
    threshold = int(args.threshold) if args.threshold else 20
    pasta_treino = args.treino

    if not args.pasta:
        print('============================================================================')
        print(f'ATENÇÃO: Pasta base não definida, usando pasta padrão "./meu_modelo"', end='', flush=True)
        sleep(1)
        print(' . ', end='', flush=True)
        print('\n============================================================================')
        sleep(1)
        print('Iniciando...')

    PASTA_BASE = args.pasta if args.pasta else './meu_modelo'

    # espera-se que exista uma pasta padronizada de modelos
    # pasta_modelo
    # pasta_modelo\textos_vocab
    # pasta_modelo\textos_treino
    # pasta_modelo\analise_ngramas (criada automaticamente)
    if not os.path.isdir(PASTA_BASE):
        msg = f'Não foi encontrada a pasta "{PASTA_BASE}" para criação das sugestões'
        raise Exception(msg)

    pasta_saida_ngramas = os.path.join(PASTA_BASE,'analise_ngramas')
    if not os.path.isdir(pasta_saida_ngramas):
        os.makedirs(pasta_saida_ngramas,exist_ok=True)

    # gera os ngramas com os textos do vocab
    pasta_textos = os.path.join(PASTA_BASE,'textos_treino') if pasta_treino else os.path.join(PASTA_BASE,'textos_vocab')
    if not os.path.isdir(pasta_textos):
        pasta_textos = PASTA_BASE
    gerar_bigramas_preprocessados(pasta_textos=pasta_textos, 
                                  pasta_saida_ngramas= pasta_saida_ngramas, 
                                  min_count=min_count, 
                                  threshold=threshold)

    # testa com os textos de treino ou do vocab
    pasta_textos_treino = os.path.join(PASTA_BASE,'textos_treino')
    if os.path.isdir(pasta_textos_treino):
        pasta_textos = pasta_textos_treino
    testar_ngramas(pasta_textos=pasta_textos, 
                   pasta_saida_ngramas = pasta_saida_ngramas, 
                   maximo_linhas=100)

    print('=========================================================================================')
    print(f'nGrams gerados com min_count = {min_count} e threshold = {threshold}')
    print(f'Sugestões e modelos de bigramas e quadrigramas criados em: ', pasta_saida_ngramas)
    print('=========================================================================================')


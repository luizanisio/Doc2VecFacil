# -*- coding: utf-8 -*-
#######################################################################
# Código complementar ao Doc2VecFacil para criar nGramas para o documento VOCAB_TRADUTOR.txt 
# Esse código, dicas de uso e outras informações: 
#   -> https://github.com/luizanisio/Doc2VecFacil/
# Luiz Anísio 
# 21/10/2021 - disponibilizado no GitHub  
#######################################################################

from util_doc2vec_facil import UtilDoc2VecFacil, listar_arquivos, carregar_arquivo, map_thread
import os 

import numpy as np
import pandas as pd
import random
from scipy import spatial
import seaborn as sns
from matplotlib import pyplot as plt
from collections import Counter
from sklearn.manifold import TSNE

CST_TAMANHO_COLUNA_TEXTO = 250

''' 
> agrupar uma lista de vetores
  > retorna um objeto de agrupamento com um dataframe com os dados do agrupamento: grupo,centroide,similaridade,vetor
  util_grupos = UtilAgrupamentoFacil.agrupar_vetores(vetores,sim)
  print(util_grupos.dados)

> agrupar arquivos de uma pasta gerando um excel no final
  > será gerado um aquivo "agrupamento {pasta de textos} sim {similaridade}.xlsx"
  > também retorna o objeto de agrupamento com o dataframe de agrupamento
  > arquivo_saida = None só retorna o datagrame em gerar o arquivo
  > se plotar = True, então vai gerar um arquivo "arquivo_saida.png"
  util_grupos = UtilAgrupamentoFacil.agrupar_arquivos(pasta_modelo, pasta_arquivos, 
                                                      arquivo_saida = '', 
                                                      similaridade = 90,
                                                      plotar = True):
  print(util_grupos.dados)

> usar o objeto
  util_grupos = UtilAgrupamentoFacil(dados=meu_dataframe, similaridade=90)
  print(util_grupos.dados)
'''

def progress_bar(current_value, total, msg=''):
    increments = 25
    percentual = int((current_value / total) * 100)
    i = int(percentual // (100 / increments))
    text = "\r[{0: <{1}}] {2:.2f}%".format('=' * i, increments, percentual)
    print('{} {}           '.format(text, msg), end="\n" if percentual == 100 else "")

class UtilAgrupamentoFacil():

    def __init__(self, dados, similaridade = 90, distancia = 'cosine'):
        if type(dados) in (list, np.array, np.ndarray):
           if any(dados) and type(dados[0]) is not dict:
              # recebeu uma lista de vetores
              dados = [{'vetor': v} for v in dados]
        self.dados = pd.DataFrame(dados) if type(dados) is not pd.DataFrame else dados
        self.similaridade = similaridade if similaridade>1 else int(similaridade*100)
        self.distancia = 'cosine' if distancia.lower() in ('c','cosine') else 'euclidean'
        self.dados['vetor_np'] = [np.array(v) for v in self.dados['vetor']]
        self.dados['grupo'] = [-1 for _ in range(len(self.dados))]
        self.dados['centroide'] = [0 for _ in range(len(self.dados))]
        self.agrupar()
        #self.dados.drop('vetor_np', axis='columns', inplace=True)

    def vec_similaridades(self, vetor, lst_vetores):
        #_v = np.array(vetor) if type(vetor) is list else vetor
        _v = vetor.reshape(1, -1)
        return ( 1-spatial.distance.cdist(lst_vetores, _v, self.distancia).reshape(-1) )  
  
    def plotar(self, show_plot=True, arquivo = None):
      if len(self.dados) ==0:
         return
      # ajusta os x,y
      if not 'x' in self.dados.columns:
         # verifica se tem 2 dimensões
         if len(self.dados['vetor'][0]) >2:
            print(f'Reduzindo dimensões para plotagem de {len(self.dados["vetor"][0])}d para 2d')
            tsne_model = TSNE(n_components=2, init='pca', method='exact', n_iter=1000)
            vetores_2d = tsne_model.fit_transform(list(self.dados['vetor_np']) )
            x,y = zip(*vetores_2d)
         else:
            x,y = zip(*self.dados['vetor_np'])
         self.dados['x'] = x
         self.dados['y'] = y
      if arquivo:
         plt.figure(dpi=300, figsize=(15,15))
      else:
        plt.figure(figsize=(13,13))
      sns.set_theme(style="white")
      grupos = list(set(self.dados['grupo']))
      custom_palette = sns.color_palette("Set3", len(grupos))
      custom_palette ={c:v if c >=0 else 'k' for c,v in zip(grupos,custom_palette)}
      #centroides
      tamanhos = [100 if t==1 else 50 if s==0 else 20 for t,s in zip(self.dados['centroide'],self.dados['similaridade']) ]
      sns_plot = sns.scatterplot( x="x", y="y", data=self.dados, hue='grupo', legend=False,  s = tamanhos, palette=custom_palette)
      if arquivo:
         plt.savefig(f'{arquivo}')
      if not show_plot:
         plt.close()

    def grupos_vetores(self):
        grupos = self.dados[self.dados.centroide == 1]
        vetores = list(grupos['vetor_np'])
        grupos = list(grupos['grupo'])
        return grupos, vetores
  
    def melhor_grupo(self, vetor):
      # busca os grupos e vetores dos centróides
      grupos, vetores = self.grupos_vetores()
      # retorna -1 se não existirem centróides
      if not vetores:
          return -1,0
      # busca a similaridade com os centróides
      sims = list(self.vec_similaridades(vetor,vetores))
      # verifica a maior similaridade
      maxs = max(sims)
      # busca o centróide com maior similaridade
      imaxs = sims.index(maxs) if maxs*100 >= self.similaridade else -1
      # retorna o número do grupo e a similaridade com o melhor centróide
      grupo = grupos[imaxs] if imaxs>=0 else -1
      sim = maxs*100 if imaxs>=0 else 0
      return grupo, sim
  
    def agrupar(self, primeiro=True):
      grupos = self.dados['grupo']
      centroides = self.dados['centroide']
      passo = 'Criando centróides' if primeiro else 'Reorganizando similares'
      for i, (g,c) in enumerate(zip(grupos,centroides)):
          progress_bar(i+1,len(grupos),f'{passo}')
          if g==-1 or c==0:
            v = self.dados.iloc[i]['vetor_np']
            # identifica o melhor centróide para o vetor
            g,s = self.melhor_grupo(v)
            if g >=0:
              self.dados.at[i,'grupo'] = g
              self.dados.at[i,'similaridade'] = s
            else:
              # não tendo um melhor centróide, cria um novo grupo
              g = max(self.dados['grupo']) +1
              self.dados.at[i,'grupo'] = g
              self.dados.at[i,'similaridade'] = 100
              self.dados.at[i,'centroide'] = 1
      if primeiro:
         # um segundo passo é feito para corrigir o centróide de quem ficou ente um grupo e outro
         # buscando o melhor dos centróides dos grupos que poderia pertencer
         self.agrupar(False)
         # corrige os grupos órfãos e renumera os grupos
         self.dados['grupo'] = [f'tmp{_}' for _ in self.dados['grupo']]
         grupos = Counter(self.dados['grupo'])
         #print('Grupos e quantidades: ', list(grupos.items()))
         ngrupo = 1
         for grupo,qtd in grupos.items():
             if qtd==1:
                self.dados.loc[self.dados['grupo'] == grupo, 'similaridade'] = 0
                self.dados.loc[self.dados['grupo'] == grupo, 'centroide'] = 0
                self.dados.loc[self.dados['grupo'] == grupo, 'grupo'] = -1
             else:
                self.dados.loc[self.dados['grupo'] == grupo, 'grupo'] = ngrupo
                ngrupo +=1
         # ordena pelos grupos
         self.dados['tmp_ordem_grupos'] = [g if g>=0 else float('inf') for g in self.dados['grupo']]
         self.dados.sort_values(['tmp_ordem_grupos','similaridade','centroide'], ascending=[True,False, False], inplace=True)
         self.dados.drop('tmp_ordem_grupos', axis='columns', inplace=True)


    @classmethod
    # retorna tuplas com o nome dos arquivos e seus vetores (nome, vetor)
    # os arquivos são ordenados para permitir que os testes sejam menos randômicos.
    # pode-se, por exemplo, nomear os arquivos com a ordem que se espera de agrupamento
    # para avaliar se foram agrupados como desejado
    # coluna_texto - inclui o início do texto no retorno 
    def vetorizar_arquivos(self, pasta_arquivos, pasta_modelo, epocas = 3, coluna_texto = False):
        assert os.path.isdir(pasta_modelo), 'A pasta do modelo não é válida'
        assert os.path.isdir(pasta_arquivos), 'A pasta de arquivos não e válida'
        print(f'\t - carregando lista de arquivos de {pasta_arquivos}')
        lista = listar_arquivos(pasta_arquivos)
        lista.sort()
        modelo = UtilDoc2VecFacil(pasta_modelo=pasta_modelo)
        print(f'\t - vetorizando {len(lista)} arquivos com {epocas} época{"s" if epocas>1 else ""} cada ... ')
        progresso=[0]
        def _vetorizar(i):
            arq = lista[i]
            texto = carregar_arquivo(arq, juntar_linhas=True)
            vetor = modelo.vetor_sentenca(sentenca=texto, epocas=epocas) if texto else None
            # atualiza a lista com o nome do arquivo e o vetor
            if coluna_texto:
                resumo = texto.replace('\n',' | ')
                resumo = f'{resumo[:CST_TAMANHO_COLUNA_TEXTO]} [..]' if len(resumo)>CST_TAMANHO_COLUNA_TEXTO else resumo
                lista[i] = (lista[i], vetor, resumo)
            else:
                lista[i] = (lista[i], vetor)
            if i % 10 ==0:
                progresso[0] = max(progresso[0],i)
                progress_bar(progresso[0],len(lista),f' vetorizando {os.path.split(arq)[-1]}' )
        # vetoriza os arquivos para o agrupamento
        map_thread(_vetorizar, lista = range(len(lista)), n_threads=10)
        progress_bar(1,1,' finalizado ')
        # filtra os arquivos sem vetor / texto
        return [dados for dados in lista if dados[0] and dados[1]]

    # cria um dataframe com os grupos, exporta para o excel (arquivo_saida) e retorna o dataframe
    @classmethod
    def agrupar_vetores(self, vetores, similaridade = 90):
        return UtilAgrupamentoFacil(vetores, similaridade=similaridade)

    # cria um dataframe com os grupos, exporta para o excel (arquivo_saida) e retorna o dataframe
    # textos = True/False - inclui uma parte do texto do documento no dataframe
    @classmethod
    def agrupar_arquivos(self, pasta_modelo, pasta_arquivos, arquivo_saida = '', 
                         similaridade = 90, epocas = 3, plotar=True, coluna_texto = False):
        assert os.path.isdir(pasta_modelo), 'A pasta do modelo não é válida'
        assert os.path.isdir(pasta_arquivos), 'A pasta de arquivos não e válida'
        if not arquivo_saida:
            comp = os.path.split(pasta_arquivos)[-1]
            arquivo_saida = f'./agrupamento {comp} sim {similaridade}.xlsx'
        lista = self.vetorizar_arquivos(pasta_modelo=pasta_modelo, 
                                        epocas = epocas, 
                                        pasta_arquivos=pasta_arquivos, 
                                        coluna_texto=coluna_texto)
        #arquivos, vetores = zip(*lista)
        if coluna_texto:
            _dados = [{'pasta':os.path.split(a)[0], 'arquivo': os.path.splitext(os.path.split(a)[1])[0], 'vetor':v, 'texto':t} 
                    for a,v,t in lista]
        else:
            _dados = [{'pasta':os.path.split(a)[0], 'arquivo': os.path.splitext(os.path.split(a)[1])[0], 'vetor':v} 
                    for a,v in lista]
        util = UtilAgrupamentoFacil(_dados, similaridade=similaridade)
        util.dados['similaridade'] = [round(s,2) for s in util.dados['similaridade']]
        print('\t - construindo planilha de dados')
        print('\t - finalizando arquivo excel')
        colunas = ['pasta','arquivo', 'grupo', 'similaridade','centroide']
        if coluna_texto:
            colunas.append('texto')
        util.dados.to_excel(arquivo_saida,sheet_name=f'Agrupamento de arquivos',
                             index = False, columns=colunas)
        if plotar:
           if arquivo_saida.endswith('.xlsx'):
               arquivo_plot = arquivo_saida.replace('.xlsx','.png')
           else:
               arquivo_plot = f'{arquivo_saida}.png'
           print(f'\t - finalizando arquivo plot {arquivo_plot}') 
           util.plotar(show_plot=False, arquivo= arquivo_plot)

        print('Agrupamento finalizado em: ', arquivo_saida)
        return util
        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Pasta do modelo')
    parser.add_argument('-modelo', help='pasta contendo o modelo - padrao meu_modelo ou doc2vecfacil', required=False)
    parser.add_argument('-textos', help='pasta contendo os textos que serão agrupados - padrao ./textos_treino', required=False)
    parser.add_argument('-sim', help='similaridade padrão 90%', required=False)
    parser.add_argument('-epocas', help='épocas para inferir o vetor padrão 3', required=False)
    parser.add_argument('-plotar', help='plota um gráfico com a visão 2d do agrupamento', required=False, action='store_const', const=1)
    parser.add_argument('-texto', help='inclui uma coluna com parte do texto no resultado', required=False, action='store_const', const=1)

    args = parser.parse_args()

    arq_modelo = 'doc2vec.model'
    similaridade = int(args.sim or 90)
    epocas = int(args.epocas or 3)
    epocas = 1 if epocas<1 else epocas
    plotar = args.plotar
    coluna_texto = args.texto

    PASTA_BASE = args.modelo or './meu_modelo' or './doc2vecfacil'
    PASTA_MODELO = PASTA_BASE
    # se a pasta não tive o modelo dentro, verifica se ele está na subpasta doc2vecfacil
    if not os.path.isfile(os.path.join(PASTA_MODELO,arq_modelo) ):
       if os.path.isfile(os.path.join(PASTA_MODELO,'doc2vecfacil', arq_modelo) ):
           PASTA_MODELO = os.path.join(PASTA_MODELO,'doc2vecfacil')
    
    if not os.path.isfile(os.path.join(PASTA_MODELO,arq_modelo) ):
        print(f'ERRO: pasta do modelo com vocab não encontrada em "{PASTA_MODELO}"')
        exit()

    PASTA_TEXTOS = args.textos 
    if not PASTA_TEXTOS:
       if os.path.isdir(os.path.join(PASTA_BASE,'textos_grupos')):
           PASTA_TEXTOS = os.path.join(PASTA_BASE,'textos_grupos')
       elif os.path.isdir(os.path.join(PASTA_BASE,'textos_treino')):
           PASTA_TEXTOS = os.path.join(PASTA_BASE,'textos_treino')
       elif os.path.isdir(os.path.join(PASTA_BASE,'textos_teste')):
           PASTA_TEXTOS = os.path.join(PASTA_BASE,'textos_teste')
       elif os.path.isdir('./textos'):
           PASTA_TEXTOS = './textos'
    if (not PASTA_TEXTOS) or (not os.path.isdir(PASTA_TEXTOS)):
        print(f'ERRO: pasta de textos não encontrada em "{PASTA_TEXTOS}"')
        exit()

    util = UtilAgrupamentoFacil.agrupar_arquivos(pasta_modelo=PASTA_MODELO, 
                                          pasta_arquivos=PASTA_TEXTOS, 
                                          similaridade=similaridade,
                                          epocas = epocas,
                                          plotar = plotar,
                                          coluna_texto = coluna_texto)




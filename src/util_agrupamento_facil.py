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

from scipy import spatial
import numpy as np
import pandas as pd

''' 
> agrupar uma lista de vetores
  > retorna uma lista de tuplas na mesma ordem com [(grupo,similaridade), ...]
  grupos_sim = UtilAgrupamentoFacil.agrupar_vetores_similares(vetores,sim)

> agrupar arquivos de uma pasta gerando um excel no final
  > será gerado um aquivo "agrupamento {pasta de textos} sim {similaridade}.xlsx"
  UtilAgrupamentoFacil.agrupar_arquivos(pasta_modelo, pasta_arquivos, arquivo_saida = '', similaridade = 90, ver_print=True):

'''

def progress_bar(current_value, total, msg=''):
    increments = 25
    percentual = int((current_value / total) * 100)
    i = int(percentual // (100 / increments))
    text = "\r[{0: <{1}}] {2:.2f}%".format('=' * i, increments, percentual)
    print('{} {}           '.format(text, msg), end="\n" if percentual == 100 else "")

class UtilAgrupamentoFacil():
    #método leve que considera que todos os vetores são numpy arrays e válidos
    @classmethod
    def vec_similaridades(self, vetor, lst_vetores):
        _v = vetor.reshape(1, -1)
        #calculos[0] += 1
        return ( 1-spatial.distance.cdist(lst_vetores, _v, 'cosine').reshape(-1) ) * 100  

    # recebe os vetores em lista ou numpy array para grupamento pela similaridade (0.92 ou 92)
    # retorna uma lista de tuplas (grupo, similaridade)
    # grupo >0 é um grupo válido
    # grupo <=0 é um vetor órfão
    @classmethod
    def agrupar_vetores_similares(self, vetores, similaridade, ver_print=False):
        sim_corte = similaridade*100 if similaridade<=1 else similaridade
        npvetores =  np.array(vetores)
        ngrupos = [0 for _ in range(len(npvetores))]
        similaridades = [0 for _ in range(len(npvetores))]
        itens = [_ for _ in range(len(npvetores))]
        itens_agrupar = [_ for _ in itens]
        grupo_atual = 0
        # vai montando os grupos até não ter mais itens no grupo 0
        qtd_itens = len(itens_agrupar)
        while len(itens_agrupar)>0:
            i_atual = itens_agrupar[0]
            itens_agrupar = itens_agrupar[1:]
            # progresso se não for para printar
            if not ver_print:
                _pos = qtd_itens - len(itens_agrupar)
                progress_bar(_pos, qtd_itens, f'{i_atual} grupo {grupo_atual}')

            # verifica se já foi analisado
            if ngrupos[i_atual] !=0:
                continue
            grupo_atual += 1
            if ver_print:
                print(f'Analisando item {i_atual} grupo atual: ', grupo_atual)
            vetor_atual = npvetores[i_atual]  # pega o vetor para a comparação com os próximos
            ngrupos[i_atual] = -1             # o grupo -1 é o grupo dos órfãos
            _vetores, _itens = [],[]
            # busca os próximos sem grupo
            for _i in itens_agrupar:
                # busca os não analisados ainda
                if ngrupos[_i] != 0:
                    continue
                _vetores.append(npvetores[_i])
                _itens.append(_i)
            if ver_print:
                print('\t - comparando: ', len(_itens), end='')
            # acaba quando não tiver mais itens
            if len(_itens) ==0:
                print('')
                break
            # compara os itens com o primeiro da lista
            sims = self.vec_similaridades(vetor_atual, _vetores)
            qtd = 0
            for s, i in zip(sims, _itens):
                if s>=sim_corte:
                   # indica o grupo do item i 
                   ngrupos[i] = grupo_atual
                   similaridades[i] = round(s)
                   qtd += 1
            # não encontrando semelhantes, coloca como grupo -1
            if qtd>0:
                ngrupos[i_atual] = grupo_atual
                similaridades[i_atual] = 100
            else:
                grupo_atual -= 1 # volta a contagem de grupo pois não foi usado
            if ver_print:
                print('\t - encontrados: ', qtd, flush=True)
            # finaliza a captura dos similares do vetor atual
        return [(g,s) for g,s in zip(ngrupos,similaridades)]
    
    # retorna tuplas com o nome dos arquivos e seus vetores (nome, vetor)
    @classmethod
    def vetorizar_arquivos(self, pasta_arquivos, pasta_modelo, epocas = 3):
        assert os.path.isdir(pasta_modelo), 'A pasta do modelo não é válida'
        assert os.path.isdir(pasta_arquivos), 'A pasta de arquivos não e válida'
        print(f'\t - carregando lista de arquivos de {pasta_arquivos}')
        lista = listar_arquivos(pasta_arquivos)
        modelo = UtilDoc2VecFacil(pasta_modelo=pasta_modelo)
        print(f'\t - vetorizando {len(lista)} arquivos com {epocas} época{"s" if epocas>1 else ""} cada ... ')
        progresso=[0]
        def _vetorizar(i):
            arq = lista[i]
            texto = carregar_arquivo(arq, juntar_linhas=True)
            vetor = modelo.vetor_sentenca(sentenca=texto, epocas=epocas) if texto else None
            # atualiza a lista com o nome do arquivo e o vetor
            lista[i] = (lista[i], vetor)
            if i % 10 ==0:
                progresso[0] = max(progresso[0],i)
                progress_bar(progresso[0],len(lista),f' vetorizando {os.path.split(arq)[-1]}' )
        # vetoriza os arquivos para o agrupamento
        map_thread(_vetorizar, lista = range(len(lista)), n_threads=10)
        progress_bar(1,1,' finalizado ')
        # filtra os arquivos sem conteúdo
        return [(t,v) for t,v in lista if v]

    @classmethod
    def agrupar_arquivos(self, pasta_modelo, pasta_arquivos, arquivo_saida = '', similaridade = 90, epocas = 3, ver_print=True):
        assert os.path.isdir(pasta_modelo), 'A pasta do modelo não é válida'
        assert os.path.isdir(pasta_arquivos), 'A pasta de arquivos não e válida'
        if not arquivo_saida:
            comp = os.path.split(pasta_arquivos)[-1]
            arquivo_saida = f'./agrupamento {comp} sim {similaridade}.xlsx'
        lista = self.vetorizar_arquivos(pasta_modelo=pasta_modelo, 
                                        epocas = epocas, 
                                        pasta_arquivos=pasta_arquivos)
        arquivos, vetores = zip(*lista)
        grupos_sim = self.agrupar_vetores_similares(vetores = vetores,
                                                    similaridade = similaridade, 
                                                    ver_print=ver_print)
        dados = []
        for arquivo, (grupo, sim) in zip(arquivos, grupos_sim):
            arquivo = os.path.split(arquivo)[-1]
            dados.append({'ARQUIVO': arquivo, 'GRUPO' : grupo, 'SIMILARIDADE': sim})
        dados = sorted(dados, key = lambda k:k['GRUPO'] if k['GRUPO']>0 else float('inf'))
        print('\t - construindo planilha de dados')
        dados_saida = pd.DataFrame(dados)
        print('\t - finalizando arquivo excel')
        colunas = ['ARQUIVO', 'GRUPO', 'SIMILARIDADE',]
        dados_saida.to_excel(arquivo_saida,sheet_name=f'Agrupamento de arquivos',
                             index = False, columns=colunas)
        print('Agrupamento finalizado em: ', arquivo_saida)
        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Pasta do modelo')
    parser.add_argument('-modelo', help='pasta contendo o modelo - padrao meu_modelo ou doc2vecfacil', required=False)
    parser.add_argument('-textos', help='pasta contendo os textos que serão agrupados - padrao ./textos_treino', required=False)
    parser.add_argument('-sim', help='similaridade padrão 90%', required=False)
    parser.add_argument('-epocas', help='épocas para inferir o vetor padrão 3', required=False)
    args = parser.parse_args()

    arq_modelo = 'doc2vec.model'
    similaridade = int(args.sim or 90)
    epocas = int(args.epocas or 3)
    epocas = 1 if epocas<1 else epocas

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
       if os.path.isdir(os.path.join(PASTA_BASE,'textos_treino')):
           PASTA_TEXTOS = os.path.join(PASTA_BASE,'textos_treino')
       elif os.path.isdir(os.path.join(PASTA_BASE,'textos_teste')):
           PASTA_TEXTOS = os.path.join(PASTA_BASE,'textos_teste')
       elif os.path.isdir('./textos'):
           PASTA_TEXTOS = './textos'
    if (not PASTA_TEXTOS) or (not os.path.isdir(PASTA_TEXTOS)):
        print(f'ERRO: pasta de textos não encontrada em "{PASTA_TEXTOS}"')
        exit()

    UtilAgrupamentoFacil.agrupar_arquivos(pasta_modelo=PASTA_MODELO, 
                                          pasta_arquivos=PASTA_TEXTOS, 
                                          similaridade=similaridade,
                                          epocas = epocas)

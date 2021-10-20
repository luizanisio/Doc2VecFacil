# -*- coding: utf-8 -*-
#######################################################################
# Código complementar ao Doc2VecFacil para criar traduzir termos simples 
# ou compostos do documento VOCAB_TRADUTOR.txt, foi otimizado para textos
# grandes levando em consideração o case, já que será usado após a
# tokenização e limpeza do documento
# Esse código, dicas de uso e outras informações: 
#   -> https://github.com/luizanisio/Doc2VecFacil/
# Luiz Anísio 
# 09/10/2021 - disponibilizado no GitHub  
#######################################################################

import re

class TradutorTermosRe(dict):
    """ Inspirador em: https://www.oreilly.com/library/view/python-cookbook/0596001673/ch03s15.html
        Consultado em: 08/10/2021
    """
    def __init__(self, termos, ignore_case = True):
        # caso tenha uma lista, é uma lista de remoção de termos
        # caso tenha um dict, é um conjunto de tradução de termos
        # caso tenha uma tupla, é um conjunto de tradução de termos
        if type(termos) is dict:
           self.termos = {c.strip():v for c,v in termos.items() if c.strip()}
        else:
           self.termos = {c.strip():v for c,v in termos if c.strip()}
        flags = re.IGNORECASE if ignore_case else 0
        self.ignore_case = ignore_case
        if self.ignore_case:
            self.termos = {c.lower():v for c,v in self.termos.items()}
        self.re_termos = re.compile(r'\b'+r'\b|\b'.join(map(re.escape, self.termos.keys(  )))+r'\b', flags)

    def __sub_match__(self, match):
        """ será acionado para cada match do regex """
        if self.ignore_case:
            return self.termos[match.group(0).lower()]
        else:
            return self.termos[match.group(0)]

    def sub(self, text):
        """ Aplica o regex e faz a tradução/remoção de cada termo """
        return self.re_termos.sub(self.__sub_match__, text)


class TradutorTermos(dict):
    """ Traduz tuplas de texto - case sensitive 
        Objetivo fazer tradução rápida em grande volume de 
        textos já tokenizados
    """
    def __init__(self, termos):
        # caso tenha uma lista, é uma lista de remoção de termos
        # caso tenha um dict, é um conjunto de tradução de termos
        # caso tenha uma tupla, é um conjunto de tradução de termos
        if type(termos) is dict:
           self.termos = [ (c.strip(),v) for c,v in termos.items() if c.strip() ]
        else:
           self.termos = [ (c.strip(),v) for c,v in termos if c.strip() ]
        self.termos.sort(key= lambda k:len(k[0]), reverse = True )
        # inclui espaço nos termos
        self.termos = [(f' {c} ',f' {v} ') for c,v in self.termos]

    def sub(self, texto):
        """ Aplica a tradução/remoção de cada termo  """
        _texto = f' {texto} '
        for composto, novo in self.termos:
            _texto = _texto.replace(composto, novo)
        return _texto.strip()

def teste_tradutores():
    termos = {'cível' : '(cível)', 'danos':'(danos)', 'ERÁRIO': '(ERÁRIO)','de':''}
    texto    = 'APELAÇÃO CÍVEL AÇÃO QUE OBJETIVA O RESSARCIMENTO DE DANOS CAUSADOS AO ERÁRIO EM FUNÇÃO DA CONCESSÃO ILEGAL E LESIVA DE PERMISSÃO DE USO DE BEM PÚBLICO PELO ENTÃO CHEFE DO PODER EXECUTIVO ESTADUAL EM FAVOR DA SEGUNDA RÉ'
    texto_ci = 'APELAÇÃO (cível) AÇÃO QUE OBJETIVA O RESSARCIMENTO  (danos) CAUSADOS AO (ERÁRIO) EM FUNÇÃO DA CONCESSÃO ILEGAL E LESIVA  PERMISSÃO  USO  BEM PÚBLICO PELO ENTÃO CHEFE DO PODER EXECUTIVO ESTADUAL EM FAVOR DA SEGUNDA RÉ'
    texto_cs = 'APELAÇÃO CÍVEL AÇÃO QUE OBJETIVA O RESSARCIMENTO DE DANOS CAUSADOS AO (ERÁRIO) EM FUNÇÃO DA CONCESSÃO ILEGAL E LESIVA DE PERMISSÃO DE USO DE BEM PÚBLICO PELO ENTÃO CHEFE DO PODER EXECUTIVO ESTADUAL EM FAVOR DA SEGUNDA RÉ'

    for i in range(3):
        if i == 0:
            tipo =  'TradutorTermosRe ic'
            tradutor = TradutorTermosRe(termos, True)
            texto_traduzido = tradutor.sub(texto)
            esperado = texto_ci
        elif i == 1:
            tipo =  'TradutorTermosRe cs'
            tradutor = TradutorTermosRe(termos, False)
            texto_traduzido = tradutor.sub(texto)
            esperado = texto_cs
        elif i == 2:
            tipo =  'TradutorTermos cs'
            tradutor = TradutorTermos(termos)
            texto_traduzido = tradutor.sub(texto)
            esperado = texto_cs

        # tradutor Re ic
        if texto_traduzido == esperado:
            print(f'{tipo} OK')
        else:
            print(f'{tipo} FALHOU:')
            print(' - Esperado : ', esperado)
            print(' - Retornado: ', texto_traduzido)

if __name__ == "__main__":
    teste_tradutores()

# -*- coding: utf-8 -*-

#######################################################################
# Classes: 
# UtilDoc2VecFacil : permite carregar um modelo Doc2Vec treinado para aplicar em documentos processados pelo TokenizadorInteligente
# UtilDoc2VecFacil_Treinamento: permite usar documentos em estrutura pré-definida para treinar um modelo de similaridade semântica
# TokenizadorInteligente: permite carregar configurações de termos e termos/frases que não devem ser tokenizadas para tokenização para treinamento
# Esse código, dicas de uso e outras informações: 
#   -> https://github.com/luizanisio/Doc2VecFacil/
# Luiz Anísio 
# Ver 0.1.0 - 03/10/2021 - disponibilizado no GitHub  
# Ver 0.1.1 - 08/10/2021 - substituição de termos - arquivo vocab_composto_substituir
#                        - a substituição ocorre após a remoção de acentos e da união de siglas
#                        - os números são substituídos depois dos compostos
#######################################################################

from functools import cmp_to_key
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import os
import logging
from timeit import default_timer as timer
from datetime import datetime
from scipy.spatial import distance
from unicodedata import normalize
import re
import json
from nltk.stem.snowball import SnowballStemmer
from util_tradutor_termos import TradutorTermos

STEMMER = SnowballStemmer('portuguese')

# função simples de carga de arquivos que tenta descobrir o tipo de arquivo (utf8, ascii, latin1)
def carregar_arquivo(arq, limpar=False, juntar_linhas=False, retornar_tipo=False):
    tipos = ['utf8', 'ascii', 'latin1']
    linhas = None
    tipo = None
    for tp in tipos:
        try:
            with open(arq, encoding=tp) as f:
                tipo, linhas = (tp, f.read().splitlines())
                break
        except UnicodeError:
            continue
    if not linhas:
        with open(arq, encoding='latin1', errors='ignore') as f:
            tipo, linhas = ('latin1', f.read().splitlines())
    # otimiza os tipos de retorno
    if limpar and juntar_linhas:
        linhas = re.sub('\s+\s*', ' ', ' '.join(linhas))
    elif limpar:
        linhas = [re.sub('\s+\s*', ' ', l) for l in linhas]
    elif juntar_linhas:
        linhas = ' '.join(linhas)
    if retornar_tipo:
        return tipo, linhas
    else:
        return linhas

def listar_arquivos(pasta, extensao='txt', inicio=''):
    if not os.path.isdir(pasta):
        msg = f'Não foi encontrada a pasta "{pasta}" para listar os arquivos "{extensao}"'
        raise Exception(msg)
    res = []
    _inicio = str(inicio).lower()
    _extensao = f".{extensao}".lower() if extensao else ''
    for path, dir_list, file_list in os.walk(pasta):
        for file_name in file_list:
            if (not inicio) and file_name.lower().endswith(f"{_extensao}"):
                res.append(os.path.join(path,file_name))
            elif file_name.lower().endswith(f"{_extensao}") and file_name.lower().startswith(f"{_inicio}"):
                res.append(os.path.join(path,file_name))
    return res

# essa classe recebe uma pasta onde estão as configurações de tokenização
# pasta_vocab: pasta com os arquivos de tokenização
# registrar_oov: 
# tokenizar_tudo: 
class TokenizadorInteligente():
    RE_TOKENIZAR = re.compile(r'[^a-z0-9á-ú\_]')
    REGEX_SIGLAS = re.compile(r"(?<=\W[a-z])\.(?=[a-z]\W)" )
    RE_ESPACOS_QUEBRAS = re.compile(r'(\s|<br>|\\n)+')
    RE_ESPACOS_QUEBRAS_COMPOSTO = re.compile(r'(\s|<br>|\\n|_)+')
    NUMEROS = [_ for _ in ' zero um dois tres quatro cinco seis sete oito nove '.split(' ') if _]

    RE_URL = re.compile(r'\b(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})\b')
    RE_EMAIL = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')

    def __init__(self, pasta_vocab, registrar_oov = False, tokenizar_tudo = False):
        self.pasta_vocab = str(pasta_vocab) if pasta_vocab else '' # pasta vocab vazia indica que o vocab é criado pelo texto
        self.nome_oov = None                    # palavras não localizadas no vocab
        self.nome_oov_quebrados = None          # palavras quebradas e ainda não localizadas
        self.nome_oov_quebrados_tokens = None   # palavras quebradas e ainda não localizadas
        self.nome_ok_quebrados  = None          # palavras quebradas e encontradas - token quebrado por linha
        self.nome_vocab_final  =  None          # palavras quebradas e encontradas
        self.tokenizar_tudo = tokenizar_tudo    # True/False se retorna todos os tokens mesmo fora do vocab
        self.oov = set()                        # palavras não localizadas no vocab
        #self.oov_final = set()
        self.oov_quebrados = set()              # palavras não localizadas mesmo quebradas - token quebrado
        self.oov_quebrados_tokens = set()       # palavras não localizadas mesmo quebradas - token original
        self.ok_quebrados = set()               # palavras localizadas após quebradas - token quebrado prefixo sufixo/oov
        self.registrar_oov = registrar_oov      # registra as listas de oov durante a tokenização (usado na geração do vocab, não usar em treinamento)
        self.vocab_final = set()                # palavras encontradas no vocab
        self.vocab_tradutor_termos = set()      # termos simples ou compostos para substituição ou remoção
        self.tradutor_termos = None             # tradutor de termos compostos
        if registrar_oov and self.pasta_vocab:
            self.nome_oov = os.path.join(self.pasta_vocab,'treinamento - vocab_fora.txt')
            self.nome_oov_quebrados = os.path.join(self.pasta_vocab,'treinamento - vocab_fora_quebrados.txt')
            self.nome_oov_quebrados_tokens = os.path.join(self.pasta_vocab,'treinamento - vocab_fora_quebrados_tokens.txt')
            self.nome_ok_quebrados = os.path.join(self.pasta_vocab,'treinamento - vocab_ok_quebrados.txt')
            self.nome_vocab_final = os.path.join(self.pasta_vocab,'treinamento - vocab_final.txt')
            # recarrega os arquivos se existirem pois ao usar o cache dos textos 
            # não há verificação de tokens oov
            if os.path.isfile(self.nome_oov):
                self.oov = set(carregar_arquivo(self.nome_oov,juntar_linhas=False))
            if os.path.isfile(self.nome_oov_quebrados):
                self.oov_quebrados = set(carregar_arquivo(self.nome_oov_quebrados,juntar_linhas=False))
            if os.path.isfile(self.nome_oov_quebrados_tokens):
                self.oov_quebrados_tokens = set(carregar_arquivo(self.nome_oov_quebrados_tokens,juntar_linhas=False))
            if os.path.isfile(self.nome_ok_quebrados):
                self.ok_quebrados = set(carregar_arquivo(self.nome_ok_quebrados,juntar_linhas=False))
        self.carregar_vocabs()
        if self.pasta_vocab:
            print(f'\t Vocab carregado com {len(self.vocab)} termos', flush=True)
        else:
            print(f'\t Pasta do vocab não definida, todos os termos serão tokenizados', flush=True)

    def carregar_vocabs(self):
        vocab = ''
        vocab_removido = ''
        _termos_traducao = []
        if self.pasta_vocab:
            print(f'\t - identificando arquivos de vocab "{self.pasta_vocab}"')
            for path, dir_list, file_list in os.walk(self.pasta_vocab):
                for file_name in file_list:
                    if file_name.lower().endswith(".txt"):
                        if file_name.lower()[:10]=='vocab_base' or file_name.lower()[:11]=='vocab_extra':
                            pathfile_name = os.path.join(path,file_name)        
                            vocab += (' ' + carregar_arquivo(pathfile_name,limpar=False,juntar_linhas=True))
                            print(f'\t - vocab: {file_name} carregado o/')
                        if file_name.lower()[:14]=='vocab_tradutor':
                            pathfile_name = os.path.join(path,file_name)        
                            _termos_traducao +=  carregar_arquivo(pathfile_name,limpar=False,juntar_linhas=False)
                            print(f'\t - vocab: tradutor de termos {file_name} carregado o/')
                        elif file_name.lower()[:14]=='vocab_removido':
                            pathfile_name = os.path.join(path,file_name)        
                            vocab_removido += (' ' + carregar_arquivo(pathfile_name,limpar=False,juntar_linhas=True))
                            print(f'\t - vocab removido: {file_name} carregado o/')

        self.vocab_vazio = len(vocab) == 0
        # vocab de substituição ou remoção
        self.vocab_tradutor_termos = []
        self.tradutor_termos = None
        _incluidos = []
        _compostos = []
        def _preparar_termo_traducao(_termo, saida: bool):
            # substitui os números pelos nomes deles
            # saída é o termo que vai substituir
            for i, n in enumerate(self.NUMEROS):
                _termo = _termo.replace(str(i),f' {n} ')
            _termo = self.remover_acentos(_termo.lower()).strip()
            # a saída pode ser termo simples ou composto, mas símbolos serão convertidos para _
            if saida:
                _termo = self.RE_TOKENIZAR.sub('_', _termo)
                _termo = self.RE_ESPACOS_QUEBRAS_COMPOSTO.sub('_',_termo ).strip()
            else:
                _termo = self.RE_TOKENIZAR.sub(' ', _termo)
                _termo = self.RE_ESPACOS_QUEBRAS.sub(' ',_termo ).strip()
            _termo = _termo[:-1] if _termo and _termo[-1] == '_' else _termo
            return _termo

        for composto in _termos_traducao:
            # avalia se tem termo para substituição
            pos_divisor = composto.find('=>')
            novo_termo = '' # por padrão remove o termo se não tiver termo de substituição
            if pos_divisor>=0:
               novo_termo = composto[pos_divisor+2:]
               composto = composto[:pos_divisor]
               novo_termo = _preparar_termo_traducao(novo_termo, True)
            # padroniza os compostos igual o tokenizador removendo acentos e juntando siglas
            composto = _preparar_termo_traducao(composto, False)
            if composto and composto not in _incluidos:
                _compostos.append((composto, novo_termo))
                _incluidos.append(composto)
        # criar o tradutor
        self.tradutor_termos = None
        if any(_compostos):
            #_compostos = sorted(_compostos, key = lambda k:len(k[0]), reverse=True)
            for composto, novo_termo in _compostos:
                self.vocab_tradutor_termos.append((composto, novo_termo))
            arq_log_composto = os.path.join(self.pasta_vocab,'vocab_tradutor_termos.log')
            self.tradutor_termos = TradutorTermos(self.vocab_tradutor_termos)
            # grava o arquivo de substituição de termos compostos processado
            # grava se não existir ou se for uma tokenização completa (início de um vocab ou treinamento)
            if (self.tokenizar_tudo or self.registrar_oov or not os.path.isfile(arq_log_composto)):
                with open(arq_log_composto,'w') as f:
                        for composto, novo_termo in self.tradutor_termos.termos:
                            f.write(f'{composto} => {novo_termo}\n')

        # identificação de termos que devem ser retirados do vocab
        vocab_removido = self.remover_acentos( vocab_removido.replace('\n',' ').lower() )
        vocab_removido = {_ for _ in vocab_removido.split(' ') if _}
        # construção do vocab sem os termos removidos
        _txt_numeros = ' '.join(self.NUMEROS)
        vocab += _txt_numeros
        vocab = self.remover_acentos( vocab.replace('\n',' ').lower() )
        vocab = {_ for _ in vocab.split(' ') if _ and (_ not in vocab_removido) }
        if self.vocab_vazio:
            if self.pasta_vocab:
                # o alerta aparece apenas se a pasta foi definida e não foram encontrados os arquivos
                print(f'TokenizadorInteligente: ATENÇÃO ARQUIVOS VOCAB_BASE*.txt ou VOCAB_EXTRA*.txt NÃO ENCONTRADOS', flush=True)
            vocab = set()
            self.vocab_vazio = True
        self.vocab = vocab

    @classmethod
    def remover_acentos(self,txt):
        return normalize('NFKD', txt).encode('ASCII', 'ignore').decode('ASCII')

    # tokeniza a sentença em tokens do vocab e tokens quebrados do vocab
    # retorna todos os tokens caso self.tokenizar_tudo seja True ou não tenha vocab carregado
    # rapido True retorna os tokens após limpeza
    # rapido 'str' retorna strings após limpeza sem split
    def tokenizar(self, sentenca, rapido = False):
        
        #remover links e emails
        sentenca = self.RE_URL.sub(' ', sentenca)
        sentenca = self.RE_EMAIL.sub(' ', sentenca)

        sentenca = self.remover_acentos(sentenca.lower().replace('<br>',' ').replace('\n',' '))
        sentenca = self.REGEX_SIGLAS.sub('', ' ' + sentenca + ' ')

        # substitui os números pelos nomes deles
        for i, n in enumerate(self.NUMEROS):
            sentenca = sentenca.replace(str(i),f' {n} ')

        tokens = self.RE_TOKENIZAR.sub(' ',sentenca)
        if rapido:
            return tokens if rapido=='str' else tokens.split(' ') 

        # realiza a tradução/remoção de termos
        if self.tradutor_termos is not None:
            tokens = self.tradutor_termos.sub(tokens)

        tokens = tokens.replace('_ ',' ').replace(' _',' ') # compostos órfãos
        # tokeniza o resultado final
        tokens = tokens.split(' ')

        # se não tiver vocab base, volta todos os tokens
        res = []
        #print(f'Iniciando quebra de {len(tokens)} tokens')
        for t in tokens:
            if not t:
                continue
            _quebrados = self.quebrar_tokens(t)
            if (self.vocab_vazio or self.tokenizar_tudo) and len(_quebrados) ==0:
               _quebrados = [t]  # sem vocab, o vocab é criado mas os oov são registrados
            elif len(_quebrados)==0:
                 continue
            res += _quebrados
        # remove tokens de um caractere
        res = [_ for _ in res if len(_)>1]
        if self.registrar_oov:
            self.vocab_final.update(res)
        return res if len(res)>0 else ['.'] # evita documentos vazios

    # retorna o token ou o prefixo/sufixo do token que fazem parte do vocab
    def quebrar_tokens(self, token):
        if not token:
            return []
        # está no vocab, é o token completo
        if token in self.vocab: 
            return [token]
        # não tem o token completo no vocab - registra oov
        if self.registrar_oov: 
            self.oov.update({token}) 
        # não está no vocab e tem 1 letra, não tem como quebrar
        if len(token) == 1:  
            if self.registrar_oov: 
                self.oov.update({token})
            return []
        # não está no vocab, quebra o token em:  prefixo #sufixo
        prefixo = STEMMER.stem(token)
        if len(prefixo) < len(token):  
           sufixo = token[len(prefixo):]
        else:
            sufixo = ''
        if prefixo in self.vocab: # tem que ter pelo menos o prefixo no vocab
            if sufixo and (sufixo in self.vocab):
                if self.registrar_oov:
                    self.ok_quebrados.update({f'{prefixo} #{sufixo}'})
                return [f'{prefixo}',f'#{sufixo}']
            else:
                if self.registrar_oov:
                    self.ok_quebrados.update({f'{prefixo} #OOV'})
                return [f'{prefixo}','#OOV']
        # atualiza o oov quebrado
        if self.registrar_oov: 
           self.oov_quebrados.update({f'{prefixo} {sufixo}'.strip()})
           self.oov_quebrados_tokens.update({token}) # não entrou no vocab nem inteiro e nem quebrado
        return []

    @classmethod
    def quebrar_token_simples(self, token):
        if len(token)<=1:
            return token
        prefixo = STEMMER.stem(token)
        if len(prefixo) < len(token):  
           sufixo = token[len(prefixo):]
        else:
            sufixo = ''
        return f'{prefixo} {sufixo}'.strip()
    
    @classmethod
    def retornar_vocab_texto(self, texto):
        # verifica se é texto ou lista
        if type(texto) is str:
           return {self.quebrar_token_simples(_) for _ in texto.split(' ') if _}
        return {_ for _ in texto if _}

    def gravar_oov(self):
        if self.registrar_oov:
            with open(self.nome_oov,'w') as f:
                f.writelines([f'{w}\n' for w in sorted(self.oov)])
            with open(self.nome_oov_quebrados,'w') as f:
                f.writelines([f'{w.replace("_"," ")}\n' for w in sorted(self.oov_quebrados)])
            with open(self.nome_ok_quebrados,'w') as f:
                f.writelines([f'{w}\n' for w in sorted(self.oov_quebrados)])
            with open(self.nome_vocab_final,'w') as f:
                f.writelines([f'{w}\n' for w in sorted(self.vocab_final)])
            print(f'\t Termos vocab_final gravados com {len(self.vocab_final)} termos', flush=True)
            print(f'\t Termos oov gravados com {len(self.oov)} termos', flush=True)
            print(f'\t Termos oov quebrados com {len(self.oov_quebrados)} termos', flush=True)
            print(f'\t Termos ok quebrados com {len(self.ok_quebrados)} termos', flush=True)
    
class Documentos:
    # pasta_textos: é a pasta onde estão os documentos para tokenização
    # ignorar_cache: não gera o cache de documento limpo e nem lê se existir o cache, sempre processa o texto
    # - bom para testes e criação de vocab
    # retornar_tokens: retorna a lista de tokens no lugar de um TaggedDocument para Doc2Vec
    # registrar_oov: grava um arquivo tokens_fora.txt com os tokens não encontrados no vocab
    def __init__(self, pasta_textos, maximo_documentos=None, 
                    pasta_vocab=None, 
                    registrar_oov = False, 
                    retornar_tokens = False,
                    ignorar_cache = False,
                    tokenizar_tudo = False):
        self.current = -1
        self.pasta = pasta_textos
        self.documentos = self.listar_documentos()
        # se a pasta_vocab vier vazia, o vocab vai ser criado pelo texto
        self.tokenizer = TokenizadorInteligente( pasta_vocab=pasta_vocab, 
                                                 registrar_oov=registrar_oov, 
                                                 tokenizar_tudo=tokenizar_tudo)
        self.retornar_tokens = retornar_tokens
        self.ignorar_cache = ignorar_cache
        if maximo_documentos:
           self.documentos=self.documentos[:maximo_documentos]
        self.high = len(self.documentos)
        self.qtd_processados = 0
        self.timer = timer()
        print(f'Documentos: Lista criada com {self.high} documentos')

    def listar_documentos(self):
        documentos = []
        pastas_documentos = [self.pasta] if type(self.pasta) is str else list(self.pasta)
        for pasta_doc in pastas_documentos:
            for path, dir_list, file_list in os.walk(pasta_doc):      
                for file_name in file_list:
                    if file_name.lower().endswith(".txt"):
                        file_name = os.path.join(path,file_name)
                        documentos.append(file_name)
        return documentos

    def __iter__(self):
        return self

    def __next__(self): # Python 2: def next(self)
        self.current += 1
        if self.current < self.high:
            #print('Documento: ',self.current)
            tokens = self.get_tokens_doc(self.documentos[self.current])
            if self.retornar_tokens:
                return tokens
            return TaggedDocument(tokens, [self.current]) 
        print(' o/') # finaliza o progresso de iterações
        raise StopIteration    

    def get_tokens_doc(self, arquivo):
        self.qtd_processados +=1
        # o progresso é quando está ignorando o cache
        def _print_progresso(_cache):
            if self.qtd_processados % 100 == 0:
                print(f'[{self.qtd_processados}|{round(timer()-self.timer)}s]', end='', flush=True)
            elif self.qtd_processados % 10 == 0:
                if _cache:
                   print('-', end='', flush=True)
                else:
                   print('+', end='', flush=True)

        # existe o arquivo limpo - ignorar o cache se solicitado
        if not self.ignorar_cache:
            if os.path.isfile(f'{arquivo}.clr'):
                texto = carregar_arquivo(arq=f'{arquivo}.clr', juntar_linhas=True, limpar=True) 
                if self.ignorar_cache:
                    _print_progresso(True)
                return texto.split(' ')
        # carrega e faz a limpeza do arquivo
        texto = carregar_arquivo(arq=f'{arquivo}', juntar_linhas=True, limpar=True) 
        tokens = self.tokenizer.tokenizar(texto)
        # grava o cache - se ignorou o cache, o novo cache será criado
        with open(f'{arquivo}.clr', 'w',encoding='ISO-8859-1') as writer:
            writer.writelines(' '.join(tokens))
        if self.ignorar_cache:
            _print_progresso(False)
        return tokens

# classe de uso do modelo
# pasta_modelo = pasta com o modelo doc2vec e os arquivos de configuração do tokenizador
# dv = UtilDoc2VecFacil(pasta_modelo)
# print('Similaridade: ', dv.similaridade(texto1, texto2))

class UtilDoc2VecFacil():
    NOME_ARQ_MODELO = "doc2vec.model"
    NOME_ARQ_MODELO_LOG = "doc2vec.log"

    def __init__(self, pasta_modelo) -> None:
        self.pasta_modelo = str(pasta_modelo)
        self.nome_modelo = self.get_nome_arquivo_modelo(self.pasta_modelo)
        self.nome_log = os.path.join(self.pasta_modelo,self.NOME_ARQ_MODELO_LOG)
        self.log_treino = dict({})  
        self.model = None
        self.tokenizer = TokenizadorInteligente(pasta_vocab=pasta_modelo)
        if os.path.isfile(self.nome_modelo):
            self.carregar_modelo()

    @classmethod
    def modelo_existe(self, pasta_modelo):
        nome_modelo = self.get_nome_arquivo_modelo(pasta_modelo=pasta_modelo)
        return os.path.isfile(nome_modelo)

    @classmethod
    def get_nome_arquivo_modelo(self, pasta_modelo):
        return os.path.join(pasta_modelo,self.NOME_ARQ_MODELO)

    @classmethod
    def get_arquivos_modelo(self, pasta_modelo):
        if not self.NOME_ARQ_MODELO:
            raise Exception('Nome do arquivo do modelo doc2vec não definido')
        lista = listar_arquivos(pasta = pasta_modelo, inicio=self.NOME_ARQ_MODELO, extensao='npy')
        lista.append( os.path.join(pasta_modelo,self.NOME_ARQ_MODELO) )
        lista.append( os.path.join(pasta_modelo,self.NOME_ARQ_MODELO_LOG) )
        return [_ for _ in lista if os.path.isfile(_)]

    def carregar_modelo(self):
        self.model = None
        if self.tokenizer.vocab == 0:
            raise Exception('UtilDoc2VecFacil: é necessário existir um ou mais arquivos de vocab na pasta do modelo')
        print('UtilDoc2VecFacil: CARREGANDO MODELO')
        _model = Doc2Vec.load(self.nome_modelo)  
        self.model = _model
        if os.path.isfile(self.nome_log):
            with open(self.nome_log) as f:
                self.log_treino = f.read()
            self.log_treino = json.loads(self.log_treino) if len(self.log_treino)>5 else dict({})
        else:
            self.log_treino = dict({})
        print('\tModelo carregado: ', self.nome_modelo, ' épocas: ', self.log_treino.get('opochs',0), ' termos: ', len(_model.wv))
        print('\tTermos: ', list(_model.wv.key_to_index)[:10],'...')
        print('\tModelo e tokenizador carregados o/')  

    def tokens_sentenca(self, sentenca):
        return self.tokenizer.tokenizar(sentenca)

    def vetor_sentenca(self, sentenca):
        return self.model.infer_vector(self.tokens_sentenca(sentenca))

    def similaridade_vetor(self, vetor1,vetor2):
        return 1- distance.cosine(vetor1,vetor2)

    def similaridade(self, sentenca1,sentenca2):
        vetor1 = self.vetor_sentenca(str(sentenca1))
        vetor2 = self.vetor_sentenca(str(sentenca2))
        return self.similaridade_vetor(vetor1,vetor2)    

    def teste_modelo(self):
        frase1 = 'EXECUÇÃO POR TÍTULO EXTRAJUDICIAL DE HONORÁRIO ADVOCATÍCIO EMBARGOS ADJUDICAÇÃO PENHORAS'
        frase2 = 'EMENTA SEGUROs de VIDA COBRANÇA CUMULADA C PRETENSÃO INDENIZATÓRIA PRESCRIÇÃO RECONHECIDA'
        frase3 = 'ATENDIAM A TESTEMUNHA SEU DEPOIMENTO APESAR DE TRAZER ALGUMAS IMPRECISÕES SOBRE OS FATOS ATENDO-SE OS JURADOS ÀS PROVAS PRODUZIDAS EM PLENÁRIOS'
        frase3 += ' frase removida 1 abcdef 2 frase também removida 3 frase também 4'
        print('Frase1: ', frase1, '\nFrase2: ', frase2, '\n\t - Similaridade: ', self.similaridade(frase1,frase2))
        print('Frase1: ', frase1, '\nFrase3: ', frase3, '\n\t - Similaridade: ', self.similaridade(frase1,frase3))        
        print('\nTokens frase 3: ', self.tokens_sentenca(frase3))

    def teste_termos(self, termos=None):
        if termos is None:
            termos = self.carregar_lista_termos_comparacao()
        q = len(termos)
        termos = self.comparar_termos(termos, retorno_string=True)
        print('=================================================')
        print('= Teste de comparação de termos do modelo')
        print(f'= Número de termos para comparação: {q}')
        print('-------------------------------------------------')
        [print(_) for _ in termos]

    # recebe uma lista de termos e retorna os mais similares >=50%
    # retorna um dict termo: [(termo, sim)] ou uma string para print ou arquivo
    def comparar_termos(self, termos, retorno_string = False):
        if not any(termos):
            return []
        res = {} # {termo: [(termo, % sim)]) exemplo: {'juiz': [('minitro',85),('juiza',71)]}
        for termo in sorted(termos):
            try:
                ms = self.model.wv.most_similar(termo, topn=3)
                ms = [(f'{_[0]}',int(round(_[1]*100))) for _ in ms if _[1]>=0.5]
                res[termo] = ms
            except KeyError:
                continue
        if not retorno_string:
            return res
        # retorna uma lista de strings para print ou arquivo
        res_txt = []
        for termo, similares in res.items():
            if not any(similares):
                continue
            ms = [f' {_[0]} ({_[1]}%) ' for _ in similares]
            ms = [_.ljust(25) for _ in ms]
            ms = ' | '.join(ms)
            res_txt.append(f'{termo.ljust(20)} | {ms}')
        return res_txt

    # carrega uma lista de termos de comparação de um arquivo - em geral é preparado para testes
    def carregar_lista_termos_comparacao(self):
        arq = os.path.join(self.pasta_modelo, 'termos_comparacao_treino.txt')
        lista = []
        if os.path.isfile(arq):
            lista = carregar_arquivo(arq,False,True)
            lista = re.sub(r'\s+',' ',lista).split(' ')
        return [_ for _ in lista if _]

# Classe para treinamento do modelo usando o tokenizador inteligente, o modelo é gravado a cada época
# pasta_modelo = pasta com o modelo doc2vec (ou sem modelo se for novo) e os arquivos de configuração do tokenizador
# pasta_textos = pasta com os arquivos texto que serão usados para treinamento
# workers = número de threads usadas no treinamento
# epocas = número de épocas 
# min_count = número mínimos de termos encontrados nos documentos para o termo ser incluído no vocabulário
# janela_termos = número de termos usados no treinamento do contexto de cada termo
# dimensoes = número de dimensões dos vetores treinados
# comparar_termos = lista de termos para comparações entre termos a cada época treinada
#                   gera o arquivo comparar_termos.txt 
#                   se existir o arquivo termos_comparacao_treino.txt, usa esses termos para a comparação
#
# dvt = UtilDoc2VecFacil_Treinamento(pasta_modelo, pasta_textos)
# dvt.treinar()

class UtilDoc2VecFacil_Treinamento():
    def __init__(self, pasta_modelo, pasta_textos, workers=10, epocas=10, min_count = 5, janela_termos = 10, dimensoes = 200, comparar_termos=None) -> None:
        self.doc2vec = UtilDoc2VecFacil(pasta_modelo=pasta_modelo)
        # facilitadores
        self.nome_modelo = self.doc2vec.nome_modelo
        self.pasta_modelo = self.doc2vec.pasta_modelo
        self.nome_log = self.doc2vec.nome_log
        self.pasta_textos = pasta_textos
        self.nome_log_comparacao = os.path.join(self.pasta_modelo,'comparar_termos.log')
        self.nome_vocab_treino = os.path.join(self.pasta_modelo,'vocab_treino.txt')
        # parâmetros de treino
        self.workers = workers if workers else 10
        self.tempo_epoch = self.doc2vec.log_treino.get('batch_segundos',0)
        self.epocas_treinadas = self.doc2vec.log_treino.get('epochs',0)
        self.n_epocas = epocas
        self.min_count = self.doc2vec.log_treino.get('min_count',min_count) 
        self.janela_termos = self.doc2vec.log_treino.get('window',janela_termos) 
        if self.doc2vec.model:
            self.dimensoes = self.doc2vec.model.vector_size
        else:
            self.dimensoes = dimensoes
        self.comparar_termos = comparar_termos
        self.carregar_lista_termos_comparacao()

    def treinar(self):
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.WARNING)
        #logging.basicConfig(level=logging.ERROR)
        #assert self.doc2vec.model is not None, 'É necessário carregar ou criar um modelo antes de iniciar o treino'
        if os.path.isfile(self.nome_modelo):
            if self.doc2vec.model is None:
                self.doc2vec.carregar_modelo()
            self.epocas_anteriores = self.doc2vec.model.epochs
        else:
            if os.path.isfile(self.nome_vocab_treino):
                os.remove(self.nome_vocab_treino)
            if os.path.isfile(self.nome_log_comparacao):
                os.remove(self.nome_log_comparacao)
            if os.path.isfile(self.nome_log):
                os.remove(self.nome_log)
            self.criar_modelo()
            print(f'Carregando lista de documentos para criação do VOCAB de treino ... ')
            # ignora o cache de pré-processamento na criação do vocab do modelo
            # evita que os documentos tenham sido processados para criação dos vocabs complementares
            documentos_treino = Documentos(pasta_textos = self.pasta_textos, 
                                           pasta_vocab = self.doc2vec.pasta_modelo, 
                                           registrar_oov = True,
                                           ignorar_cache = True)
            inicio_treino = timer()
            print(f'Criando VOCAB de treino e pré-processando {documentos_treino.high} documentos ')
            self.doc2vec.model.build_vocab(documentos_treino, update=False)
            documentos_treino.tokenizer.gravar_oov()
            print('\t - Vocab criado em ', timer() - inicio_treino, ' segundos', flush=True)
            self.epocas_anteriores = 0

        print(f'Carregando lista de documentos para o treino ... ')
        documentos_treino = Documentos(self.pasta_textos, pasta_vocab = self.doc2vec.pasta_modelo)
        print(f'\t - Treinando com {documentos_treino.high} documentos ')
        print('\t - inicio_treino:', datetime.today().strftime('%Y-%m-%d %H:%M:%S%z'), flush=True)
        for i in range(self.n_epocas):
            inicio_treino = timer()
            print(f'\t - Treinando época número ', self.epocas_treinadas+1, flush=True)
            # recria o iterator a cada época
            if not documentos_treino:
                documentos_treino = Documentos(self.pasta_textos, pasta_vocab = self.doc2vec.pasta_modelo)
            self.doc2vec.model.train(documentos_treino, epochs=1, total_examples=documentos_treino.high)
            documentos_treino = None
            self.epocas_treinadas += 1
            fim_treino = timer()
            print( '\t   - treino de 1 época em ', fim_treino-inicio_treino, ' segundos', flush=True)
            # log json do treino
            self.doc2vec.log_treino['alpha'] = self.doc2vec.model.alpha
            self.doc2vec.log_treino['min_alpha'] = self.doc2vec.model.min_alpha
            self.doc2vec.log_treino['size'] = self.doc2vec.model.vector_size
            self.doc2vec.log_treino['window'] = self.doc2vec.model.window
            self.doc2vec.log_treino['corpus_words'] = len(self.doc2vec.model.wv)
            self.doc2vec.log_treino['corpus_docs'] = len(self.doc2vec.model.dv)
            self.doc2vec.log_treino['workers'] = self.workers
            self.doc2vec.log_treino['epochs'] = self.epocas_treinadas
            self.doc2vec.log_treino['epoch_dt'] = datetime.today().strftime('%Y-%m-%d %H:%M:%S%z')
            self.doc2vec.log_treino['batch_segundos'] = fim_treino - inicio_treino
            self.doc2vec.log_treino['min_count'] = self.min_count
            self.doc2vec.log_treino['nome_modelo'] = self.nome_modelo
            self.gravar_modelo()

    # 
    def carregar_lista_termos_comparacao(self):
        lista = self.doc2vec.carregar_lista_termos_comparacao()
        if self.comparar_termos is not None:
           lista =  list(self.comparar_termos) + lista
        self.comparar_termos = list(set(lista))
        self.comparar_termos.sort()

    # grava um log do treino, um arquivo com o vocab de treino e o modelo
    def gravar_modelo(self):
        # gravando o modelo
        print('UtilDoc2VecFacil_Treinamento: GRAVANDO MODELO')
        self.doc2vec.model.save(self.nome_modelo)
        print( '\tModelo treinado e gravado: ', self.nome_modelo, ' épocas: ', self.doc2vec.model.epochs, ' alpha: ', self.doc2vec.model.alpha )
        print(f'\tTermos: {len(self.doc2vec.model.wv)} >> ', list(self.doc2vec.model.wv.key_to_index)[:10],' ... ')                                          
        print( '\tDocumentos: ', len(self.doc2vec.model.dv))  
        with open(self.doc2vec.nome_log, "w") as f:
            f.write(json.dumps(self.doc2vec.log_treino))
        if not os.path.isfile(self.nome_vocab_treino):
            with open(self.nome_vocab_treino, "w") as f:
                [f.write(f'{w}\n') for w in sorted(self.doc2vec.model.wv.key_to_index)] 
        self.log_comparacao()
        print('\tModelo gravado o/')  

    # grava um log de comparação dos termos
    def log_comparacao(self):
        if self.comparar_termos is None:
            return 
        res = self.doc2vec.comparar_termos(self.comparar_termos, retorno_string=True)
        print('\tTermos de comparação: ', len(self.comparar_termos), len(res))
        with open(self.nome_log_comparacao,'w') as f:
            [f.write(f'{c}\n') for c in res ]

    def criar_modelo(self):
        print('UtilDoc2VecFacil_Treinamento: CRIANDO NOVO MODELO')
        dm = 1 # garante a ordem
        hs = 1
        _model = Doc2Vec( 
                    vector_size=self.dimensoes, 
                    window=self.janela_termos, 
                    min_count=self.min_count, 
                    workers=self.workers, 
                    epochs=1,
                    dm=dm, hs = hs)
        print('Novo Modelo iniciado')
        self.doc2vec.model = _model



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Pasta do modelo')
    parser.add_argument('-pasta', help='pasta raiz contendo as pastas do modelo e dos textos - padrao meu_modelo', required=False)
    parser.add_argument('-testar', help='realiza um teste do modelo recriado o arquivo termos_comparacao.log', required=False, action='store_const', const=1)
    parser.add_argument('-treinar', help='inicia ou continua o treinamento', required=False, action='store_const', const=1)
    parser.add_argument('-reiniciar', help='use "-reiniciar sim" >> apaga o modelo e inicia um novo treinamento', required=False)
    parser.add_argument('-epocas', help='número de épocas para treinamento - padrão 5000 ', required=False)
    parser.add_argument('-dimensoes', help='número de dimensões para treinamento - padrão 200 ', required=False)
    parser.add_argument('-workers', help='threads de treinamento - padrão 100 ', required=False)
    args = parser.parse_args()

    PASTA_BASE = args.pasta if args.pasta else './meu_modelo'
    PASTA_MODELO = os.path.join(PASTA_BASE,'doc2vecfacil')
    PASTA_TEXTOS_TREINO = os.path.join(PASTA_BASE,'textos_treino')
    teste_modelo = args.testar
    treinar = args.treinar
    reiniciar = str(args.reiniciar).lower() if args.reiniciar else ''
    
    # sem parâmetros, faz apenas o teste
    if teste_modelo or not (reiniciar or treinar):
       print('############################################################')
       print('# Nenhum parâmetro recebido, iniciando teste do modelo     #') 
       print('# Use -h para verificar os parâmetros disponíveis          #')
       print('#----------------------------------------------------------#')
       if UtilDoc2VecFacil.modelo_existe(PASTA_BASE):
           PASTA_MODELO = PASTA_BASE
           print('- modelo encontrado em :', PASTA_MODELO)
       elif UtilDoc2VecFacil.modelo_existe(PASTA_MODELO):
           print('- modelo encontrado em :', PASTA_MODELO)
       else:
           print(f'- modelo não encontrado em uma das pastas: {PASTA_MODELO} ou {PASTA_BASE}')
           exit()
       dv = UtilDoc2VecFacil(pasta_modelo=PASTA_MODELO)
       dv.teste_modelo()
       dv.teste_termos()
       exit()

    if not os.path.isdir(PASTA_BASE):
        msg = f'Não foi encontrada a pasta base de treinamento do modelo em "{PASTA_BASE}" '
        raise Exception(msg)

    if (not treinar) and (not teste_modelo):
      if reiniciar and UtilDoc2VecFacil.modelo_existe(PASTA_MODELO):
        if reiniciar not in ('sim','reiniciar'):
            print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
            print('>>>> Para reiniciar o modelo utilize -reiniciar sim')
            print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
            print('Reiniciar = ', reiniciar)
            exit()
        print('>>>> REINICIANDO O MODELO: ', PASTA_MODELO)
        lista = UtilDoc2VecFacil.get_arquivos_modelo(PASTA_MODELO)
        for arq in lista:
            if os.path.isfile(arq):
               os.remove(arq)

    # treinamento
    dimensoes = int(args.dimensoes) if args.dimensoes else 200
    epocas = int(args.epocas) if args.epocas else 5000
    workers = int(args.workers) if args.workers else 100
    doc2vecTreina = UtilDoc2VecFacil_Treinamento(pasta_modelo= PASTA_MODELO,
                                                    pasta_textos= PASTA_TEXTOS_TREINO,
                                                    epocas=epocas,
                                                    dimensoes=dimensoes,
                                                    min_count=5,
                                                    workers=workers,
                                                    comparar_termos=None)
    doc2vecTreina.treinar()


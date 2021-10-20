from util_doc2vec_facil import TokenizadorInteligente
import os 

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Pasta do modelo')
    parser.add_argument('-pasta', help='pasta raiz contendo as pastas do modelo e dos textos - padrao meu_modelo', required=False)
    args = parser.parse_args()

    PASTA_BASE = args.pasta if args.pasta else './meu_modelo'
    PASTA_MODELO = os.path.join(PASTA_BASE,'doc2vecfacil')
    if not os.path.isdir(PASTA_MODELO):
        print(f'ERRO: pasta do modelo com vocab não encontrada em {PASTA_MODELO}')
        exit()

    tokenizador = TokenizadorInteligente(pasta_vocab=PASTA_MODELO)
    frases = ['aguas_vivas de aguas-vivas por aguas vivas EXECUÇÃO POR TÍTULO EXTRAJUDICIAL DE HONORÁRIO ADVOCATÍCIO EMBARGOS ADJUDICAÇÃO PENHORAS',
              'EMENTA SEGUROs de VIDA COBRANÇA CUMULADA C PRETENSÃO INDENIZATÓRIA PRESCRIÇÃO RECONHECIDA',
              'ATENDIAM A TESTEMUNHA SEU DEPOIMENTO APESAR DE TRAZER ALGUMAS IMPRECISÕES SOBRE OS FATOS ATENDO-SE OS JURADOS ÀS PROVAS PRODUZIDAS EM PLENÁRIOS']
    for frase in frases:
        print('----------------------------------------')
        print('Frase: ', frase)
        print('Tokens: ', tokenizador.tokenizar(frase))


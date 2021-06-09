INV_TOKEN = '<INV>'
ENT_TOKEN = '<ENT>'
REL_TOKEN = '<REL>'


def inverse_relation(r: str) -> str:
    global INV_TOKEN
    return INV_TOKEN + ' ' + r


if __name__ == '__main__':
    import argparse
    import sentencepiece as spm

    parser = argparse.ArgumentParser()
    parser.add_argument('train_file')
    parser.add_argument('model_name')
    parser.add_argument('vocab_size', type=int)  # e.g., 8000, 16000, 32000
    parser.add_argument('--model-type', default='bpe')  # 'unigram'
    parser.add_argument('--char-cov', default='1.0')  # default='0.99999')
    args = parser.parse_args()

    # --model_type=bpe (default is unigram)
    spm_arguments = '''
    --input={} --vocab_size={} --character_coverage={} --model_type={}
    --input_sentence_size=100000000 --user_defined_symbols={}
    --shuffle_input_sentence=true --hard_vocab_limit=false
    --pad_id=0 --bos_id=1 --eos_id=2 --unk_id=3
    --model_prefix={}
    '''.format(args.train_file, args.vocab_size, args.char_cov,
               args.model_type, ','.join([INV_TOKEN, ENT_TOKEN, REL_TOKEN]),
               args.model_name)

    spm.SentencePieceTrainer.Train(spm_arguments.replace('\n', ' '))

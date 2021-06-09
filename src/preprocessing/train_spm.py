def agendalize(s: str) -> str:
    return '<AGENDA-' + s + '>'


def inverse_relation(r: str) -> str:
    return r + '-INV'


TYPE_REL = "HAS-TYPE"
AGENDA_RELATIONS = [
    "USED-FOR",
    "CONJUNCTION",
    "FEATURE-OF",
    "PART-OF",
    "COMPARE",
    "EVALUATE-FOR",
    "HYPONYM-OF",
    TYPE_REL
]
AGENDA_TYPES = [
    "<task>",
    "<method>",
    "<metric>",
    "<material>",
    "<otherscientificterm>"
]
ENT_TOKEN = '<AGENDA-ENT>'
REL_TOKEN = '<AGENDA-REL>'


if __name__ == '__main__':
    import argparse
    import sentencepiece as spm

    parser = argparse.ArgumentParser()
    parser.add_argument('train_file')
    parser.add_argument('model_name')
    parser.add_argument('vocab_size', type=int)  # e.g., 8000, 16000, 32000
    parser.add_argument('--model-type', default='bpe')  # 'unigram'
    parser.add_argument('--char-cov', default='0.9999')  # 1.0
    args = parser.parse_args()

    spm_arguments = '''
    --input={} --vocab_size={} --character_coverage={} --model_type={}
    --input_sentence_size=100000000
    --shuffle_input_sentence=true --user_defined_symbols={}
    --pad_id=0 --bos_id=1 --eos_id=2 --unk_id=3
    --model_prefix={}
    '''

    special_tokens = []
    special_tokens.extend([agendalize(r) for r in AGENDA_RELATIONS])
    # special_tokens.extend([agendalize(inverse_relation(r))
    #                        for r in AGENDA_RELATIONS])
    special_tokens.extend([agendalize(t) for t in AGENDA_TYPES])
    special_tokens.extend([ENT_TOKEN, REL_TOKEN])
    special_tokens = ','.join(special_tokens)

    spm.SentencePieceTrainer.Train(spm_arguments.format(
        args.train_file, args.vocab_size,
        args.char_cov, args.model_type,
        special_tokens, args.model_name
    ).replace('\n', ' '))

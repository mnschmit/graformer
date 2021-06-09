import argparse
import sentencepiece as spm
from .prepare_webnlg import preprocess_entity, preprocess_relation


def test_equal(result, expected):
    if result != expected:
        print('ERROR: Got', result, 'expected', expected)


def test_ident(fun, *args):
    for arg in args:
        test_equal(fun(arg), arg)


def show_segmentation(sp, s, remove_special=False):
    seg = sp.EncodeAsPieces(s)
    if remove_special:
        seg = seg[1:-1]
    print(seg)


test_sample = {
    "modifiedtripleset": [
        {
            "subject": "Barny_Cakes",
            "property": "protein",
            "object": "1.8 g"
        }
    ],
    "lexicalisations": [
        {
            "lex": "Barny cakes contain 1.8 g of protein."
        },
        {
            "lex": "Barny cakes contain 1.8g of protein."
        }
    ]
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('spm_model')
    args = parser.parse_args()

    triple = (
        test_sample['modifiedtripleset'][0]['subject'],
        test_sample['modifiedtripleset'][0]['property'],
        test_sample['modifiedtripleset'][0]['object']
    )
    test_equal(preprocess_entity(triple[0]), 'Barny Cakes')
    test_ident(preprocess_entity, triple[2])
    test_ident(preprocess_relation, triple[1])

    sp = spm.SentencePieceProcessor()
    sp.Load(args.spm_model)
    sp.SetEncodeExtraOptions('bos:eos')

    for lex in test_sample['lexicalisations']:
        show_segmentation(sp, lex['lex'])
    for elem in triple:
        show_segmentation(sp, preprocess_entity(elem), remove_special=True)

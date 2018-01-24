def word2features(seq):

    features = {
        'bias': 1.0,
        'tag_out': seq[5],
        '-1:tag': seq[4],
        '-2:tag': seq[3],
        '-3:tag': seq[2],
        '-4:tag': seq[1],
        '-5:tag': seq [0]
    }

    return features


def sent2features(sent):
    return [word2features(sent)]

# def sent2labels(sent):
#     return [label for token, postag, label in sent]
#
# def sent2tokens(sent):
#     return [token for token, postag, label in sent]
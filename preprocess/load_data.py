import os
import sklearn_crfsuite
import sklearn_crfsuite.metrics as metrics
import sklearn
from sklearn.metrics import confusion_matrix

rootdir = '/home/hunaif/code/hunaif/data'
tags = []
file_out = open('tags_sequence_data.txt','w')
valid_tags = "aa,b,ba,bk,nn,ny,qy,qw,sd,sv".split(",")


def word2features(seq):

    features = {
        'bias': 1.0,
        # '-1:tag': seq[4],
        # '-2:tag': seq[3],
        '-3:tag': seq[2],
        '-4:tag': seq[1],
        '-5:tag': seq[0]
    }

    return features


def sent2features(sent):
    return [word2features(sent)]

def sent2labels(sent):
    return [sent[len(sent) - 1]]


tags_list = [[]]
batch_len = 3
count = 0
chat_sequence_all = []

print("loading data....\n")
for subdir,dirs,files in os.walk(rootdir):
   for file in files:
       if(file.endswith('.utt')):
           file_path = os.path.join(subdir, file)
           inner_tags_list = []
           with open(file_path) as fp:
                line = fp.readline()
                begin = False
                while line:
                    if(not begin) :
                        if(line.startswith('=====')):
                            begin = True
                            line = fp.readline()
                        else:
                            line = fp.readline()
                            continue

                    if(len(line.strip()) > 1):
                        splitted_line =line.strip().split(" ")
                        tag = splitted_line[0]
                        if(valid_tags.__contains__(tag.strip())):
                            inner_tags_list.append(tag)

                    line = fp.readline()
           tags_list.append(inner_tags_list)


for tag_list in tags_list:
    for i in range(0,(len(tag_list) - batch_len)):
        chat_sequence = tag_list[i:i+batch_len + 1]
        chat_sequence_all.append(chat_sequence)


print("preparing data.....\n")

X_train = [sent2features(s) for s in chat_sequence_all]
y_train = [sent2labels(s) for s in chat_sequence_all]

X_test = [sent2features(s) for s in chat_sequence_all]
y_test = [sent2labels(s) for s in chat_sequence_all]

crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True,
    verbose=True
)

print("starting train..........\n")
crf.fit(X_train, y_train)

print("Following are the classes: \n")
labels = list(crf.classes_)
print(labels)

y_pred = crf.predict(X_test)
print("weighted f1 score......\n")
print(metrics.flat_f1_score(y_test, y_pred,
                      average='weighted', labels=labels))

sorted_labels = sorted(
    labels,
    key=lambda name: (name[1:], name[0])
)

print("class wise distribution.......\n")
print(metrics.flat_classification_report(
    y_test, y_pred, labels=sorted_labels, digits=3
))

file_out.close()
# The code below maps the whole dictionary to our selected vocab.
import re
import gzip
import numpy as np
from six.moves import cPickle
vocab_ny = file('dataset/vocab.nytimes.txt').readlines()
vocab = file('./dictnostops.txt').readlines()
revised_vocab = dict()
for word in vocab:
    word = word.lower()
    word = re.sub(r'[^a-z]', '', word)
    revised_vocab[word] = len(revised_vocab)
    
vocab = revised_vocab
 
import numpy as np
# phi is a map from the whole vocab to our selected vocab.
phi = np.tile(0,len(vocab_ny))
for j in range(len(vocab_ny)):
    word = vocab_ny[j].lower() 
    word = re.sub(r'[^a-z]', '', word)
    if word in vocab:
        phi[j] = vocab[word]
    else:
        phi[j] = -1
        
# The code below splits the large txt file into small pieces.
count_old = 0
MM = 4686
with gzip.open('docword.nytimes.txt.gz','r') as fin:
    itr = 1
    for line in fin:
        triple = line.split()
        if len(triple) == 3:
            if int(triple[0]) % 64 == 1 and count_old != int(triple[0]):
                count_old = int(triple[0])
                # Write the previous two lists to the files.
                if itr > 1:
                    with open('dataset/wordids-%d.p'%(itr-1), 'wb') as f:
                        cPickle.dump(wordids, f)
                    with open('dataset/wordcts-%d.p'%(itr-1), 'wb') as f:
                        cPickle.dump(wordcts, f)
                    if itr == MM:
                        break
                # Create two new lists.
                wordids = [[] for i in range(64)]
                wordcts = [[] for i in range(64)]
                itr = itr + 1
            
            if phi[int(triple[1]) - 1] != -1:
                wordids[int(triple[0]) % 64 - 1] = wordids[int(triple[0]) % 64 - 1] + [phi[int(triple[1]) - 1]]
                wordcts[int(triple[0]) % 64 - 1] = wordcts[int(triple[0]) % 64 - 1] + [int(triple[2])]

                
# Construct the test dataset by combining the last few files after conversion.
import re
import gzip
import numpy as np
from six.moves import cPickle
size = 4
MM = 4686
wordids = []
wordcts = []
for itr in range(MM - size,MM):
    with open('dataset/wordids-%d.p'%itr, 'rb') as f:
        wordids = wordids + cPickle.load(f)
    with open('dataset/wordcts-%d.p'%itr, 'rb') as f:
        wordcts = wordcts + cPickle.load(f)

with open('dataset/wordids-test-small.p', 'wb') as f:
    cPickle.dump(wordids, f)
with open('dataset/wordcts-test-small.p', 'wb') as f:
    cPickle.dump(wordcts, f)  
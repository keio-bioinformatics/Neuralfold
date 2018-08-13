# Neuralfold
## Requirements
<dl>
  <dd>Python 3</dt>
  <dd>Numpy</dt>
  <dd>Chainer 2.0.0 or later</dt>
  <dd>CPLEX(optional)</dt>
</dl>

## Usage
NEURALfold can take FASTA and BPseq formatted RNA sequences as input, then predicts its secondary structure.
```
% python NEURALfold.py test  -h
usage: prediction of RNA secondary structures test [-h] [-bp] [-ip] [-g GAMMA]
                                                   test_file

positional arguments:
  test_file             FASTA or BPseq file for test

optional arguments:
  -h, --help            show this help message and exit
  -bp, --bpseq          use bpseq format
  -ip, --ipknot         predict pseudoknotted secaondary structure
  -g GAMMA, --gamma GAMMA
                        balance between the sensitivity and specificity
                        
% python NEURALfold.py test sample.fa
start testing...
> DS4440
GGAUGGAUGUCUGAGCGGUUGAAAGAGUCGGUCUUGAAAACCGAAGUAUUGAUAGGAAUACCGGGGGUUCGAAUCCCUCUCCAUCCG
((((((((((....((..((.....))(((((.......))))).))....))).........((((.......)))).))))))).
```
If you want to predict pseudoknotted seconrdary structure, use "-ip" option.
```
% python NEURALfold.py test sample.fa -ip
start testing...
> DS4440
GGAUGGAUGUCUGAGCGGUUGAAAGAGUCGGUCUUGAAAACCGAAGUAUUGAUAGGAAUACCGGGGGUUCGAAUCCCUCUCCAUCCG
(((((([((([[..((..(....]][)(((((.]][.[.))))).))....))).........((((].]....))))..)))))).
```

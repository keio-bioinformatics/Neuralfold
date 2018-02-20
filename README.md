# Neuralfold
## Requirements
<dl>
  <dd>Python 3</dt>
  <dd>Numpy</dt>
  <dd>Chainer 2.0.0 or later</dt>
  <dd>CPLEX</dt>
</dl>

## Usage
NEURALfold can take FASTA and BPseq formatted RNA sequences as input, then predicts its secondary structure.
```
% python NEURALfold.py test sample.fa
start testing...
> DS4440
GGAUGGAUGUCUGAGCGGUUGAAAGAGUCGGUCUUGAAAACCGAAGUAUUGAUAGGAAUACCGGGGGUUCGAAUCCCUCUCCAUCCG
((((((((((....((..((.....))(((((.......))))).))....))).........((((.......)))).))))))).
```

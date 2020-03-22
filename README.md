# VecText

VecText is a Python library for Vectorization of text in the TF-IDF representation.

## Usage

You can create the TF-IDF representation from a corpus.
```python

import pandas as pd
from VecText import VecText

corpus = ["dogs love to chase cats",
       "cats talk to other cats about the problem",
       "dogs just want to play"]

VT = VecText(corpus)
TF = VT.TF()
IDF = VT.IDF()
TF_IDF = VT.TF_IDF(TF, IDF)
vocabulary = VT.getVocabulary()
df = pd.DataFrame(TF_IDF, columns=vocabulary)
print(df)
```
Output:
```
     about      cats     chase      dogs  ...     talk      the   to      want
0  0.00000  0.035218  0.095424  0.035218  ...  0.00000  0.00000  0.0  0.000000
1  0.05964  0.044023  0.000000  0.000000  ...  0.05964  0.05964  0.0  0.000000
2  0.00000  0.000000  0.000000  0.035218  ...  0.00000  0.00000  0.0  0.095424
```
Two vectors can be compared using cosine similarity.
```python
print(VecText.cosSim(TF_IDF[0, :],TF_IDF[1, :]))
print(VecText.cosSim(TF_IDF[1, :],TF_IDF[1, :]))
```
Output:
```
0.07674644445238342
0.9999999999999998
```
You can also create a query based on the object's corpus to make a search.
```
query = "cats talk"
qv = VT.query2vec(IDF, query)
print(VecText.cosSim(qv[0,:], TF_IDF[0, :]))
print(VecText.cosSim(qv[0,:], TF_IDF[1, :]))
print(VecText.cosSim(qv[0,:], TF_IDF[2, :]))
```
Output:
```
0.08477023
0.50694124
0.0
```
## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.
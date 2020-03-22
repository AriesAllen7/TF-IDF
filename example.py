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

query = "cats talk"
qv = VT.query2vec(IDF, query)

print(VecText.cosSim(qv[0, :], TF_IDF[0, :]))
print(VecText.cosSim(qv[0, :], TF_IDF[1, :]))
print(VecText.cosSim(qv[0, :], TF_IDF[2, :]))

from tallem import tallem_transform
import pickle as pd
mob = pd.load(open('mobius_band.pickle', 'rb'))
X = mob['data']
f = mob['f_map']
tallem_transform(X, f, D=3)
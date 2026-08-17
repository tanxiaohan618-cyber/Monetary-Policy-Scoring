#!/usr/bin/env python3
"""对人工审核并集与原始Seed使用统一成对排序方向评分。"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
IN=ROOT/'intensity_final_union_input.csv'; PAIRS=ROOT/'rank_pairs.csv'; VEC=ROOT/'sgns.financial.word'; OUT=ROOT/'intensity_final_lexicon.csv'; META=ROOT/'intensity_final_meta.json'
C=1.; EPOCHS=700; LR=.08
def vectors(wanted):
 out={}
 with VEC.open(encoding='utf-8',errors='ignore') as f:
  dim=int(f.readline().split()[1])
  for line in f:
   w,_,rest=line.partition(' ')
   if w not in wanted:continue
   a=np.fromstring(rest,sep=' ',dtype=np.float32);n=np.linalg.norm(a)
   if len(a)==dim and n:out[w]=a/n
 return dim,out
def fit(pairs,vec):
 D=np.vstack([vec[r.strong]-vec[r.weak] for r in pairs.itertuples() if r.strong in vec and r.weak in vec]).astype(float);w=np.zeros(D.shape[1]);lr=LR;reg=1/(C*len(D))
 for _ in range(EPOCHS):
  z=np.clip(D@w,-40,40);w-=lr*(-D.T@(1/(1+np.exp(z)))/len(D)+reg*w);lr*=.998
 return w
def main():
 terms=pd.read_csv(IN,encoding='utf-8-sig').fillna('');pairs=pd.read_csv(PAIRS,encoding='utf-8-sig');wanted=set(terms.term)|set(pairs.strong)|set(pairs.weak);dim,vec=vectors(wanted);w=fit(pairs,vec)
 terms['in_embedding']=terms.term.isin(vec);terms['rank_score']=terms.term.map(lambda x:float(vec[x]@w) if x in vec else np.nan)
 scored=terms.rank_score.dropna();lo=float(scored.min());hi=float(scored.max());terms['intensity_weight_0_5_1']=terms.rank_score.map(lambda s:.5+.5*(s-lo)/(hi-lo) if pd.notna(s) and hi>lo else np.nan)
 seed_mat=np.vstack([vec[x] for x in set(pairs.strong)|set(pairs.weak) if x in vec]);terms['max_seed_cosine']=terms.term.map(lambda x:float(np.max(seed_mat@vec[x])) if x in vec else np.nan);terms['score_status']=terms.in_embedding.map(lambda x:'统一RankSVM已评分' if x else '缺少完整词向量-未评分')
 terms=terms.sort_values(['in_embedding','intensity_weight_0_5_1','term'],ascending=[False,False,True]);terms.to_csv(OUT,index=False,encoding='utf-8-sig')
 ss={x:float(vec[x]@w) for x in set(pairs.strong)|set(pairs.weak) if x in vec};acc=float(np.mean([ss[r.strong]>ss[r.weak] for r in pairs.itertuples() if r.strong in ss and r.weak in ss]));meta={'union_terms':len(terms),'scored_terms':int(terms.in_embedding.sum()),'unscored_terms':int((~terms.in_embedding).sum()),'training_pairs':len(pairs),'pair_accuracy':acc,'vector_file':VEC.name,'vector_dim':dim,'normalization':'0.5 + 0.5*(rank_score-min)/(max-min), based on scored final union','score_range':[float(terms.intensity_weight_0_5_1.min()),float(terms.intensity_weight_0_5_1.max())]};META.write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(meta,ensure_ascii=False,indent=2))
if __name__=='__main__':main()

#!/usr/bin/env python3
"""PBOC程度候选自动发现与成对排序。

候选发现不依赖程度词典：从变化载体周围生成1—3词/1—6字候选，再用
语料分布、搭配纯度和金融词向量筛选。程度词典只提供RankSVM训练锚点。
仅读写本目录的 intensity_* 结果，不访问 strategy。
"""
from __future__ import annotations
import re, json, math, logging
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
LOG=ROOT/'intensity_pipeline.log'
logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',handlers=[logging.FileHandler(LOG,encoding='utf-8'),logging.StreamHandler()])
PARAM={'start_year':2011,'end_year':2025,'min_occurrences':3,'min_doc_freq':2,'min_carriers':2,'min_candidate_chars':2,'max_candidate_chars':6,'C':1.0,'epochs':700,'learning_rate':0.08,'ood_cosine':0.22}
LEXICON=ROOT/'程度副词六级词典_清洗版.csv'
VECTOR_CANDIDATES=[ROOT/'sans.financial.word',ROOT/'sgns.financial.word']
REPORT_DIR=ROOT/'央行沟通交流文本数据（2001-2026）'/'货币政策执行报告TXT'
MEETING_DIR=ROOT/'monetary_policy_meetings'/'raw_txt'
OUT_CAND=ROOT/'intensity_candidates.csv'; OUT_PAIRS=ROOT/'rank_pairs.csv'; OUT_INST=ROOT/'intensity_instances_review.csv'; OUT_JSON=ROOT/'intensity_workbook_data.json'

CARRIERS=sorted(set('上升 下降 增加 减少 扩大 缩小 扩张 收缩 改善 恶化 加快 放缓 上调 下调 提高 降低 增强 减弱 回升 回落 上涨 下跌 增长 下行 上行 攀升 减缓 加速 收窄 扩宽 扩围 压降 投放 释放 收紧 放松 提升 削弱 缩减 增多'.split()),key=len,reverse=True)
CURATED_FORMAL_SEED=set('不大 不甚 微 轻度 一点 些微 或多或少 有些 略 略为 略微 稍 稍为 稍微 稍许 愈加 愈发 更为 更加 越来越 较为 远远 大为 分外 格外 很 尤为 甚 着实 颇 颇为 过于 过分 过度 过热 过猛 万分 十分 十足 异常 极为 极其 极度 极端 无比 非常'.split())
ORAL=set('不丁点儿 一点儿 有点儿 没怎么 聊 好生 怪 挺 蛮 大不了 如斯 这般 那般 何等 多么 要命 要死 贼'.split())
BAD_EXACT=set('同比 环比 分别 同期 月末 年初 年末 上年 上半年 上季度 全年 累计 实际 总体 经济 金融机构 机构 银行 产业 工业 制造业 第三产业 价格 贷款 利率 增速 规模 水平 余额 信贷 投资 需求 能力 结构 弹性 作用 指数 成本 支出 出口 存款 占比 涨幅 继续 持续 保持 通过 出现 趋于'.split())
BAD_SUFFIX=('同比','环比','年初','年末','上年','上季','季度','月份','价格','贷款','利率','余额','增速','规模','水平','结构','能力','作用','指数','经济','投资','需求','市场','政策','资金','货币','信贷')

def clean_text(t):
    t=re.sub(r'=+\s*第?\s*\d+\s*页\s*=+','。',t); t=re.sub(r'https?://\S+|www\.\S+',' ',t)
    return re.sub(r'[\u200b\ufeff\xa0\s]+',' ',t).strip()
def period(name,kind):
    if kind=='例会':
        m=re.search(r'(20\d{2})_Q([1-4])',name); return (int(m.group(1)),int(m.group(2))) if m else None
    q={'第一':1,'第二':2,'第三':3,'第四':4}; m=re.search(r'(20\d{2})年(第一|第二|第三|第四)季度',name)
    return (int(m.group(1)),q[m.group(2)]) if m else None
def load_docs(folder,kind):
    g={}
    for p in folder.glob('*.txt'):
        per=period(p.name,kind)
        if not per or not PARAM['start_year']<=per[0]<=PARAM['end_year']: continue
        try:t=clean_text(p.read_text(encoding='utf-8'))
        except UnicodeDecodeError:t=clean_text(p.read_text(encoding='gb18030',errors='ignore'))
        row={'document_id':f'{kind}-{per[0]}Q{per[1]}','date':f'{per[0]}-Q{per[1]}','text_type':kind,'text':t,'file':p.name}
        if per not in g or len(t)>len(g[per]['text']):g[per]=row
    return list(g.values())

def candidate_ok(x):
    return (PARAM['min_candidate_chars']<=len(x)<=PARAM['max_candidate_chars'] and re.fullmatch(r'[\u4e00-\u9fff]+',x) is not None and x not in BAD_EXACT and not x.endswith(BAD_SUFFIX))
def discover(docs):
    """无词典高召回：对每个载体抽取左侧连续汉字尾部1—6字和常见二段短语。"""
    rows=[]; sent_re=re.compile(r'[^。！？!?；;]+[。！？!?；;]?')
    for d in docs:
        for sm in sent_re.finditer(d['text']):
            sent=sm.group().strip()
            if len(sent)<4:continue
            for carrier in CARRIERS:
                for cm in re.finditer(re.escape(carrier),sent):
                    cs=cm.start(); left=sent[max(0,cs-12):cs]
                    run=re.search(r'([\u4e00-\u9fff]{1,12})$',left)
                    if not run:continue
                    raw=run.group(1); seen=set()
                    for n in range(1,min(PARAM['max_candidate_chars'],len(raw))+1):
                        cand=raw[-n:]
                        if candidate_ok(cand):seen.add(cand)
                    for cand in seen:
                        start=cs-len(cand)
                        rows.append({'candidate':cand,'carrier':carrier,'sentence':sent[:360],'candidate_start':start,'candidate_end':cs,'carrier_start':cs,'carrier_end':cm.end(),'dependency_relation':'automatic_left_suffix_1_6','document_id':d['document_id'],'date':d['date'],'text_type':d['text_type'],'file':d['file']})
    x=pd.DataFrame(rows).drop_duplicates(['document_id','sentence','candidate','carrier','candidate_start'])
    return x

def aggregate(inst,corpus):
    total=Counter()
    for w in set(inst.candidate): total[w]=len(re.findall(re.escape(w),corpus))
    out=[]
    for cand,g in inst.groupby('candidate'):
        near=len(g); docs=g.document_id.nunique(); cars=g.carrier.nunique(); purity=near/max(total[cand],1)
        # 兼顾频率、跨文档、跨载体、载体邻近纯度；不把此分数当程度强弱。
        recall_score=.25*math.log1p(near)+.25*math.log1p(docs)+.20*math.log1p(cars)+.30*min(purity,1)
        rep=g.sort_values(['text_type','date']).iloc[0]
        out.append({'candidate':cand,'total_count':total[cand],'near_carrier_count':near,'document_frequency':docs,'carrier_count':cars,'carrier_proximity_purity':purity,'discovery_score':recall_score,'carriers':'；'.join(sorted(g.carrier.unique())),'representative_sentence':rep.sentence,'source_document_id':rep.document_id,'source_date':rep.date,'source_text_type':rep.text_type,'source_file':rep.file})
    return pd.DataFrame(out)

def read_seeds(corpus):
    df=pd.read_csv(LEXICON,encoding='utf-8-sig'); wc=next(c for c in ['词语','词','程度词'] if c in df); lc=next(c for c in ['等级序号(1弱-6强)','等级','level'] if c in df)
    x=df[[wc,lc]].rename(columns={wc:'word',lc:'level'}).dropna(); x.word=x.word.astype(str).str.strip(); x.level=pd.to_numeric(x.level,errors='coerce')
    return x[x.level.between(1,6)&x.word.map(lambda w:w in corpus)&x.word.isin(CURATED_FORMAL_SEED)&~x.word.isin(ORAL)].drop_duplicates('word')
def load_vectors(wanted):
    vp=next((p for p in VECTOR_CANDIDATES if p.exists()),None)
    if not vp:raise FileNotFoundError('未找到sans.financial.word或sgns.financial.word')
    vec={}
    with vp.open(encoding='utf-8',errors='ignore') as f:
        h=f.readline().split(); dim=int(h[1])
        for line in f:
            w,_,rest=line.partition(' ')
            if w not in wanted:continue
            a=np.fromstring(rest,sep=' ',dtype=np.float32); n=np.linalg.norm(a)
            if a.size==dim and n>0:vec[w]=a/n
    return vp,dim,vec
def make_pairs(seeds,vec):
    s=seeds[seeds.word.isin(vec)]; rows=[]
    for a in s.itertuples():
        for b in s.itertuples():
            if a.level>b.level:rows.append({'strong':a.word,'strong_level':int(a.level),'weak':b.word,'weak_level':int(b.level),'level_gap':int(a.level-b.level),'constraint':'strong>weak','pair_status':'train'})
    return pd.DataFrame(rows),s
def fit_rank(pairs,vec):
    """线性软间隔成对排序：最小化log(1+exp(-w·d))+L2；正反样本等价对称加入。"""
    D=np.vstack([vec[r.strong]-vec[r.weak] for r in pairs.itertuples()]).astype(np.float64)
    w=np.zeros(D.shape[1]); lr=PARAM['learning_rate']; reg=1/max(PARAM['C']*len(D),1)
    for epoch in range(PARAM['epochs']):
        z=np.clip(D@w,-40,40); grad=-(D.T@(1/(1+np.exp(z))))/len(D)+reg*w
        w-=lr*grad; lr*=0.998
    return w
def validate(seeds,pairs,vec,w):
    scores={x:float(vec[x]@w) for x in seeds.word}; acc=float(np.mean([scores[r.strong]>scores[r.weak] for r in pairs.itertuples()]))
    # 无scipy的Spearman：等级与分数各自平均秩后求相关。
    a=pd.Series([scores[x.word] for x in seeds.itertuples()]).rank().to_numpy(); b=pd.Series([x.level for x in seeds.itertuples()]).rank().to_numpy(); rho=float(np.corrcoef(a,b)[0,1])
    return acc,rho,scores
def calibrate(score,seed_scores):
    lo=np.quantile(list(seed_scores.values()),.10); hi=np.quantile(list(seed_scores.values()),.90)
    z=np.clip((score-lo)/(hi-lo),.01,.99) if hi>lo else .5
    return float(.3+.5*z)

def main():
    logging.info('只使用textmining；候选发现忽略程度词典，词典仅训练锚点')
    docs=load_docs(REPORT_DIR,'执行报告')+load_docs(MEETING_DIR,'例会'); corpus='\n'.join(d['text'] for d in docs)
    inst=discover(docs); agg=aggregate(inst,corpus); seeds0=read_seeds(corpus)
    wanted=set(agg.candidate)|set(seeds0.word); vp,dim,vec=load_vectors(wanted); pairs,seeds=make_pairs(seeds0,vec)
    if len(seeds)<6 or pairs.empty:raise RuntimeError('有效训练Seed或排序对不足')
    w=fit_rank(pairs,vec); pair_acc,rho,seed_scores=validate(seeds,pairs,vec,w); seed_mat=np.vstack([vec[x] for x in seeds.word])
    c=agg[(agg.near_carrier_count>=PARAM['min_occurrences'])&(agg.document_frequency>=PARAM['min_doc_freq'])&(agg.carrier_count>=PARAM['min_carriers'])].copy()
    c['in_embedding']=c.candidate.isin(vec); c=c[c.in_embedding&~c.candidate.isin(set(seeds.word))].copy()
    c['rank_score']=c.candidate.map(lambda x:float(vec[x]@w)); c['normalized_intensity']=c.rank_score.map(lambda x:calibrate(x,seed_scores))
    c['max_seed_cosine']=c.candidate.map(lambda x:float(np.max(seed_mat@vec[x]))); c['ood_flag']=c.max_seed_cosine.map(lambda x:'人工复核-分布外' if x<PARAM['ood_cosine'] else '分布内')
    c['manual_review']='待审核'; c['review_decision']=''; c['review_note']=''; c['seed_or_candidate']='candidate_auto_discovery'
    c=c.sort_values(['discovery_score','rank_score'],ascending=False); c.to_csv(OUT_CAND,index=False,encoding='utf-8-sig'); pairs.to_csv(OUT_PAIRS,index=False,encoding='utf-8-sig')
    # 实例表只保留最终候选对应实例，控制文件大小并便于复核。
    inst[inst.candidate.isin(set(c.candidate))].to_csv(OUT_INST,index=False,encoding='utf-8-sig')
    meta={'parameters':PARAM,'vector_file':vp.name,'vector_dim':dim,'documents':len(docs),'raw_instances':len(inst),'raw_candidates':len(agg),'embedding_candidates':len(c),'training_seeds':len(seeds),'rank_pairs':len(pairs),'pair_accuracy':pair_acc,'spearman':rho,'candidate_discovery':'carrier-context suffix/ngram; intensity lexicon not used','score_formula':'rank_score=w·L2_embedding; normalized_intensity anchored to seed P10/P90 and clipped to 0.305—0.795'}
    OUT_JSON.write_text(json.dumps({'meta':meta,'candidates':c.replace({np.nan:None}).to_dict('records'),'pairs':pairs.to_dict('records')},ensure_ascii=False,indent=2),encoding='utf-8')
    logging.info('完成: docs=%d raw_candidates=%d embedding_candidates=%d pair_acc=%.3f rho=%.3f',len(docs),len(agg),len(c),pair_acc,rho); print(json.dumps(meta,ensure_ascii=False,indent=2))
if __name__=='__main__':main()

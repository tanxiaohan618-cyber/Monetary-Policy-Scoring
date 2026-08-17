#!/usr/bin/env node
/** 从两份人工审核Excel读取当前非空行，生成最终并集输入。 */
import fs from 'node:fs/promises';
import {createRequire} from 'node:module';
const require=createRequire('/Users/xiaohan/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/runtime.cjs');
const {FileBlob,SpreadsheetFile}=require('@oai/artifact-tool');
const root='/Users/xiaohan/Desktop/project/textmining';
const files=[
  [`${root}/程度词提取与RankSVM_人工审核.xlsx`,'人工程度候选'],
  [`${root}/程度候选_自动发现_人工审核.xlsx`,'自动发现审核']
];
const terms=new Map();
for(const [path,label] of files){
  const wb=await SpreadsheetFile.importXlsx(await FileBlob.load(path));const sh=wb.worksheets.getItemAt(0);const v=sh.getUsedRange().values;const h=v[0].map(String);const ci=h.indexOf('candidate');
  for(const r of v.slice(1)){const term=String(r[ci]??'').trim();if(!term)continue;const obj=Object.fromEntries(h.map((x,i)=>[x,r[i]??'']));
    if(!terms.has(term))terms.set(term,{term,sources:new Set(),representative_sentence:'',source_file:'',usage_rule:'',resource_class:'',intensity_dimension:''});
    const x=terms.get(term);x.sources.add(label);for(const k of ['representative_sentence','source_file','usage_rule','resource_class','intensity_dimension'])if(!x[k]&&obj[k])x[k]=String(obj[k]);
  }
}
// 原始训练Seed以实际Rank训练词对为准。
const pwb=await SpreadsheetFile.importXlsx(await FileBlob.load(files[0][0]));const pv=pwb.worksheets.getItem('02RankSVM训练词对').getUsedRange().values;const ph=pv[0].map(String),si=ph.indexOf('strong'),wi=ph.indexOf('weak'),sli=ph.indexOf('strong_level'),wli=ph.indexOf('weak_level');const levels=new Map();
for(const r of pv.slice(1)){for(const [i,li] of [[si,sli],[wi,wli]]){const t=String(r[i]??'').trim();if(t)levels.set(t,Number(r[li]));}}
for(const [term,level] of levels){if(!terms.has(term))terms.set(term,{term,sources:new Set(),representative_sentence:'',source_file:'',usage_rule:'',resource_class:'训练Seed',intensity_dimension:'rank_anchor'});const x=terms.get(term);x.sources.add('原始RankSVM Seed');x.seed_level=level;}
const header=['term','source_union','is_training_seed','seed_level','representative_sentence','source_file','usage_rule','resource_class','intensity_dimension'];
const esc=x=>{x=String(x??'');return /[",\n]/.test(x)?`"${x.replaceAll('"','""')}"`:x};
const rows=[header.join(',')];for(const x of [...terms.values()].sort((a,b)=>a.term.localeCompare(b.term,'zh-CN'))){rows.push([x.term,[...x.sources].join('；'),levels.has(x.term),x.seed_level??'',x.representative_sentence,x.source_file,x.usage_rule,x.resource_class,x.intensity_dimension].map(esc).join(','));}
await fs.writeFile(`${root}/intensity_final_union_input.csv`,'\uFEFF'+rows.join('\n'),'utf8');console.log(JSON.stringify({review_terms:[...terms.values()].filter(x=>![...x.sources].every(s=>s==='原始RankSVM Seed')).length,seeds:levels.size,union:terms.size},null,2));

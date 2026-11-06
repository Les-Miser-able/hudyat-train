'use strict';

const token = document.querySelector('meta[name="studio-token"]').content;
const app = document.getElementById('app'), modal = document.getElementById('modal');
const esc = value => String(value ?? '').replace(/[&<>"']/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const num = value => Number(value||0).toLocaleString('en-US');
const pct = value => typeof value==='number' && Number.isFinite(value) ? (value*100).toFixed(1)+'%' : '—';
const selected = (a,b) => a===b?' selected':'';
const pageNames = {overview:'Overview',dataset:'Dataset',extract:'Extract',train:'Train',test:'Test models',results:'Results',export:'Export',settings:'Settings'};
const state = {
  page:pageNames[location.hash.slice(1)]?location.hash.slice(1):'overview',category:'alphabet',version:'sequence',
  protocol:'random',epochs:150,seed:42,augmentation:5,mirroring:'all',development:false,chart:'accuracy',
  model:'',camera:0,mode:'automatic',target:'',testTab:'automatic',profile:'baseline',candidate:'',simulation_settings:'',
  output:'',allow_development:false,query:'',runFilter:'all',resultId:'',resultTab:'summary',graph:'',
  logOpen:false,tools:{},toolsDirty:false,classes:[],revision:0,sequence:0,jobId:null,offline:false,exiting:false
};
let data, resultData, toastTimer, frameReader=false, frameUrl='', refreshPending=false;
let runtime={job:{status:'idle',kind:''},live:{},simulator_url:''}, logLines=[];
const busy = ()=>['running','stopping'].includes(runtime.job.status);
const disabled = ()=>busy()||state.offline?' disabled':'';
const trainingRuns = ()=>data.records.filter(r=>['training','training_fold'].includes(r.kind));
const currentCategory = ()=>data.categories.find(c=>c.id===state.category)||{id:'all',name:'All categories',signs:data.categories.reduce((a,c)=>a+c.signs,0)};
const currentRecord = ()=>data.records.find(r=>r.id===state.resultId)||data.records.find(r=>r.path===state.model)||null;
const artifact = (id,file)=>'/artifacts/'+encodeURIComponent(id)+'/'+file.split('/').map(encodeURIComponent).join('/');
const pill = (text,good=true)=>`<span class="pill ${good?'good':'warn'}">${good?'<i class="dot"></i>':''}${esc(text)}</span>`;
const field = (label,html)=>`<label class="field">${label}${html}</label>`;
const input = (name,type='text',extra='')=>`<input type="${type}" data-field="${name}" value="${esc(state[name])}" ${extra}>`;
const pathInput = (name,label,kind='directory')=>field(label,`<span class="path-input">${input(name)}<button type="button" class="btn" data-browse="${name}" data-kind="${kind}" aria-label="Browse ${esc(label)}">${icon('folder')}</button></span>`);
const jobButton = (action,label,primary=true)=>`<button type="button" class="btn ${primary?'primary':''}" data-job="${action}"${disabled()}>${icon(action==='train'?'play':action==='live'?'test':action==='extract'?'extract':action==='export'?'export':'refresh')}${label}</button>`;

async function api(path,body) {
  const response=await fetch(path,body===undefined?{cache:'no-store'}:{method:'POST',headers:{'Content-Type':'application/json','X-Studio-Token':token},body:JSON.stringify(body)});
  const value=await response.json();
  if(!response.ok)throw Error(value.error||'The trainer could not complete this request.');
  return value;
}
function notify(message) {
  const toast=document.getElementById('toast');toast.textContent=message;toast.classList.add('show');
  clearTimeout(toastTimer);toastTimer=setTimeout(()=>toast.classList.remove('show'),4500);
}
function showModal(title,body,actions='') {
  modal.innerHTML=`<div class="modal-header"><h2 id="modal-title">${esc(title)}</h2><button type="button" class="icon-btn" data-action="close-modal" aria-label="Close dialog">${icon('close')}</button></div>${body}${actions?`<div class="modal-actions">${actions}</div>`:''}`;
  if(!modal.open)modal.showModal();
}
function go(page){state.page=page;location.hash=page;render();window.scrollTo({top:0});}
async function refreshData(){
  const query=new URLSearchParams({version:state.version,category:state.category});
  if(state.page==='results'&&state.resultId)query.set('run',state.resultId);
  data=await api('/api/state?'+query);
  if(!state.toolsDirty)state.tools={...data.tools};
  state.revision=data.revision;
  if(!state.output)state.output=data.paths.exports_dir+'/'+state.version;
  if(document.activeElement?.matches('input,select,textarea')||modal.open)refreshPending=true;
  else{render();refreshPending=false;}
}
function categorySelect(scope=false){return `<select aria-label="${scope?'Workspace category':'Category'}" data-field="category"${disabled()}><option value="all"${selected(state.category,'all')}>All categories</option>${data.categories.map(c=>`<option value="${c.id}"${selected(state.category,c.id)}>${esc(c.name)}</option>`).join('')}</select>`;}
function versionSelect(scope=false){return `<select aria-label="${scope?'Workspace input version':'Input version'}" data-field="version"${disabled()}><option value="sequence"${selected(state.version,'sequence')}>${scope?'Sequence':'Sequence · static + movement'}</option><option value="static"${selected(state.version,'static')}>${scope?'Static':'Static · held poses'}</option></select>`;}
function navRail(){return `<aside class="sidebar">${brand()}<nav class="nav-list" aria-label="Trainer navigation">${Object.entries(pageNames).filter(([p])=>p!=='settings').map(([page,label])=>`<button class="nav-item ${state.page===page?'active':''}" data-page="${page}" data-tooltip="${label}" aria-label="${label}"${state.page===page?' aria-current="page"':''}>${icon(page==='overview'?'home':page)}</button>`).join('')}</nav><div class="sidebar-bottom"><button class="nav-item ${state.page==='settings'?'active':''}" data-page="settings" data-tooltip="Settings" aria-label="Settings">${icon('settings')}</button><button class="nav-item" data-action="exit" data-tooltip="Exit trainer" aria-label="Exit trainer">${icon('stop')}</button></div></aside>`;}
function topbar(){return `<header class="topbar"><div class="workspace-label">${icon('folder')}<span>Workspace</span><b class="slash">/</b><span>HUDYAT Train</span></div><div class="top-actions"><div class="workspace-scopes">${categorySelect(true)}${versionSelect(true)}</div><span class="local-status"><i class="dot"></i>${state.offline?'Disconnected':busy()?esc(runtime.job.kind)+' running':'Local environment'}</span><button class="icon-btn" data-action="job-log" aria-label="Open job log">${icon('bell')}${runtime.job.status==='failed'?'<i class="notification"></i>':''}</button><div class="avatar" aria-label="Local workspace">HT</div></div></header>`;}
function pageHeading(title,subtitle,eyebrow='YOUR WORKSPACE'){return `<div class="page-heading"><div><div class="eyebrow">${eyebrow}</div><h1>${title}</h1><p>${subtitle}</p></div><div class="row"><button class="btn" data-page="results">${icon('results')} View reports</button><button class="btn primary" data-action="configure"${disabled()}>${icon('plus')} New training run</button></div></div>`;}
function stats(){
  const signers=new Set(data.scan?.coverage.flatMap(c=>c.signers)||[]),target=data.categories.reduce((a,c)=>a+c.target,0);
  const completed=trainingRuns().find(r=>r.kind==='training'&&r.status==='complete'&&r.variant===state.version&&(state.category==='all'||r.category===state.category));
  const values=[['sign','Signs in vocabulary',data.categories.reduce((a,c)=>a+c.signs,0),'9 categories'],['dataset','Raw sources · last scan',data.scan?num(data.scan.sources):'—','of '+num(target)],['signers','Recorded signers',data.scan?String(signers.size).padStart(2,'0'):'—',data.expected_signers.at(0)+'–'+data.expected_signers.at(-1)],['results','Latest test accuracy',pct(completed?.metrics.accuracy),completed?`${completed.protocol} · ${completed.development?'dev':'study'}`:'No completed run']];
  return `<div class="stats">${values.map(([i,label,value,hint])=>`<div class="stat"><div class="stat-symbol">${icon(i)}</div><div><div class="stat-label">${label}</div><div class="stat-value">${esc(value)}<small>${esc(hint)}</small></div></div></div>`).join('')}</div>`;
}
function chartSvg(loss=false,history=data.history){
  const valid=a=>Array.isArray(a)?a.map(v=>typeof v==='number'&&Number.isFinite(v)?v:null):[];
  const train=valid(history[loss?'loss':'accuracy']),val=valid(history[loss?'val_loss':'val_accuracy']);
  const count=Math.max(train.length,val.length),max=loss?Math.max(.01,...train.filter(v=>v!==null),...val.filter(v=>v!==null))*1.05:1;
  const path=a=>a.map((v,i)=>v===null?'':`${i&&a[i-1]!==null?'L':'M'}${(4+i/Math.max(1,count-1)*588).toFixed(1)},${(191-Math.max(0,Math.min(1,v/max))*168).toFixed(1)}`).join(' ');
  return `<svg class="chart" viewBox="0 0 644 225" role="img" aria-label="Recorded training and validation ${loss?'loss':'accuracy'}"><defs><linearGradient id="chart-fill" x1="0" y1="0" x2="0" y2="1"><stop stop-color="#ed8137" stop-opacity=".4"/><stop offset="1" stop-color="#ed8137" stop-opacity="0"/></linearGradient></defs>${[22,65,107,149,191].map((y,i)=>`<path d="M4 ${y}H598" stroke="#ffffff09" stroke-dasharray="3 5"/><text x="609" y="${y+4}">${loss?(max*(1-i/4)).toFixed(2):[100,75,50,25,0][i]+'%'}</text>`).join('')}${val.length>1&&val.every(v=>v!==null)?`<path d="${path(val)} L592 191 L4 191Z" fill="url(#chart-fill)"/>`:''}${count?`<path d="${path(train)}" fill="none" stroke="#6ce6ac" stroke-width="1.6"/><path d="${path(val)}" fill="none" stroke="#ef9b60" stroke-width="1.8"/>${Array.from({length:Math.min(6,count)},(_,i)=>{const n=Math.round(i/Math.max(1,Math.min(6,count)-1)*(count-1));return `<text x="${4+n/Math.max(1,count-1)*588}" y="216">${n===0?'Epoch 1':n+1}</text>`;}).join('')}`:'<text x="300" y="110" text-anchor="middle">Learning curves appear after the first training epoch.</text>'}</svg>`;
}
function chartPanel(){
  const last=(data.history[state.chart==='loss'?'val_loss':'val_accuracy']||[]).at(-1),r=data.history_record;
  const score=state.chart==='accuracy'?pct(last):typeof last==='number'?last.toFixed(4):'—';
  return `<section class="panel chart-panel"><div class="panel-head"><div><h2>Model performance</h2><div class="chart-summary"><div class="chart-number">${esc(score)}</div><div class="chart-improvement">${r?esc(r.protocol?.toUpperCase())+' · '+(r.development?'Development':'Study'):'Ready for your first run'}</div></div><div class="chart-subtitle">Validation ${state.chart} · ${esc(currentCategory().name)} / ${state.version}${r?.kind==='training_fold'?' · '+esc(r.path.split(/[\\/]/).at(-1)):''}</div></div><div class="segmented" aria-label="Chart metric">${['accuracy','loss'].map(key=>`<button class="${state.chart===key?'active':''}" data-chart="${key}" aria-pressed="${state.chart===key}">${key==='accuracy'?'Accuracy':'Loss'}</button>`).join('')}</div></div><div class="chart-wrap">${chartSvg(state.chart==='loss')}</div><div class="chart-footer"><span>Conv1D–BiLSTM <span class="mono"> / 32 frames</span></span><div class="legend"><span><i class="mint"></i>Training</span><span><i></i>Validation</span></div></div></section>`;
}
function featureCard(){return `<section class="feature-card"><h2>Make every<br>sign count.</h2><p>From a hand movement<br>to a model that understands.</p><div class="feature-hand">${hand()}</div><button class="btn light" data-page="test">Open recognition lab ${icon('arrow')}</button></section>`;}
function coveragePanel(){
  const total=data.categories.reduce((a,c)=>a+c.target,0),sources=data.scan?.sources||0;
  return `<section class="panel coverage-panel"><div class="panel-head"><div><h2>Dataset coverage</h2><div class="subtext">Raw sources · before extraction checks</div></div><button class="text-btn" data-page="dataset">Explore ${icon('arrow')}</button></div><div class="bars">${data.categories.slice(0,6).map((c,i)=>{const ratio=c.sources/Math.max(1,c.target);return `<div class="bar-group"><span class="bar-value">${data.scan?Math.round(ratio*100)+'%':'—'}</span><div class="bar ${c.id===state.category?'highlight':''}" style="height:${Math.min(1,ratio)*89}px" title="${esc(c.name)}: ${c.sources} of ${c.target} raw sources"></div><span class="bar-label">${['Alpha','Num','Greet','Cal','Shape','Color'][i]}</span></div>`;}).join('')}</div><div class="coverage-note"><span><span class="green">${data.scan?num(sources):'Not scanned'}</span> / ${num(total)} target sources</span><span>${data.scan?(sources/Math.max(1,total)*100).toFixed(1)+'%':'—'}</span></div></section>`;
}
function runCard(){
  const r=data.history_record,progress=runtime.job.progress?.match(/^Epoch (\d+)\/(\d+)/),ratio=progress?Number(progress[1])/Number(progress[2])*100:0;
  return `<section class="run-card"><div class="row between run-card-head"><span>Training workspace</span><span class="tag">LOCAL</span></div><div class="model-stack"><div class="model-tab">HUDYAT / ${state.version.toUpperCase()}</div><div class="model-tab second">Conv1D–BiLSTM · ${state.protocol==='loso'?'7 signer folds':'random split'}</div><div class="model-face"><div class="row between"><div class="eyebrow">YOUR MODEL</div>${icon('train')}</div><div class="model-title">${esc(currentCategory().name)}</div><div class="model-kind">${state.version==='sequence'?'Static + moving signs':'Held-pose comparison'} · ${currentCategory().signs} classes</div><div class="model-metrics"><div><div class="subtext" style="font-size:9px">Validation accuracy</div><div class="model-score">${pct((data.history.val_accuracy||[]).at(-1))}</div></div>${r?pill(r.development?'Development':r.status,r.status==='complete'):pill('Not trained',false)}</div></div></div><div class="run-progress"><div class="row between"><span>${busy()?esc(runtime.job.progress||runtime.job.kind+' in progress'):'Ready for your next experiment'}</span><span>${busy()?esc(runtime.job.status):'Local'}</span></div><div class="progress-track"><div class="progress-fill" style="width:${ratio}%"></div></div></div><div class="run-buttons"><button class="btn" data-action="configure"${disabled()}>${icon('settings')}Configure</button>${busy()?'<button class="btn primary" data-action="stop">'+icon('stop')+'Stop job</button>':jobButton('train','Start training')}</div><div class="run-card-bottom">${icon('shield')}<span>Local processing · original media retained</span></div></section>`;
}
function recentExperiments(){
  const recent=data.records.filter(r=>r.kind==='training').slice(0,3);
  return `<section class="recent-panel"><div class="panel-head"><h2>Recent experiments</h2><button class="text-btn" data-page="results">View all ${icon('arrow')}</button></div>${recent.length?recent.map(r=>`<button class="recent-item" data-run="${esc(r.id)}"><span class="recent-icon">${icon('train')}</span><span><span class="recent-name">${esc(data.categories.find(c=>c.id===r.category)?.name||r.category)} <span class="muted">/ ${esc(r.variant)}</span></span><span class="recent-meta" style="display:block">${esc(r.protocol?.toUpperCase())} · ${r.development?'Development':'Study'}</span></span><span class="recent-score">${pct(r.metrics.accuracy)}<small>${esc(r.status)}</small></span></button>`).join(''):'<div class="empty">No experiments yet.<br>Your saved runs will appear here.</div>'}</section>`;
}
function overview(){return `${pageHeading('Your training, in motion.','A clearer view of your signs, models, and experiments.')}${stats()}<div class="dashboard"><div class="dashboard-main">${chartPanel()}<div class="under-chart">${featureCard()}${coveragePanel()}</div></div><aside class="dashboard-aside">${runCard()}${recentExperiments()}</aside></div>`;}

function datasetPage(){
  const category=currentCategory(),coverage=category.coverage;
  const cells=coverage?.signs.map(sign=>`<tr><td>${esc(sign.label)}</td>${data.expected_signers.map(s=>`<td class="${sign.counts[s]>=data.takes_per_signer?'green':'muted'}">${sign.counts[s]||0}</td>`).join('')}</tr>`).join('');
  return `${pageHeading('Every recording starts here.','Explore your local recordings across nine sign categories.','DATASET / RAW RECORDINGS')}
    <div class="row between section-tools"><span class="subtext">${data.scan?`${num(data.scan.sources)} eligible files · ${data.scan.issues.length} errors · ${data.scan.excluded.length} excluded`:'No saved scan for this input version.'}</span><div class="row">${jobButton('scan','Scan dataset')}<button class="btn" data-open="${esc(data.paths.raw_dir)}">${icon('folder')}Open raw folder</button></div></div>
    <div class="category-grid">${data.categories.map(c=>`<button class="category-card ${state.category===c.id?'selected':''}" data-category="${c.id}" aria-pressed="${state.category===c.id}"><div class="row between"><h3>${esc(c.name)}</h3>${icon(c.sources>=c.target?'check':'sign')}</div><div class="count">${num(c.sources)} <small>/ ${num(c.target)}</small></div><div class="progress-track"><div class="progress-fill" style="width:${Math.min(100,c.sources/Math.max(1,c.target)*100)}%;background:${c.sources>=c.target?'var(--green)':'var(--orange)'}"></div></div><div class="subtext">${c.signs} signs · raw source target</div></button>`).join('')}</div>
    <section class="panel section-gap"><div class="panel-head"><div><h2>${esc(category.name)} · sign-by-signer coverage</h2><p class="subtext">Target: ${data.takes_per_signer} independent takes per sign, per signer. Raw counts precede extraction quality checks.</p></div></div>${coverage?`<div class="table-wrap"><table><thead><tr><th>Sign</th>${data.expected_signers.map(s=>`<th>${esc(s)}</th>`).join('')}</tr></thead><tbody>${cells}</tbody></table></div>`:'<div class="empty">Scan this version, then select a category to inspect its recordings.</div>'}</section>
    ${data.scan&&(data.scan.issues.length||data.scan.excluded.length)?`<details class="panel section-gap"><summary>Scan errors and excluded sources</summary><pre class="log">${esc(JSON.stringify({issues:data.scan.issues,excluded:data.scan.excluded},null,2))}</pre></details>`:''}`;
}
function extractPage(){return `${pageHeading('Turn movement into meaning.','Prepare model inputs from your original media.','LANDMARK EXTRACTION')}<div class="two-column"><section class="panel"><h2>Extraction setup</h2><div class="form-grid section-gap">${field('Category',categorySelect())}${field('Input version',versionSelect())}</div><div class="pipeline">${[['folder','Original media','Read in place'],['sign','Hand landmarks','80% detection gate'],['train','Model inputs',state.category==='family'?'32 × 141 features':'32 × 128 features']].map(([i,t,s],n)=>`${n?icon('arrow'):''}<div class="pipeline-step">${icon(i)}${t}<small>${s}</small></div>`).join('')}</div><div class="notice">${icon('shield')}<span>Quality gates use the trimmed sign window. Two-hand signs require both hands; Family includes face-location features. Static inputs repeat a stable pose and exclude J/Z. Cached detections are reused, and original media stays in place.</span></div><div class="row">${jobButton('extract','Extract model inputs')}<button class="btn" data-open="${esc(data.paths.extracted_dir)}">${icon('folder')}Open samples</button></div></section><aside>${runCard()}</aside></div>`;}
function trainingFields(){return `<div class="form-grid">${field('Category',categorySelect())}${field('Input version',versionSelect())}${field('Evaluation protocol',`<select aria-label="Evaluation protocol" data-field="protocol"${disabled()}><option value="random"${selected(state.protocol,'random')}>Random · 70 / 15 / 15</option><option value="loso"${selected(state.protocol,'loso')}>LOSO · unseen signers</option></select>`)}${field('Maximum epochs',input('epochs','number','min="1" max="150" step="1" aria-label="Maximum epochs" required'))}${field('Random seed',input('seed','number','min="0" max="4294967295" step="1" aria-label="Random seed" required'))}</div><div class="divider"></div><div class="row between"><h3>Data augmentation</h3>${pill('Training split only')}</div><div class="form-grid section-gap">${field('Augmented copies per original',input('augmentation','number','min="0" max="100" step="1" aria-label="Augmented copies per original" required'))}${field('Left / right mirroring',`<select aria-label="Left / right mirroring" data-field="mirroring"${disabled()}><option value="all"${selected(state.mirroring,'all')}>All signs</option><option value="approved"${selected(state.mirroring,'approved')}>Approved signs only</option><option value="off"${selected(state.mirroring,'off')}>Off</option></select>`)}</div><p class="subtext section-gap">Zero copies disables augmentation. The study uses five copies per original: noise, scale, rotation, timing changes and mirrored samples. Validation and test sources stay unmodified.</p><div class="divider"></div><label class="check"><input type="checkbox" data-field="development"${state.development?' checked':''}${disabled()}>Development run</label><p class="subtext indent-note">Required for incomplete study coverage, fewer epochs, or non-default augmentation.</p>`;}
function trainingPage(){return `${pageHeading('A new experiment. A new possibility.','Configure the same Conv1D–BiLSTM pipeline used by the trainer.','TRAIN / CONFIGURE')}<div class="two-column"><form class="panel" data-form="training">${trainingFields()}<div class="notice">${icon('shield')}<span>${state.protocol==='loso'?'Each signer is held out once, with separate training and validation signers.':'Stratified 70 / 15 / 15 split. Augmentation happens after splitting.'} Study runs require every expected class and ${data.takes_per_signer} accepted takes for each configured signer.</span></div>${jobButton('train','Start training')}</form><aside class="stack">${runCard()}<section class="panel"><h3>Study configuration</h3>${[['Optimizer','Adam · 0.001'],['Batch size','32'],['Early stopping','Patience 20'],['Architecture','Conv1D–BiLSTM'],['Model input',state.category==='family'?'32 × 141':'32 × 128']].map(([k,v])=>`<div class="setting-line"><span>${k}</span>${v}</div>`).join('')}</section></aside></div>`;}
function modelPicker(){return `<div class="model-picker"><label class="field">Saved model<select aria-label="Saved model" data-field="model"${disabled()}><option value="">Choose a run or paste a path below</option>${trainingRuns().map(r=>`<option value="${esc(r.path)}"${selected(state.model,r.path)}>${esc(r.category)} / ${esc(r.variant)} · ${esc(r.protocol)} · ${esc(r.path.split(/[\\/]/).at(-1))}${r.development?' · DEV':''}</option>`).join('')}</select></label>${pathInput('model','Model / training run')}</div>`;}
function expectedSign(){return `<select aria-label="Expected sign" data-field="target"><option value="">(unscored)</option>${state.classes.map(c=>`<option value="${esc(c)}"${selected(state.target,c)}>${esc(c)}</option>`).join('')}</select>`;}
function testPage(){
  const ready=runtime.live.ready&&runtime.job.kind==='live',manual=ready&&state.mode==='manual';
  const automatic=`<section class="panel"><h2>Automated held-out testing</h2><p class="subtext section-gap">Replay the saved test split with its original classes, checksums and preprocessing. A LOSO parent replays every fold. Warm-up is excluded from model timing.</p><div class="notice">${icon('shield')}<span>Reports include accuracy, precision, recall, macro F1, rejection counts and model latency. Python timing does not establish HUDYAT browser response time.</span></div>${jobButton('test','Run automated test')}</section>`;
  const live=`<div class="two-column"><section class="panel"><div class="panel-head"><h2>Live recognition</h2>${pill(ready?'Camera running':'Camera stopped',ready)}</div><div class="camera-stage"><img id="camera-feed" alt="Live camera preview with hand landmarks"${frameUrl?' src="'+frameUrl+'"':''}${frameUrl?'':' hidden'}><div id="camera-placeholder"${frameUrl?' hidden':''}>${hand()}<p class="subtext">Start the camera to see live recognition.</p></div></div><p id="live-diagnostics" class="subtext section-gap"></p><div class="row section-gap">${jobButton('live','Start camera')}${runtime.job.kind==='live'&&busy()?'<button class="btn" data-action="stop">'+icon('stop')+'Stop & save reports</button>':''}</div></section><aside class="panel"><h2>Capture a sign</h2><div class="form-grid section-gap">${field('Camera index',input('camera','number','min="0" max="100" step="1"'))}${field('Capture mode',`<select aria-label="Capture mode" data-field="mode"><option value="automatic"${selected(state.mode,'automatic')}>Automatic sign window</option><option value="manual"${selected(state.mode,'manual')}>Manual capture</option></select>`)}</div><label class="field section-gap">Expected sign${expectedSign()}</label><div class="prediction"><strong id="live-prediction">—</strong><p id="live-feedback">Choose a model and start the camera.</p></div><div class="row"><button class="btn primary" data-live="capture"${manual?'':' disabled'}>Start capture (Space)</button><button class="btn" data-live="finish"${manual?'':' disabled'}>Finish & evaluate (Enter)</button></div><div id="live-top3" class="section-gap"></div><div class="notice">${icon('shield')}<span>Live estimates are exploratory. Each completed capture records one attempt. Camera images are previewed only; they are not saved in reports.</span></div></aside></div>`;
  const simulation=`<section class="panel"><h2>HUDYAT camera simulator</h2><p class="subtext section-gap">Test this model with HUDYAT’s actual browser camera, Recognition lab, preprocessing, and decision rules.</p><div class="form-grid section-gap">${field('Initial profile',`<select aria-label="Initial profile" data-field="profile"><option value="baseline"${selected(state.profile,'baseline')}>Baseline</option><option value="enhanced"${selected(state.profile,'enhanced')}>Enhanced</option></select>`)}${field('Expected sign',expectedSign())}${pathInput('candidate','Alphabet candidate (optional)')}${pathInput('simulation_settings','Settings JSON (optional)','json')}</div><div class="notice">${icon('shield')}<span>Use a completed random-split run or an exported model version. Set the HUDYAT source folder in Settings and run setup-simulator.cmd once. Browser trials and settings configurations are recorded separately from held-out tests.</span></div><div class="row">${jobButton('simulate','Start HUDYAT simulator')}${runtime.simulator_url?`<a class="btn primary" href="${esc(runtime.simulator_url)}" target="_blank" rel="noopener">Open recognition lab ${icon('arrow')}</a>`:''}</div></section>`;
  return `${pageHeading('Give your model a hand.','Evaluate held-out sources, capture signs live, or use HUDYAT’s browser pipeline.','TEST MODELS / RECOGNITION LAB')}<section class="panel">${modelPicker()}</section><nav class="test-tabs segmented section-gap" aria-label="Model testing modes">${[['automatic','Automated held-out'],['live','Live camera / manual'],['simulation','Simulate HUDYAT']].map(([key,label])=>`<button class="${state.testTab===key?'active':''}" data-test-tab="${key}">${label}</button>`).join('')}</nav><div class="section-gap">${state.testTab==='automatic'?automatic:state.testTab==='live'?live:simulation}</div>`;
}
function resultRows(){
  const records=data.records.filter(r=>(state.runFilter==='all'||r.protocol===state.runFilter)&&(`${r.category} ${r.variant} ${r.path}`.toLowerCase().includes(state.query.toLowerCase())));
  return records.length?records.map(r=>`<tr data-run="${esc(r.id)}" tabindex="0" aria-label="Inspect ${esc(r.path.split(/[\\/]/).at(-1))}" class="${r.id===state.resultId?'selected-row':''}"><td>${esc(r.path.split(/[\\/]/).at(-1))}<div class="subtext">${esc(r.kind)}</div></td><td>${esc(r.category)}<div class="subtext">${esc(r.variant)}</div></td><td>${esc(r.protocol||'—')}</td><td>${pill(r.status,r.status==='complete')}</td><td class="green">${pct(r.metrics.accuracy)}</td><td>${pct(r.metrics.macro_f1)}</td><td>${r.development?pill('Development',false):'Study'}</td><td><button class="icon-btn" data-run="${esc(r.id)}" aria-label="Open experiment ${esc(r.id)}">${icon('chevron')}</button></td></tr>`).join(''):'<tr><td colspan="8" class="empty">No recorded experiments match this view.</td></tr>';
}
function table(keys,rows){return `<div class="table-wrap"><table><thead><tr>${keys.map(k=>`<th>${esc(k.replaceAll('_',' '))}</th>`).join('')}</tr></thead><tbody>${rows.length?rows.map(row=>`<tr>${keys.map(k=>`<td>${esc(typeof row[k]==='object'?JSON.stringify(row[k]):row[k])}</td>`).join('')}</tr>`).join(''):`<tr><td colspan="${keys.length}" class="empty">No recorded rows.</td></tr>`}</tbody></table></div>`;}
function resultsDetail(){
  if(!resultData||resultData.record.id!==state.resultId)return '';
  const r=resultData.record,metrics=r.metrics;
  const tabs=[['summary','Summary'],['classes','Per class'],['attempts','Attempts'],['graphs','Graphs']];
  let content='';
  if(state.resultTab==='summary'){
    const pairs=Object.entries(r).filter(([k])=>!['metrics','parameters'].includes(k));
    pairs.push(...Object.entries(metrics).map(([k,v])=>['metric.'+k,v]),...Object.entries(r.parameters||{}).map(([k,v])=>['parameter.'+k,v]));
    content=table(['name','value'],pairs.map(([name,value])=>({name,value})));
  }else if(state.resultTab==='classes')content=table(['label','support','precision','recall','f1'],(metrics.per_class||[]).map(row=>({...row,precision:pct(row.precision),recall:pct(row.recall),f1:pct(row.f1)})));
  else if(state.resultTab==='attempts')content=table(Object.keys(resultData.attempts[0]||{expected:'',predicted:'',outcome:''}),resultData.attempts)+(resultData.truncated?'<p class="subtext section-gap">Showing the first 500 attempts. The CSV contains every row.</p>':'');
  else{
    const graphs=resultData.files.filter(name=>name.endsWith('.png'));
    if(!graphs.includes(state.graph))state.graph=graphs[0]||'';
    content=graphs.length?`<label class="field">Graph<select aria-label="Graph" data-field="graph">${graphs.map(name=>`<option${selected(state.graph,name)}>${esc(name)}</option>`).join('')}</select></label><a href="${artifact(r.id,state.graph)}" target="_blank" rel="noopener"><img class="report-image" src="${artifact(r.id,state.graph)}" alt="${esc(state.graph)}"></a>`:'<div class="empty">Graphs are generated when a run or session finishes.</div>';
  }
  return `<section class="panel section-gap"><div class="panel-head"><div><h2>${esc(r.category)} / ${esc(r.variant)}</h2><p class="subtext">${esc(r.path)}</p></div><div class="row"><button class="btn" data-action="use-model">${icon('test')}Use model for testing</button><button class="btn" data-action="export-model">${icon('export')}Use for export</button></div></div><nav class="segmented" aria-label="Result details">${tabs.map(([key,label])=>`<button class="${state.resultTab===key?'active':''}" data-result-tab="${key}">${label}</button>`).join('')}</nav><div class="section-gap">${content}</div><div class="artifact-links section-gap">${resultData.files.filter(name=>name.endsWith('.csv')||name.endsWith('.html')).map(name=>`<a class="btn small" href="${artifact(r.id,name)}" target="_blank" rel="noopener">${esc(name)}</a>`).join('')}</div></section>`;
}
function resultsPage(){return `${pageHeading('Small experiments. Clearer answers.','Inspect the recorded results behind each model.','RESULTS / EXPERIMENT HISTORY')}<section class="panel"><div class="panel-head"><h2>Recorded experiments <span class="muted">${data.records.length}</span></h2><div class="row"><button class="btn small" data-action="refresh-results">${icon('refresh')}Refresh tables</button>${jobButton('results','Build older reports',false)}</div></div><div class="runroom-toolbar"><input aria-label="Search experiments" data-field="query" value="${esc(state.query)}" placeholder="Search experiments…"><select aria-label="Filter protocol" data-field="runFilter"><option value="all"${selected(state.runFilter,'all')}>All protocols</option><option value="random"${selected(state.runFilter,'random')}>Random split</option><option value="loso"${selected(state.runFilter,'loso')}>LOSO</option></select></div><div class="table-wrap"><table><thead><tr><th>Run / session</th><th>Model</th><th>Protocol</th><th>Status</th><th>Accuracy</th><th>Macro F1</th><th>Run type</th><th><span class="visually-hidden">Inspect</span></th></tr></thead><tbody id="experiment-rows">${resultRows()}</tbody></table></div></section>${resultsDetail()}`;}
function exportPage(){
  const r=data.records.find(r=>r.path===state.model),loso=r?.protocol==='loso';
  return `${pageHeading('Ready for the next chapter.','Stage a browser-compatible bundle, then verify it against HUDYAT.','EXPORT / HUDYAT')}<div class="two-column"><section class="panel">${modelPicker()}<div class="section-gap">${pathInput('output','Bundle directory')}</div><label class="check section-gap"><input type="checkbox" data-field="allow_development"${state.allow_development?' checked':''}${disabled()}>Allow export of a development model for testing</label><div class="bundle-tree">${esc(state.output)}\n├─ catalog.json\n├─ models/&lt;category&gt;/&lt;version&gt;/\n│  ├─ model.json\n│  ├─ weights.bin\n│  └─ parity.json\n└─ shared detector assets</div><div class="notice">${icon('shield')}<span>${loso?'LOSO checkpoints are evaluation artifacts. Select a completed random-split run for export.':'Export stages local files and preserves other catalog categories. It does not register a model in HUDYAT’s database. Parity verifies compatibility; recognition accuracy and device response time need separate testing.'}</span></div><div class="row wrap">${loso?'<button class="btn primary" disabled>Export model bundle</button>':jobButton('export','Export model bundle')}${jobButton('verify','Verify against HUDYAT',false)}<button class="btn" data-open="${esc(state.output)}">${icon('folder')}Open bundle folder</button>${jobButton('audit','Check lesson compatibility',false)}</div></section><aside>${runCard()}</aside></div>`;
}
function settingsPage(){
  const fields=[['extraction_python','Extraction Python','python'],['training_python','Train & test Python','python'],['hudyat_path','HUDYAT project folder','directory'],['tfjs_path','TF.js package (optional)','directory']];
  return `${pageHeading('A workspace that feels like yours.','Configure the local tools used by the trainer’s workers.','SETTINGS / LOCAL ENVIRONMENT')}<section class="panel"><h2>Tool locations</h2><p class="subtext section-gap">Blank interpreter fields use the Python environment that launched this trainer. Settings are saved only in this checkout.</p><div class="form-grid section-gap">${fields.map(([key,label,kind])=>field(label,`<span class="path-input"><input data-setting="${key}" value="${esc(state.tools[key])}" aria-label="${label}"${disabled()}><button class="btn" data-browse="tools.${key}" data-kind="${kind}" aria-label="Browse ${label}">${icon('folder')}</button></span>`)).join('')}</div><div class="row wrap section-gap"><button class="btn primary" data-action="save-settings"${disabled()}>${icon('check')}Save local settings</button><button class="btn" data-doctor="extraction"${disabled()}>Check extraction dependencies</button><button class="btn" data-doctor="training"${disabled()}>Check training dependencies</button><button class="btn" data-open="${esc(data.root)}">${icon('folder')}Open project folder</button></div><div class="notice">${icon('shield')}<span>Study runs require TensorFlow 2.19.1. Use the dependency checks to inspect installed versions in the job log. The fixed vocabulary and study paths remain in signs.csv and project.json.</span></div></section>`;
}

function render(){
  if(!data)return;
  const views={overview,dataset:datasetPage,extract:extractPage,train:trainingPage,test:testPage,results:resultsPage,export:exportPage,settings:settingsPage};
  app.innerHTML=`<div class="shell">${navRail()}<main class="workspace">${topbar()}${state.offline?'<div class="notice">The local trainer is disconnected. Reopen launch.cmd if it has stopped.</div>':''}${views[state.page]()}<details id="job-console" class="job-console"${state.logOpen?' open':''}><summary>Job log <span id="job-summary" class="muted"></span></summary><pre id="job-output"></pre><div id="job-log-path" class="subtext"></div></details><footer class="workspace-foot"><span class="foot-mark"><i class="dot"></i> HUDYAT Train <span class="slash">/</span> Built for the language of movement.</span><span>Local workspace · original media retained</span></footer></main></div><div id="job-dock"></div>`;
  document.getElementById('job-console').addEventListener('toggle',event=>{state.logOpen=event.target.open;});
  if(data.warnings?.length)document.querySelector('.topbar').insertAdjacentHTML('afterend',data.warnings.map(message=>`<div class="notice">${icon('shield')}<span>${esc(message)}</span></div>`).join(''));
  updateStatus();
}
function updateStatus(){
  const summary=document.getElementById('job-summary'),output=document.getElementById('job-output'),dock=document.getElementById('job-dock');
  if(summary)summary.textContent=runtime.job.id?' · '+runtime.job.kind+' / '+runtime.job.status:' · ready';
  if(output){const bottom=output.scrollTop+output.clientHeight>=output.scrollHeight-25;output.textContent=logLines.join('\n');if(bottom)output.scrollTop=output.scrollHeight;}
  const logPath=document.getElementById('job-log-path');
  if(logPath)logPath.innerHTML=runtime.job.log_path?`Full log: <button class="text-btn" data-open="${esc(runtime.job.log_path)}">${esc(runtime.job.log_path)}</button>`:'';
  if(dock)dock.innerHTML=busy()?`<div class="job-dock"><i class="dot"></i><span>${esc(runtime.job.progress||runtime.job.kind+' · '+runtime.job.status)}</span><button class="btn small" data-action="job-log">Job log</button><button class="btn small primary" data-action="stop"${runtime.job.status==='stopping'?' disabled':''}>${icon('stop')}Stop job</button></div>`:'';
  const diagnostics=document.getElementById('live-diagnostics'),feedback=document.getElementById('live-feedback'),prediction=document.getElementById('live-prediction');
  const live=runtime.live,d=live.diagnostics||{};
  if((!busy()||runtime.job.kind!=='live')&&frameUrl){
    URL.revokeObjectURL(frameUrl);frameUrl='';
    const image=document.getElementById('camera-feed'),placeholder=document.getElementById('camera-placeholder');
    if(image){image.removeAttribute('src');image.hidden=true;}if(placeholder)placeholder.hidden=false;
  }
  if(diagnostics)diagnostics.textContent=[d.preview_fps!=null?'Preview '+d.preview_fps.toFixed(1)+' FPS':'',d.fps!=null?'Detection '+d.fps.toFixed(1)+' FPS':'',d.detector_ms!=null?'Detector '+d.detector_ms.toFixed(1)+' ms':'',d.capturing?'Capturing':''].filter(Boolean).join(' · ');
  if(live.attempt_count!==state.lastAttemptCount){state.lastAttemptCount=live.attempt_count;state.lastAttemptAt=Date.now();}
  const row=live.attempt&&Date.now()-(state.lastAttemptAt||0)<1800?live.attempt:live.estimate||live.attempt;
  if(prediction)prediction.textContent=row?.predicted!=null?row.predicted+' '+pct(row.confidence):'—';
  if(feedback)feedback.textContent=row===live.attempt&&row?({correct:'Correct',incorrect:'Incorrect',rejected:'Below confidence threshold',invalid:'Invalid capture',unscored:'Unscored prediction'}[row.outcome]||row.error||'Completed capture'):row?'Live estimate · not a scored attempt':live.message||'Choose a model and start the camera.';
  const top=document.getElementById('live-top3');
  if(top)top.innerHTML=table(['prediction','confidence'],(row?.top3||[]).map(r=>({prediction:r.label,confidence:pct(r.probability)})));
  for(const button of document.querySelectorAll('[data-live]'))button.disabled=!(live.ready&&state.mode==='manual'&&runtime.job.kind==='live');
}
async function loadClasses(){
  state.classes=state.model?(await api('/api/model?'+new URLSearchParams({path:state.model}))).classes:[];
  if(!state.classes.includes(state.target))state.target='';
}
async function runJob(action,extra={}){
  const form=modal.open?modal.querySelector('form'):document.querySelector('[data-form="training"]');
  if(form&&!form.reportValidity())return;
  const payload={action,...Object.fromEntries(['category','version','protocol','epochs','seed','augmentation','mirroring','development','model','camera','mode','target','profile','candidate','simulation_settings','output','allow_development'].map(key=>[key,state[key]])),tools:state.tools,...extra};
  if(action==='results')payload.rebuild=true;
  runtime.job=await api('/api/jobs',payload);
  modal.close();state.logOpen=true;logLines=[];
  render();notify(action==='train'?'Training started.':'Job started.');
}
function configure(){showModal('Your next experiment.',`<p class="subtext">Configure a real local training run.</p><form data-form="training" class="section-gap">${trainingFields()}</form>`,`<button class="btn" data-action="close-modal">Cancel</button>${jobButton('train','Start training')}`);}
async function inspectRun(identifier){
  state.resultId=identifier;resultData=await api('/api/results?'+new URLSearchParams({id:identifier}));
  state.model=resultData.record.source_model||resultData.record.path;
  await loadClasses();go('results');
  document.querySelector('.artifact-links')?.closest('section')?.scrollIntoView({block:'nearest'});
}
let browsing=null;
async function browse(fieldName,kind='directory',path){
  if(!browsing||browsing.field!==fieldName)browsing={field:fieldName,kind,returnConfigure:modal.open&&!!modal.querySelector('[data-form="training"]')};
  const value=path??(fieldName.startsWith('tools.')?state.tools[fieldName.slice(6)]:state[fieldName])??data.root;
  let listing;
  try{listing=await api('/api/browse?'+new URLSearchParams({path:value||data.root,kind}));}
  catch(error){if(path!==undefined)throw error;listing=await api('/api/browse?kind='+encodeURIComponent(kind));}
  browsing.path=listing.path;
  showModal('Choose '+(kind==='directory'?'a folder':kind==='json'?'a JSON file':'a Python executable'),`<p class="subtext path-caption">${esc(listing.path)}</p><div class="row section-gap"><button class="btn small" data-browse-path="${esc(listing.parent)}">← Parent folder</button></div><div class="file-list section-gap">${listing.entries.map(entry=>`<button class="file-entry" ${entry.directory?'data-browse-path':'data-browse-value'}="${esc(entry.path)}">${icon(entry.directory?'folder':'code')}<span>${esc(entry.name)}</span>${entry.directory?icon('chevron'):''}</button>`).join('')||'<div class="empty">No matching entries in this folder.</div>'}</div>`,`<button class="btn" data-action="close-modal">Cancel</button>${kind==='directory'?`<button class="btn primary" data-browse-value="${esc(listing.path)}">Use this folder</button>`:''}`);
}
async function choosePath(value){
  const {field,returnConfigure}=browsing;
  if(field.startsWith('tools.')){state.tools[field.slice(6)]=value;state.toolsDirty=true;}
  else{state[field]=value;if(field==='model'){state.resultId='';await loadClasses();}}
  browsing=null;
  if(returnConfigure)configure();else{modal.close();render();}
}
const actions={
  configure,
  'close-modal':()=>{modal.close();browsing=null;},
  'job-log':()=>{state.logOpen=true;const el=document.getElementById('job-console');if(el){el.open=true;el.scrollIntoView({block:'nearest',behavior:'smooth'});}},
  stop:async()=>{runtime.job=await api('/api/stop',{});render();notify('Stopping at the next frame or training batch…');},
  'refresh-results':async()=>{await api('/api/refresh',{});await refreshData();render();notify('Results refreshed.');},
  'save-settings':async()=>{state.tools=await api('/api/settings',state.tools);state.toolsDirty=false;notify('Local tool settings saved.');},
  'use-model':async()=>{const r=currentRecord();if(!r)return;state.model=r.source_model||r.path;await loadClasses();go('test');},
  'export-model':()=>{const r=currentRecord();if(!r)return;state.model=r.path;go('export');},
  exit:()=>showModal('Close the trainer?',`<p class="subtext">${busy()?'The current job will be asked to stop safely before the local server closes. Saved artifacts and raw recordings stay on disk.':'This stops the local server. You can reopen the trainer with launch.cmd.'}</p>`,`<button class="btn" data-action="close-modal">Keep working</button><button class="btn primary" data-action="confirm-exit">${busy()?'Stop job & exit':'Exit trainer'}</button>`),
  'confirm-exit':async()=>{await api('/api/shutdown',{});state.exiting=true;modal.close();notify('Closing the trainer safely…');},
  retry:()=>location.reload()
};
async function handleClick(event){
  const el=event.target.closest('button,tr[data-run]');if(!el||el.disabled)return;
  if(el.dataset.page)go(el.dataset.page);
  else if(el.dataset.chart){state.chart=el.dataset.chart;render();}
  else if(el.dataset.category){state.category=el.dataset.category;await refreshData();render();}
  else if(el.dataset.run)await inspectRun(el.dataset.run);
  else if(el.dataset.testTab){state.testTab=el.dataset.testTab;await loadClasses();render();}
  else if(el.dataset.resultTab){state.resultTab=el.dataset.resultTab;render();}
  else if(el.dataset.job)await runJob(el.dataset.job);
  else if(el.dataset.live)await api('/api/live',{type:el.dataset.live});
  else if(el.dataset.doctor)await runJob('doctor',{interpreter:el.dataset.doctor});
  else if(el.dataset.open){await api('/api/open',{path:el.dataset.open});}
  else if(el.dataset.browse)await browse(el.dataset.browse,el.dataset.kind);
  else if(el.dataset.browsePath!==undefined)await browse(browsing.field,browsing.kind,el.dataset.browsePath);
  else if(el.dataset.browseValue!==undefined)await choosePath(el.dataset.browseValue);
  else if(el.dataset.action)await actions[el.dataset.action]?.();
}
document.addEventListener('click',event=>{handleClick(event).catch(error=>notify(error.message));});
document.addEventListener('submit',event=>{event.preventDefault();if(event.target.dataset.form==='training')runJob('train').catch(error=>notify(error.message));});
document.addEventListener('input',event=>{
  const el=event.target,key=el.dataset.field;
  if(el.dataset.setting){state.tools[el.dataset.setting]=el.value;state.toolsDirty=true;return;}
  if(!key)return;
  state[key]=el.type==='checkbox'?el.checked:el.type==='number'?Number(el.value):el.value;
  if(key==='query'){const rows=document.getElementById('experiment-rows');if(rows)rows.innerHTML=resultRows();}
});
document.addEventListener('change',event=>{
  (async()=>{
    const el=event.target,key=el.dataset.field;if(!key)return;
    state[key]=el.type==='checkbox'?el.checked:el.type==='number'?Number(el.value):el.value;
    if(key==='category'||key==='version'){
      if(key==='version'){state.output=data.paths.exports_dir+'/'+state.version;state.resultId='';}
      await refreshData();if(!modal.open)render();
    }else if(key==='model'){state.resultId='';await loadClasses();if(!modal.open)render();}
    else if((key==='target'||key==='mode')&&runtime.job.kind==='live'&&busy())await api('/api/live',key==='target'?{type:'target',label:state.target}:{type:'mode',mode:state.mode});
    else if(['protocol','runFilter','graph'].includes(key)&&!modal.open)render();
  })().catch(error=>notify(error.message));
});
document.addEventListener('keydown',event=>{
  if(event.target.closest('input,select,textarea,[contenteditable]')||modal.open)return;
  if((event.key==='Enter'||event.key===' ')&&event.target.matches('tr[data-run]')){event.preventDefault();inspectRun(event.target.dataset.run).catch(error=>notify(error.message));}
  else if(state.page==='test'&&state.testTab==='live'&&state.mode==='manual'&&runtime.live.ready&&[' ','Enter'].includes(event.key)){
    event.preventDefault();api('/api/live',{type:event.key===' '?'capture':'finish'}).catch(error=>notify(error.message));
  }
});
window.addEventListener('hashchange',()=>{const page=location.hash.slice(1);if(pageNames[page]&&page!==state.page){state.page=page;render();}});
modal.addEventListener('click',event=>{if(event.target!==modal)return;const r=modal.getBoundingClientRect();if(event.clientX<r.left||event.clientX>r.right||event.clientY<r.top||event.clientY>r.bottom)modal.close();});
modal.addEventListener('close',()=>{browsing=null;if(data&&refreshPending){render();refreshPending=false;}});

async function readFrames(){
  if(frameReader)return;frameReader=true;
  let after=0;
  try{
    while(runtime.job.kind==='live'&&busy()&&!state.exiting){
      const response=await fetch('/api/frame?after='+after,{cache:'no-store'});
      if(!response.ok)throw Error('Camera frame unavailable');
      after=Number(response.headers.get('X-Frame-Version')||after);
      if(response.status===204){await new Promise(resolve=>setTimeout(resolve,100));continue;}
      const blob=await response.blob(),image=document.getElementById('camera-feed');
      if(!image)continue;
      const next=URL.createObjectURL(blob),old=frameUrl;frameUrl=next;image.src=next;image.hidden=false;
      const placeholder=document.getElementById('camera-placeholder');if(placeholder)placeholder.hidden=true;
      try{await image.decode();}catch{}
      if(old)URL.revokeObjectURL(old);
    }
  }catch(error){if(runtime.job.kind==='live'&&busy())notify(error.message);}
  finally{frameReader=false;if(!busy()&&frameUrl){URL.revokeObjectURL(frameUrl);frameUrl='';}}
}
let failures=0;
async function poll(){
  try{
    const previous=runtime.job.status;
    const value=await api('/api/status?after='+state.sequence);runtime=value;state.sequence=value.sequence;failures=0;
    if(state.offline){state.offline=false;await refreshData();}
    if(state.jobId!==value.job.id){state.jobId=value.job.id;logLines=[];}
    logLines.push(...value.logs.map(entry=>entry.text));logLines=logLines.slice(-2000);
    if(value.job.run_path&&state.lastRunPath!==value.job.run_path){state.lastRunPath=value.job.run_path;state.model=value.job.run_path;await loadClasses();}
    if(value.revision!==state.revision){if(state.page==='results'&&state.resultId)resultData=await api('/api/results?id='+encodeURIComponent(state.resultId));await refreshData();}
    if(previous!==value.job.status&&!busy()&&value.job.id){
      if(value.job.status==='failed'){state.logOpen=true;notify('Job failed. The worker output explains what needs attention.');}
      else notify(value.job.status==='cancelled'?'Job stopped.':'Job completed. Results saved.');
      if(!modal.open)render();
    }
    if(refreshPending&&!modal.open&&!document.activeElement?.matches('input,select,textarea')){render();refreshPending=false;}
    updateStatus();
    if(value.job.kind==='live'&&busy())readFrames();
  }catch(error){
    if(state.exiting){app.innerHTML='<div class="closed-studio"><h1>Workspace closed.</h1><p>You can close this tab. Reopen launch.cmd to return.</p></div>';return;}
    if(++failures===3){state.offline=true;render();notify('Lost connection to the local trainer. Your worker may still be running.');}
  }
  setTimeout(poll,450);
}
async function initializeStudio(){
  try{
    await refreshData();state.model=data.initial.model;state.camera=data.initial.camera;state.mode=data.initial.mode;
    if(state.model){await loadClasses();state.page='test';state.testTab='live';}
    render();poll();
  }catch(error){app.innerHTML=`<div class="closed-studio"><h1>Could not open the workspace.</h1><p>${esc(error.message)}</p><p>Start the local trainer with launch.cmd.</p><button class="btn primary" data-action="retry">Retry</button></div>`;}
}
initializeStudio();

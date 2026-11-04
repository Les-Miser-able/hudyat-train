/** Import either settings alone or HUDYAT Recognition lab's Copy results JSON. */
export function importSettings(input,baseline,enhanced){
 const raw=input?.settings??input;
 if(!raw||typeof raw!=='object'||Array.isArray(raw))throw Error('Paste a settings object or Recognition lab results JSON.');
 if(raw.profile==='baseline')return structuredClone(baseline);
 if(raw.profile==='enhanced')return structuredClone(enhanced);
 const settings={...structuredClone(baseline),...raw,profile:'custom',
  lock:{...baseline.lock,...raw.lock},window:{...baseline.window,...raw.window}};
 const enums={dynamicCapture:['fixed','sliding','sign'],detectionBackground:['off','blur','remove'],delegate:['CPU','GPU'],
  cameraResolution:['720p','480p'],alphabetHands:[1,2],alphabetModel:['published','candidate']};
 for(const [key,values] of Object.entries(enums))if(!values.includes(settings[key]))throw Error('Invalid '+key+'.');
 for(const [key,value] of Object.entries(baseline))if(typeof value==='boolean'&&typeof settings[key]!=='boolean')throw Error(key+' must be true or false.');
 const number=(value,min,max=Infinity,integer=false)=>typeof value==='number'&&Number.isFinite(value)&&value>=min&&value<=max&&(!integer||Number.isInteger(value));
 for(const [key,min,max] of [['alphabetPollMs',33],['earlyMinHandsMs',200],['endAbsentMs',100,1500],['earlyAgree',1,5]]){
  if(!number(settings[key],min,max,true))throw Error('Invalid '+key+'.');
 }
 for(const [key,min,max,integer] of [['holdMs',0,Infinity,false],['minFrames',1,Infinity,true],['graceFrames',0,Infinity,true],
  ['minScore',0,1,false],['minMargin',0,1,false],['maxMotion',0,Infinity,false]]){
  if(settings.lock[key]!==undefined&&!number(settings.lock[key],min,max,integer))throw Error('Invalid lock.'+key+'.');
 }
 for(const key of Object.keys(baseline.window))if(!number(settings.window[key],1,Infinity,true))throw Error('Invalid window.'+key+'.');
 if(settings.window.minSpanMs>settings.window.spanMs)throw Error('Window minimum span cannot exceed its full span.');
 return settings;
}

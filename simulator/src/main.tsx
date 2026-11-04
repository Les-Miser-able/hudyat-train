import React,{useEffect,useRef,useState} from 'react';
import {createRoot} from 'react-dom/client';
import {CategoryRecognition} from '@hudyat/features/fsl-recognition/category-recognition';
import {RecognitionLab} from '@hudyat/features/fsl-recognition/recognition-lab';
import {BlurredPreview} from '@hudyat/features/fsl-recognition/blurred-preview';
import {BASELINE,ENHANCED,CAMERA_SIZE,saveSettings,type RecognitionSettings} from '@hudyat/features/fsl-recognition/experiment';
import {useCamera} from '@hudyat/hooks/use-camera';
import {forgetCameraGrant} from '@hudyat/lib/camera-session';
import {usePreloadRecognition} from '@hudyat/features/fsl-recognition/preload';
import {recognitionMetrics} from '@hudyat/features/fsl-recognition/metrics';
import {signLog} from '@hudyat/features/fsl-recognition/sign-log';
import {importSettings} from './settings.js';
import './style.css';

type Config={token:string;profile:string;settings:unknown;target?:string;classes:string[];model:{id:string;version:string};candidate_available:boolean;report_directory:string};

function Preload({category}:{category:string}){usePreloadRecognition(category);return null;}

function Simulator({config}:{config:Config}){
 const initial=useRef<RecognitionSettings>(importSettings(config.settings??{profile:config.profile},BASELINE,ENHANCED));
 const [settings,setSettings]=useState(initial.current);
 const [target,setTarget]=useState(config.target??'');
 const [entered,setEntered]=useState(false),[enabled,setEnabled]=useState(true);
 const [status,setStatus]=useState({phase:'loading',message:'Enter the camera step when ready.',version:undefined});
 const [feedback,setFeedback]=useState(''),[hold,setHold]=useState<{progress:number;hint?:string}|null>(null);
 const [message,setMessage]=useState(''),[closed,setClosed]=useState(false),[retry,setRetry]=useState(0);
 const [editor,setEditor]=useState(JSON.stringify(settings,null,2));
 const [configuration,setConfiguration]=useState(crypto.randomUUID());
 const camera=useCamera(),live=camera.status==='live';
 const sounds=useRef<Record<string,HTMLAudioElement>>({});
 const decisions=useRef<any[]>([]),seen=useRef(new WeakSet<object>());
 const current=useRef({settings,configuration});current.current={settings,configuration};
 const latest=useRef<any>(null),pending=useRef<Promise<unknown>>(Promise.resolve());
 const lastCamera=useRef<any>(null);
 const feedbackTimer=useRef<ReturnType<typeof setTimeout>|undefined>(undefined);

 const snapshot=()=>{
  const condition=Array.from(document.querySelectorAll('aside label')).find(label=>label.firstChild?.textContent?.trim()==='Condition')?.querySelector('input')?.value??'';
  const metrics=JSON.parse(recognitionMetrics.export(current.current.settings,condition));
  const track=(camera.videoRef.current?.srcObject as MediaStream|null)?.getVideoTracks()[0]?.getSettings();
  if(track)lastCamera.current={width:track.width,height:track.height,frameRate:track.frameRate,facingMode:track.facingMode};
  return {...metrics,configuration:current.current.configuration,decisions:[...decisions.current],
   camera:lastCamera.current};
 };
 const post=async(route:string,payload:unknown,keepalive=false)=>{
  const response=await fetch('/simulation/'+route,{method:'POST',headers:{'Content-Type':'application/json','X-Simulation-Token':config.token},
   body:JSON.stringify(payload),keepalive});
  const result=await response.json();
  if(!response.ok)throw Error(result.error??'Could not save browser results.');
  return result;
 };
 const queuePost=(route:string,payload:unknown)=>{
  pending.current=pending.current.catch(()=>{}).then(()=>post(route,payload)).catch(error=>setMessage(error.message));
  return pending.current;
 };

 useEffect(()=>{
  saveSettings(initial.current);
  sessionStorage.setItem('hudyat-lab','1');
  const update=()=>{latest.current=snapshot();};
  const offMetrics=recognitionMetrics.subscribe(update);
  const offDecisions=signLog.subscribe(()=>{
   for(const decision of [...signLog.entries()].reverse())if(!seen.current.has(decision)){
    seen.current.add(decision);decisions.current.push(decision);
    void queuePost('event',{...decision,type:'decision',configuration:current.current.configuration});
   }
   update();
  });
  const interval=setInterval(()=>{update();void queuePost('snapshot',latest.current);},5000);
  const onHide=()=>{void post('snapshot',snapshot(),true).catch(()=>{});};
  window.addEventListener('pagehide',onHide);
  update();
  return ()=>{offMetrics();offDecisions();clearInterval(interval);window.removeEventListener('pagehide',onHide);};
 },[]);

 useEffect(()=>{
  if(closed)return;
  const heartbeat=setInterval(()=>{void fetch('/simulation/ping').then(response=>{
   if(!response.ok)throw Error();return response.json();
  }).then(result=>{if(!result.running)throw Error();}).catch(()=>{
   camera.stop();setClosed(true);setMessage('The trainer stopped the simulator. Camera released.');
  });},3000);
  return ()=>clearInterval(heartbeat);
 },[closed]);

 useEffect(()=>()=>{clearTimeout(feedbackTimer.current);},[]);
 useEffect(()=>{setFeedback('');clearTimeout(feedbackTimer.current);},[target,live]);

 const apply=(next:RecognitionSettings)=>{
  if(latest.current)void queuePost('save',latest.current);
  saveSettings(next);setSettings(next);setEditor(JSON.stringify(next,null,2));
  setConfiguration(crypto.randomUUID());decisions.current=[];seen.current=new WeakSet();
  signLog.clear();recognitionMetrics.reset();setFeedback('');setMessage('Settings applied. Restart the camera after changing its resolution.');
  setStatus({phase:'loading',message:'Applying HUDYAT settings...',version:undefined});
 };
 const sound=(kind:string)=>{
  const audio=sounds.current[kind]??new Audio('/sounds/'+kind+'.mp3');sounds.current[kind]=audio;
  audio.muted=false;audio.currentTime=0;void audio.play().catch(()=>{});
 };
 const begin=()=>{
  setMessage('');setFeedback('');setEnabled(true);
  for(const kind of ['correct','wrong']){
   const audio=sounds.current[kind]??new Audio('/sounds/'+kind+'.mp3');sounds.current[kind]=audio;audio.muted=true;
   void audio.play().then(()=>{audio.pause();audio.currentTime=0;audio.muted=false;}).catch(()=>{audio.muted=false;});
  }
  void camera.start(CAMERA_SIZE[settings.cameraResolution]);
 };
 const stop=async()=>{
  const saved=snapshot();forgetCameraGrant();camera.stop();setEnabled(false);
  await pending.current;const result=await post('save',saved);setMessage('Saved browser report: '+result.saved);
 };
 const correct=()=>{
  setFeedback('Correct');sounds.current.wrong?.pause();sound('correct');
  clearTimeout(feedbackTimer.current);
  feedbackTimer.current=setTimeout(()=>setMessage('Sign recognized. Choose another sign or press New attempt.'),650);
  void queuePost('event',{type:'correct',target,configuration:current.current.configuration,at:performance.now()});
 };
 const wrong=()=>{
  setFeedback('Try again');sound('wrong');clearTimeout(feedbackTimer.current);
  feedbackTimer.current=setTimeout(()=>setFeedback(''),1800);
  void queuePost('event',{type:'wrong',target,configuration:current.current.configuration,at:performance.now()});
 };
 const repeat=()=>{setFeedback('');setEnabled(false);setTimeout(()=>setEnabled(true),0);};
 const finish=async()=>{try{await stop();await post('stop',{});setClosed(true);setMessage('Simulation finished. Reports are available in the trainer Results tab.');}catch(error){setMessage(String(error));}};
 const readSettings=()=>{try{apply(importSettings(JSON.parse(editor),BASELINE,ENHANCED));}catch(error){setMessage(String(error));}};

 return <main>
  <header><div><p className="eyebrow">HUDYAT TRAIN · BROWSER TESTING</p><h1>HUDYAT camera simulator</h1>
   <p>Runs the main project’s camera and recognition code with your selected model.</p></div>
   <span className="badge">{config.model.id} · {settings.profile}</span></header>
  <section className="intro"><p>Choose an expected sign, then enter the camera step. Camera access, CPU/GPU tracking, background processing,
   sign windows and acceptance rules come directly from HUDYAT. No camera images or landmark arrays are saved.</p>
   <p>Results describe this browser and device. They are recorded separately from held-out training accuracy.</p></section>
  <div className="layout"><section className="camera-panel">
   <div className="toolbar"><label>Expected sign <select aria-label="Expected sign" value={target} onChange={event=>setTarget(event.target.value)}>
    <option value="">Choose an expected sign</option>{config.classes.map(label=><option key={label}>{label}</option>)}</select></label>
    {!entered?<button onClick={()=>setEntered(true)} disabled={closed}>Enter camera step</button>:
     <button onClick={live?()=>{void stop().catch(error=>setMessage(error.message));}:begin} disabled={closed||!target}>{live?'Stop camera & save':'Start camera'}</button>}
    <button onClick={repeat} disabled={!live}>New attempt</button></div>
   <div className="camera"><video ref={camera.videoRef} autoPlay muted playsInline aria-label="Live mirrored camera preview"/>
    {live&&settings.previewBlur&&<BlurredPreview video={camera.videoRef} className="blur-preview"/>}
    {!live&&<div className="camera-placeholder">{camera.status==='requesting'?'Allow camera access in your browser.':closed?'Simulation ended':entered?'Ready for the camera':'Camera step not opened'}</div>}
    {live&&feedback&&<strong className={'feedback '+(feedback==='Correct'?'correct':'wrong')} role="status">{feedback}</strong>}
    {live&&settings.holdFeedback&&hold&&!feedback&&<div className="hold">{hold.hint&&<p>{hold.hint}</p>}<progress max={1} value={hold.progress}/></div>}
   </div>
   <p className="status" data-phase={status.phase} role={camera.error||status.phase==='error'?'alert':'status'}>{camera.error||status.message}</p>
   <p className="message" role="status">{message}</p>
   <div className="toolbar"><button onClick={()=>{void queuePost('save',snapshot()).then(()=>setMessage('Browser results saved.'));}} disabled={closed}>Save reports</button>
    <button onClick={()=>{void finish();}} disabled={closed}>Finish simulation</button>
    <button onClick={()=>setRetry(value=>value+1)} disabled={!entered||closed}>Reload model</button></div>
   <p className="small">Model {config.model.version}. {config.candidate_available?'An Alphabet candidate model is available.':'Choose an optional Alphabet candidate in the trainer to compare releases.'}</p>
   <details><summary>Complete settings / import from HUDYAT</summary><p>Paste settings or JSON from HUDYAT’s “Copy results”. This also exposes sliding-window parameters. Use profile “custom” for edited JSON; named profiles use HUDYAT’s current defaults.</p>
    <textarea aria-label="Settings JSON" value={editor} onChange={event=>setEditor(event.target.value)} spellCheck={false}/>
    <button onClick={readSettings} disabled={closed}>Apply settings JSON</button></details>
   <p className="small">Reports: {config.report_directory}</p>
  </section><aside><RecognitionLab settings={settings} onChange={apply}/></aside></div>
  {settings.preload&&!closed&&<Preload key={JSON.stringify([config.model.id,settings.alphabetTracking,settings.delegate,settings.alphabetHands,settings.alphabetModel])} category={config.model.id}/>}
  {entered&&!closed&&<CategoryRecognition category={config.model.id} targetLabel={target||undefined} video={camera.videoRef}
   active={live&&enabled} onCorrect={correct} onWrong={wrong} onStatus={setStatus} retry={retry} settings={settings} onHold={setHold}/>}
 </main>;
}

fetch('/simulation/config').then(response=>{if(!response.ok)throw Error('The simulator is unavailable.');return response.json();}).then(config=>{
 config.settings=importSettings(config.settings??{profile:config.profile},BASELINE,ENHANCED);
 saveSettings(config.settings);
 createRoot(document.getElementById('root')!).render(<Simulator config={config}/>);
}).catch(error=>{document.getElementById('root')!.textContent='Could not open the HUDYAT simulator: '+error.message;});

// Uses a fake camera (optionally a recorded Y4M file), never the physical webcam.
import assert from 'node:assert/strict';
import {mkdirSync,writeFileSync} from 'node:fs';
import path from 'node:path';
import {chromium} from 'playwright-core';

const args={};
for(let i=2;i<process.argv.length;i+=2)args[process.argv[i].replace(/^--/,'')]=process.argv[i+1];
const url=args.url??'http://127.0.0.1:8766/?lab=1';
const output=path.resolve(args.output??'../tmp/simulator-browser-check');mkdirSync(output,{recursive:true});
const flags=['--use-fake-device-for-media-stream','--use-fake-ui-for-media-stream','--autoplay-policy=no-user-gesture-required'];
if(args.video)flags.push('--use-file-for-fake-video-capture='+path.resolve(args.video));
const browser=await chromium.launch({executablePath:args.browser??'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,args:flags});
const context=await browser.newContext({viewport:{width:1440,height:1100},permissions:['camera']});
const page=await context.newPage(),errors=[],snapshots=[],requests=[];
page.on('pageerror',error=>errors.push(String(error)));
page.on('request',request=>{
 requests.push(request.url());
 if(/\/simulation\/(snapshot|save)$/.test(request.url()))try{snapshots.push(request.postDataJSON());}catch{}
});
page.on('console',message=>{if(message.type()==='error')writeFileSync(path.join(output,'console.log'),message.text()+'\n',{flag:'a'});});
const waitSnapshot=predicate=>new Promise((resolve,reject)=>{const start=Date.now();const timer=setInterval(()=>{
 if(snapshots.some(predicate)){clearInterval(timer);resolve();}
 else if(Date.now()-start>90000){clearInterval(timer);reject(Error('Expected camera/inference measurement was not recorded.'));}
},250);});
try{
 await page.goto(url,{waitUntil:'networkidle'});
 await page.getByRole('heading',{name:'HUDYAT camera simulator'}).waitFor();
 let settings=JSON.parse(await page.getByLabel('Settings JSON').inputValue());
 assert.equal(settings.profile,'baseline');assert.equal(settings.endAbsentMs,500);assert.equal(settings.alphabetTracking,false);
 for(const label of ['Background (detection)','Alphabet model','Alphabet tracking (VIDEO mode)','Letters like Numbers (sign by sign)',
  'Hand tracking on','Camera (restart camera)','Alphabet hands','Preload model (reload page)','Hold progress + hints','Blurred preview',
  'Moving signs','Min confidence','Min margin','Max motion','Bad frames forgiven']){
  assert.equal(await page.getByLabel(label).count(),1,label);
 }
 await page.getByLabel('Expected sign',{exact:true}).selectOption({index:1});
 await page.getByLabel('Condition',{exact:true}).fill('Recorded fake camera; synthetic model');
 await page.getByRole('button',{name:'Enter camera step',exact:true}).click();
 await page.getByRole('button',{name:'Start camera',exact:true}).click();
 await page.waitForFunction(()=>document.querySelector('.status')?.textContent?.includes('Recognition running'),{},{timeout:90000});
 await page.waitForFunction(()=>document.querySelector('video')?.srcObject?.active===true);
 const baselineCamera=await page.locator('video').evaluate(video=>video.srcObject.getVideoTracks()[0].getSettings());
 await page.getByRole('button',{name:'Save reports',exact:true}).click();
 if(args.video)await waitSnapshot(s=>s.settings.profile==='baseline'&&s.summary.predictMs>0);
 await page.screenshot({path:path.join(output,'baseline.png'),fullPage:true});
 await page.getByRole('button',{name:'Stop camera & save',exact:true}).click();
 await page.getByRole('button',{name:'Enhanced',exact:true}).click();
 settings=JSON.parse(await page.getByLabel('Settings JSON').inputValue());
 assert.equal(settings.profile,'enhanced');assert.equal(settings.delegate,'GPU');assert.equal(settings.cameraResolution,'480p');
 assert.equal(settings.alphabetHands,1);assert.equal(settings.alphabetTracking,true);assert.equal(settings.preload,true);
 await page.getByRole('button',{name:'Start camera',exact:true}).click();
 await page.waitForFunction(()=>document.querySelector('.status')?.textContent?.includes('Recognition running'),{},{timeout:90000});
 await waitSnapshot(s=>s.settings.profile==='enhanced'&&s.summary.captureMs>0&&s.camera?.width===640);
 await page.getByRole('button',{name:'Stop camera & save',exact:true}).click();
 settings={...settings,profile:'custom',delegate:'CPU',detectionBackground:'remove',previewBlur:true,preload:false};
 await page.getByText('Complete settings / import from HUDYAT',{exact:true}).click();
 await page.getByLabel('Settings JSON').fill(JSON.stringify({settings}));
 await page.getByRole('button',{name:'Apply settings JSON',exact:true}).click();
 await page.getByRole('button',{name:'Start camera',exact:true}).click();
 await page.waitForFunction(()=>document.querySelector('.status')?.textContent?.includes('Recognition running'),{},{timeout:90000});
 await waitSnapshot(s=>s.settings.detectionBackground==='remove'&&s.summary.segmentMs>0);
 assert.ok(requests.some(url=>url.includes('selfie_segmenter.tflite')));
 assert.equal(await page.locator('canvas.blur-preview').count(),1);
 await page.getByRole('button',{name:'Stop camera & save',exact:true}).click();
 await page.screenshot({path:path.join(output,'custom.png'),fullPage:true});
 const configuration=await (await page.request.get(new URL('/simulation/config',url).href)).json();
 if(configuration.candidate_available){
  await page.getByLabel('Alphabet model',{exact:false}).selectOption('candidate');
  await page.getByRole('button',{name:'Start camera',exact:true}).click();
  await waitSnapshot(s=>s.settings.alphabetModel==='candidate'&&s.summary.predictMs>0);
  assert.ok(requests.some(url=>url.endsWith('/api/models/files/models/alphabet/candidate.json')));
  await page.getByRole('button',{name:'Stop camera & save',exact:true}).click();
 }
 assert.deepEqual(errors,[]);
 assert.ok(snapshots.some(s=>s.settings.profile==='baseline'));
 assert.ok(snapshots.some(s=>s.settings.profile==='enhanced'));
 assert.ok(snapshots.some(s=>s.settings.profile==='custom'));
 assert.ok(snapshots.some(s=>s.condition==='Recorded fake camera; synthetic model'));
 const result={physical_camera_opened:false,recorded_camera:Boolean(args.video),baselineCamera,
  browser_version:browser.version(),
  configurations:[...new Set(snapshots.map(s=>s.configuration))],
  snapshots:snapshots.map(s=>({configuration:s.configuration,settings:s.settings,summary:s.summary,camera:s.camera})),
  source_pipeline:'actual HUDYAT modules',browser_errors:errors};
 writeFileSync(path.join(output,'validation.json'),JSON.stringify(result,null,2));
 console.log('PASS: baseline/enhanced/custom, original Recognition lab, real browser detector/model, segmentation, preview blur, candidate when supplied, and separate reports.');
}catch(error){
 await page.screenshot({path:path.join(output,'failure.png'),fullPage:true});
 writeFileSync(path.join(output,'failure.json'),JSON.stringify({error:String(error),browser_errors:errors,snapshots},null,2));
 throw error;
}finally{
 await context.close();await browser.close();
}

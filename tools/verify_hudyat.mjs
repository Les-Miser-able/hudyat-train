// Read HUDYAT's actual loader and preprocessing code; do not substitute a test implementation.
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {readFileSync,writeFileSync} from 'node:fs';
import {createRequire} from 'node:module';
import path from 'node:path';
import {pathToFileURL} from 'node:url';

const values=process.argv.slice(2),args={};
for(let i=0;i<values.length;i+=2)args[values[i].replace(/^--/,'')]=values[i+1];
for(const key of ['hudyat','bundle','fixture'])if(!args[key])throw Error('--'+key+' is required');
const hudyat=path.resolve(args.hudyat),bundle=path.resolve(args.bundle);
const runtime=path.join(hudyat,'client','src','features','fsl-recognition');
const {buildModel,loadWeights,resample}=await import(pathToFileURL(path.join(runtime,'category-core.js')));
const {normalizeFrame,normalizeSequence,normalizeFrameLocation}=await import(pathToFileURL(path.join(runtime,'normalize.ts')));
const {resampleSign,mirrorFrames,autoHand,signWindow}=await import(pathToFileURL(path.join(runtime,'sign-capture.js')));
const require=createRequire(path.join(hudyat,'client','package.json'));
const tf=args.tfjs?require(path.resolve(args.tfjs)):require('@tensorflow/tfjs');
await tf.setBackend('cpu');
await tf.ready();
const sha=bytes=>createHash('sha256').update(bytes).digest('hex');
let preprocessingError=0;
function compare(actual,expected,label,tolerance=2e-4){
 const flatten=value=>Array.isArray(value)||ArrayBuffer.isView(value)?Array.from(value).flatMap(flatten):[value];
 const a=flatten(actual),b=flatten(expected);
 assert.equal(a.length,b.length,label+' length');
 let error=0;
 a.forEach((v,i)=>{assert.ok(Number.isFinite(v),label+' nonfinite');error=Math.max(error,Math.abs(v-b[i]));});
 assert.ok(error<tolerance,label+' max error '+error);
 return error;
}
const fixture=JSON.parse(readFileSync(args.fixture,'utf8'));
for(const [i,c] of fixture.cases.entries()){
 const raw=c.raw.map(f=>Float32Array.from(f));
 const located=raw.map((f,t)=>normalizeFrameLocation(f,c.faces[t],c.width,c.height));
 const sequence=normalizeSequence(raw,c.width,c.height);
 const comparisons=[
  ['normalizeFrame',raw.map(f=>normalizeFrame(f,c.width,c.height)),c.normalized],
  ['normalizeSequence',sequence,c.sequence],['normalizeFrameLocation',located,c.located],
  ['legacyResample',resample(sequence),c.legacy],['resampleSign',resampleSign(located),c.sign],
  ['mirrorFrames',mirrorFrames(located),c.mirror],
 ];
 for(const [label,actual,expected] of comparisons)preprocessingError=Math.max(preprocessingError,compare(actual,expected,`case ${i}: ${label}`));
 assert.equal(autoHand(raw,c.width,c.height),c.auto_hand,'canonical hand');
 assert.deepEqual(signWindow(raw.map(f=>Boolean(f[126]||f[127])),25),c.window,'sign window');
}
function artifact(url){
 const target=path.resolve(bundle,url.replace(/^\.\//,''));
 if(!target.startsWith(bundle+path.sep))throw Error('Artifact path escapes bundle');
 return readFileSync(target);
}
const catalog=JSON.parse(readFileSync(path.join(bundle,'catalog.json'),'utf8'));
const releases=[];
for(const entry of catalog.categories.filter(e=>e.available)){
 const metadataBytes=artifact(entry.modelUrl),weights=artifact(entry.weightsUrl),metadata=JSON.parse(metadataBytes);
 assert.equal(sha(metadataBytes),entry.modelSha256);
 assert.equal(sha(weights),entry.weightsSha256);
 assert.equal(sha(artifact(entry.handUrl)),entry.handSha256);
 if(entry.faceUrl)assert.equal(sha(artifact(entry.faceUrl)),entry.faceSha256);
 assert.equal(metadata.category,entry.id);
 assert.equal(metadata.version,entry.version);
 assert.equal(metadata.weightBytes,weights.length);
 assert.equal(metadata.weightSha256,sha(weights));
 assert.equal(metadata.preprocessing.extraction.model_sha256,entry.handSha256);
 if(metadata.inputShape[1]===141)assert.equal(metadata.preprocessing.face.model_sha256,entry.faceSha256);
 const probe=JSON.parse(artifact(entry.modelUrl.replace(/model\.json$/,'parity.json')));
 const model=buildModel(tf,metadata.classes.length,metadata.inputShape[1]);
 let input,output;
 try{
  loadWeights(tf,model,metadata,weights.buffer.slice(weights.byteOffset,weights.byteOffset+weights.byteLength));
  input=tf.tensor(probe.inputs);output=model.predict(input);
  const error=compare(await output.array(),probe.expected,entry.id+' prediction');
  releases.push({category:entry.id,variant:metadata.variant,version:entry.version,modelSha256:entry.modelSha256,weightsSha256:entry.weightsSha256,maxError:error});
  console.log(`PASS ${entry.id}/${metadata.variant}: Keras vs HUDYAT TF.js max error ${error}`);
 }finally{input?.dispose();output?.dispose();model.dispose();}
}
assert.ok(releases.length,'No available model releases in bundle');
const result={verifiedAt:new Date().toISOString(),preprocessingMaxError:preprocessingError,releases,
 runtimeHashes:Object.fromEntries(['category-core.js','normalize.ts','sign-capture.js'].map(file=>[file,sha(readFileSync(path.join(runtime,file)))]))};
writeFileSync(path.join(bundle,'verification.json'),JSON.stringify(result,null,2)+'\n');
console.log(`PASS preprocessing in 3 camera aspect ratios; max error ${preprocessingError}`);
console.log('Verification saved to '+path.join(bundle,'verification.json'));

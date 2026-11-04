// Compile the trainer shell around HUDYAT's original camera, settings, and recognition modules.
import {createHash} from 'node:crypto';
import {readFileSync,writeFileSync,mkdirSync,existsSync} from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {build,normalizePath} from 'vite';

const args={};
for(let i=2;i<process.argv.length;i+=2)args[process.argv[i].replace(/^--/,'')]=process.argv[i+1];
if(!args.hudyat||!args.output)throw Error('Use --hudyat PATH --output PATH.');
const root=path.dirname(fileURLToPath(import.meta.url));
const client=path.resolve(args.hudyat,'client'),source=path.join(client,'src');
const output=path.resolve(args.output);
const mainLock=JSON.parse(readFileSync(path.join(client,'package-lock.json'),'utf8'));
for(const name of ['react','react-dom','@tensorflow/tfjs','@mediapipe/tasks-vision']){
 const installed=JSON.parse(readFileSync(path.join(root,'node_modules',name,'package.json'),'utf8')).version;
 const expected=mainLock.packages?.['node_modules/'+name]?.version;
 if(expected&&installed!==expected)throw Error(`Simulator ${name} is ${installed}; HUDYAT uses ${expected}. Update simulator/package.json and run setup-simulator.cmd.`);
}
for(const file of ['features/fsl-recognition/category-recognition.tsx','features/fsl-recognition/recognition-lab.tsx','hooks/use-camera.ts']){
 if(!existsSync(path.join(source,file)))throw Error('Missing HUDYAT source: '+file);
}
const hashes={};
const aliases=['react-dom','react','@tensorflow/tfjs','@mediapipe/tasks-vision'].map(name=>({
 find:new RegExp('^'+name.replace(/[.*+?^${}()|[\]\\]/g,'\\$&')+'(?=/|$)'),
 replacement:normalizePath(path.join(root,'node_modules',name)),
}));
aliases.push({find:'@hudyat',replacement:normalizePath(source)},{find:'@',replacement:normalizePath(source)});
await build({
 root,configFile:false,publicDir:false,base:'/',
 define:{'import.meta.env.VITE_API_URL':JSON.stringify('')},
 resolve:{alias:aliases,dedupe:['react','react-dom','@tensorflow/tfjs','@mediapipe/tasks-vision']},
 esbuild:{jsx:'automatic'},
 build:{outDir:output,emptyOutDir:false,sourcemap:true,chunkSizeWarningLimit:3000},
 plugins:[{name:'record-hudyat-source',transform(code,id){
  const file=id.split('?')[0];
  if(file.startsWith(source+path.sep)||normalizePath(file).startsWith(normalizePath(source)+'/')){
   hashes[normalizePath(path.relative(source,file))]=createHash('sha256').update(readFileSync(file)).digest('hex');
  }
 }}],
});
const dependencies=Object.fromEntries(['react','react-dom','@tensorflow/tfjs','@mediapipe/tasks-vision'].map(name=>[
 name,JSON.parse(readFileSync(path.join(root,'node_modules',name,'package.json'),'utf8')).version,
]));
mkdirSync(output,{recursive:true});
writeFileSync(path.join(output,'hudyat-source.json'),JSON.stringify({
 hudyat:path.resolve(args.hudyat),builtAt:new Date().toISOString(),dependencies,
 files:Object.fromEntries(Object.entries(hashes).sort(([a],[b])=>a.localeCompare(b))),
},null,2)+'\n');

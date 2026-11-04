import assert from 'node:assert/strict';
import test from 'node:test';
import {importSettings} from '../src/settings.js';

const baseline={profile:'baseline',dynamicCapture:'sign',detectionBackground:'off',delegate:'CPU',cameraResolution:'720p',
 alphabetHands:2,alphabetModel:'published',alphabetPollMs:200,earlyMinHandsMs:800,endAbsentMs:500,earlyAgree:2,
 alphabetTracking:false,preload:false,lock:{holdMs:1200,minFrames:5,graceFrames:0},
 window:{spanMs:2000,minSpanMs:1800,everyMs:500,minFrames:8,agree:2}};
const enhanced={...baseline,profile:'enhanced',delegate:'GPU',cameraResolution:'480p'};

test('named profiles use the current HUDYAT definition, including after a main-project change',()=>{
 const current={...baseline,endAbsentMs:700};
 assert.deepEqual(importSettings({profile:'baseline',endAbsentMs:200},current,enhanced),current);
 assert.deepEqual(importSettings({settings:{profile:'enhanced'}},current,enhanced),enhanced);
});
test('custom Copy results imports preserve nested defaults and do not mutate the profiles',()=>{
 const custom=importSettings({settings:{profile:'custom',lock:{minScore:.9},window:{agree:3},alphabetHands:1}},baseline,enhanced);
 assert.deepEqual(custom.lock,{...baseline.lock,minScore:.9});
 assert.equal(custom.window.agree,3);assert.equal(custom.window.everyMs,500);assert.equal(custom.alphabetHands,1);
 assert.equal(baseline.lock.minScore,undefined);assert.equal(baseline.window.agree,2);
});
test('invalid camera choices, nonfinite values, and inconsistent windows fail before capture',()=>{
 for(const patch of [{cameraResolution:'1080p'},{delegate:'other'},{alphabetTracking:'false'},{endAbsentMs:Infinity},
  {lock:{minScore:2}},{window:{minSpanMs:3000}},{earlyAgree:1.5}]){
  assert.throws(()=>importSettings({profile:'custom',...patch},baseline,enhanced));
 }
});

import { spawn } from 'node:child_process';
import { resolve } from 'node:path';
import { createInterface } from 'node:readline';
import { defineConfig } from 'vite';
import { physicsPython } from './scripts/runtime.mjs';

// The local web development server uses the same line protocol as the desktop host.
function attachInference(server) {
  let worker;
  function launch() {
    const child=spawn(physicsPython(),[resolve('training/live.py')],{windowsHide:true,stdio:['pipe','pipe','inherit']});
    const queue=[];
    const lines=createInterface({input:child.stdout});
    lines.on('line',line=>queue.shift()?.resolve(line));
    function fail(error) {
      if(worker?.child===child) worker=undefined;
      for(const pending of queue.splice(0)) pending.reject(error);
    }
    child.on('error',fail);
    child.stdin.on('error',fail);
    child.on('exit',()=>fail(new Error('本地物理引擎已退出，请检查推理环境后重试')));
    return {child,queue};
  }
  server.httpServer?.once('close',()=>worker?.child.stdin.end());
  server.middlewares.use('/__inference',async(req,res)=>{
    res.setHeader('Content-Type','application/json; charset=utf-8');
    res.setHeader('Cache-Control','no-store');
    const origin=req.headers.origin;
    if(req.method!=='POST' || !req.headers['content-type']?.startsWith('application/json') ||
       (origin && !(URL.canParse(origin) && new URL(origin).host===req.headers.host))) {
      res.statusCode=403;res.end('{}');return;
    }
    try {
      let body='';
      for await(const part of req) {
        body+=part;
        if(body.length>4096) throw new Error('请求过长');
      }
      const message=JSON.parse(body);
      worker??=launch();
      const result=await new Promise((resolve,reject)=>{
        worker.queue.push({resolve,reject});
        worker.child.stdin.write(JSON.stringify(message)+'\n');
      });
      res.end(result);
    } catch(error) {
      res.statusCode=500;res.end(JSON.stringify({ok:false,error:error.message}));
    }
  });
}

export default defineConfig({
  plugins:[{name:'collaboration-inference',configureServer:attachInference,configurePreviewServer:attachInference}],
  server:{host:'127.0.0.1',strictPort:true,watch:{ignored:[
    '**/.inference-env/**','**/.training/**','**/assets/**','**/src-tauri/target/**','**/release/**'
  ]}},
});

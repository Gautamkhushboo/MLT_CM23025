const video=document.getElementById("video");
const canvas=document.getElementById("canvas");
const ctx=canvas.getContext("2d");

const objectText=document.getElementById("object");
const confidenceText=document.getElementById("confidence");
const fpsText=document.getElementById("fps");

let model;
let frames=0;
let lastTime=performance.now();

async function startCamera(){

const stream=await navigator.mediaDevices.getUserMedia({video:true});
video.srcObject=stream;

}

async function loadModel(){

model=await cocoSsd.load();
detect();

}

async function detect(){

canvas.width=video.videoWidth;
canvas.height=video.videoHeight;

const predictions=await model.detect(video);

ctx.clearRect(0,0,canvas.width,canvas.height);

predictions.forEach(p=>{

const [x,y,w,h]=p.bbox;

ctx.strokeStyle="#ff69b4";
ctx.lineWidth=3;

ctx.strokeRect(x,y,w,h);

ctx.fillStyle="#ff69b4";
ctx.fillText(p.class,x,y-5);

objectText.innerText="Object: "+p.class;
confidenceText.innerText="Confidence: "+(p.score*100).toFixed(1)+"%";

});

frames++;

const now=performance.now();

if(now-lastTime>=1000){

fpsText.innerText="FPS: "+frames;
frames=0;
lastTime=now;

}

requestAnimationFrame(detect);

}

startCamera();

video.addEventListener("loadeddata",loadModel);
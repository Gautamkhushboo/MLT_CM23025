const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const modeText = document.getElementById("mode");
const singleInput = document.getElementById("singleInput");
const multiInput = document.getElementById("multiInput");

let detector;
let singleMode = true;

const MIN_SCORE = 0.3;

async function setupCamera(){
const stream = await navigator.mediaDevices.getUserMedia({video:true});
video.srcObject = stream;

return new Promise(resolve=>{
video.onloadedmetadata=()=>{
video.play();
resolve(video);
};
});
}

// 🎯 Switch Mode UI
function switchMode(){
singleMode = !singleMode;

modeText.innerText = singleMode ? "Mode: Single Pose" : "Mode: Multi Pose";

if(singleMode){
singleInput.style.display = "block";
multiInput.style.display = "none";
}else{
singleInput.style.display = "none";
multiInput.style.display = "block";
}
}

// 🧠 Pose Detection
function detectPoseType(k){

const lw=k[9], rw=k[10], ls=k[5], rs=k[6];

if(!lw || !rw || lw.score<MIN_SCORE || rw.score<MIN_SCORE){
return "Detecting...";
}

if(lw.y < ls.y && rw.y < rs.y){
return "Hands Up";
}

if(Math.abs(lw.y-ls.y)<40 && Math.abs(rw.y-rs.y)<40){
return "T Pose";
}

return "Standing";
}

// 🎨 Draw Skeleton
function drawSkeleton(k){

const pairs=[[5,7],[7,9],[6,8],[8,10],[5,6],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16]];

pairs.forEach(([a,b])=>{
if(k[a].score>MIN_SCORE && k[b].score>MIN_SCORE){
ctx.beginPath();
ctx.moveTo(k[a].x,k[a].y);
ctx.lineTo(k[b].x,k[b].y);
ctx.strokeStyle="lime";
ctx.lineWidth=2;
ctx.stroke();
}
});
}

// Draw points
function drawPoints(k){
k.forEach(p=>{
if(p.score>MIN_SCORE){
ctx.beginPath();
ctx.arc(p.x,p.y,4,0,2*Math.PI);
ctx.fillStyle="red";
ctx.fill();
}
});
}

// UI Box
function drawUI(count){

ctx.fillStyle="rgba(0,0,0,0.6)";
ctx.fillRect(10,10,250,60);

ctx.fillStyle="cyan";
ctx.font="14px Arial";
ctx.fillText(singleMode?"Single Mode":"Multi Mode",20,25);

ctx.fillStyle="white";
ctx.fillText("People: "+count,20,45);
}

// MAIN
async function run(){

detector = await poseDetection.createDetector(
poseDetection.SupportedModels.MoveNet,
{modelType: poseDetection.movenet.modelType.MULTIPOSE_LIGHTNING}
);

await setupCamera();

canvas.width=640;
canvas.height=480;

async function detect(){

ctx.drawImage(video,0,0,640,480);

let poses;

if(singleMode){
poses = await detector.estimatePoses(video,{maxPoses:1});
}else{
poses = await detector.estimatePoses(video,{maxPoses:2});
}

// Names
let name1 = document.getElementById("name1").value || "Person";
let name1m = document.getElementById("name1_multi").value || "Person 1";
let name2m = document.getElementById("name2_multi").value || "Person 2";

poses.forEach((p,i)=>{

const k = p.keypoints;

drawSkeleton(k);
drawPoints(k);

const nose = k[0];

if(nose.score > MIN_SCORE){

let name = singleMode ? name1 : (i===0?name1m:name2m);
let pose = detectPoseType(k);

// 🎯 Attractive Label
ctx.fillStyle="rgba(0,0,0,0.7)";
ctx.fillRect(nose.x-40,nose.y-40,100,40);

ctx.fillStyle="yellow";
ctx.font="bold 14px Arial";
ctx.fillText(name, nose.x-35, nose.y-20);

ctx.fillStyle="white";
ctx.fillText(pose, nose.x-35, nose.y-5);

}

});

drawUI(poses.length);

requestAnimationFrame(detect);
}

detect();
}

run();
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const counterText = document.getElementById("counter");

let detector;
let squatCount = 0;
let stage = "UP";

// 🔥 smoothing + stability
let angleHistory = [];
const SMOOTH_SIZE = 5;
let stableFrames = 0;

async function setupCamera(){
const stream = await navigator.mediaDevices.getUserMedia({
video:{width:640,height:480}
});
video.srcObject = stream;

return new Promise(resolve=>{
video.onloadedmetadata=()=>{
video.play();
resolve(video);
};
});
}

// Angle calculation
function calculateAngle(a,b,c){
const ab=[a.x-b.x,a.y-b.y];
const cb=[c.x-b.x,c.y-b.y];

const dot = ab[0]*cb[0] + ab[1]*cb[1];
const mag1 = Math.sqrt(ab[0]*ab[0]+ab[1]*ab[1]);
const mag2 = Math.sqrt(cb[0]*cb[0]+cb[1]*cb[1]);

let angle = Math.acos(dot/(mag1*mag2));
return angle * 180 / Math.PI;
}

// Smooth angle
function smoothAngle(angle){
angleHistory.push(angle);

if(angleHistory.length > SMOOTH_SIZE){
angleHistory.shift();
}

return angleHistory.reduce((a,b)=>a+b,0)/angleHistory.length;
}

// Draw
function drawSkeleton(keypoints){
const pairs=[[5,7],[7,9],[6,8],[8,10],[5,6],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16]];

pairs.forEach(([a,b])=>{
const kp1=keypoints[a];
const kp2=keypoints[b];

if(kp1.score>0.3 && kp2.score>0.3){
ctx.beginPath();
ctx.moveTo(kp1.x,kp1.y);
ctx.lineTo(kp2.x,kp2.y);
ctx.strokeStyle="lime";
ctx.lineWidth=2;
ctx.stroke();
}
});
}

function drawKeypoints(keypoints){
keypoints.forEach(kp=>{
if(kp.score>0.3){
ctx.beginPath();
ctx.arc(kp.x,kp.y,4,0,2*Math.PI);
ctx.fillStyle="red";
ctx.fill();
}
});
}

// UI
function drawUI(angle, feedback){

ctx.fillStyle="rgba(0,0,0,0.7)";
ctx.fillRect(10,10,260,100);

ctx.fillStyle="white";
ctx.font="16px Arial";

ctx.fillText("Squats: " + squatCount, 20, 30);
ctx.fillText("Stage: " + stage, 20, 55);
ctx.fillText("Angle: " + angle.toFixed(1), 20, 80);

// Center feedback
ctx.textAlign="center";

if(feedback==="Good Squat"){
ctx.fillStyle="lime";
ctx.font="bold 40px Arial";
ctx.fillText("GOOD SQUAT", canvas.width/2, canvas.height/2);
}
else if(feedback==="Go Lower"){
ctx.fillStyle="orange";
ctx.font="bold 35px Arial";
ctx.fillText("GO LOWER", canvas.width/2, canvas.height/2);
}
else{
ctx.fillStyle="red";
ctx.font="bold 35px Arial";
ctx.fillText("STAND UP", canvas.width/2, canvas.height/2);
}

ctx.textAlign="start";
}

async function run(){

const model = poseDetection.SupportedModels.MoveNet;

detector = await poseDetection.createDetector(model,{
modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING
});

await setupCamera();

canvas.width=640;
canvas.height=480;

async function detect(){

ctx.drawImage(video,0,0,640,480);

const poses = await detector.estimatePoses(video);

if(poses.length > 0){

const k = poses[0].keypoints;

// 🔥 BOTH LEGS
const leftAngle = calculateAngle(k[11],k[13],k[15]);
const rightAngle = calculateAngle(k[12],k[14],k[16]);

// Average
let angle = (leftAngle + rightAngle)/2;

// Smooth
angle = smoothAngle(angle);

let feedback = "";

// 🔥 Stability check
if(angle < 100){
stableFrames++;
}else{
stableFrames = 0;
}

// DOWN
if(angle < 100 && stage === "UP" && stableFrames > 3){
stage = "DOWN";
}

// UP + count
if(angle > 160 && stage === "DOWN"){
stage = "UP";
squatCount++;
}

// Feedback
if(angle > 160){
feedback = "Stand Up";
}
else if(angle > 100){
feedback = "Go Lower";
}
else{
feedback = "Good Squat";
}

drawSkeleton(k);
drawKeypoints(k);
drawUI(angle, feedback);

counterText.innerText = "Squats: " + squatCount;

}

requestAnimationFrame(detect);
}

detect();
}

run();
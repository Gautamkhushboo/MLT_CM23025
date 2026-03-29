const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

let detector;

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

function drawKeypoints(keypoints){

let count = 0;

keypoints.forEach(kp=>{

if(kp.score > 0.3){

count++;

ctx.beginPath();
ctx.arc(kp.x,kp.y,6,0,2*Math.PI);
ctx.fillStyle="red";
ctx.fill();

}

});

return count;
}

function drawSkeleton(keypoints){

const pairs = [
[5,7],[7,9],
[6,8],[8,10],
[5,6],
[5,11],[6,12],
[11,12],
[11,13],[13,15],
[12,14],[14,16]
];

pairs.forEach(([i,j])=>{

const kp1 = keypoints[i];
const kp2 = keypoints[j];

if(kp1.score>0.3 && kp2.score>0.3){

ctx.beginPath();
ctx.moveTo(kp1.x,kp1.y);
ctx.lineTo(kp2.x,kp2.y);
ctx.strokeStyle="lime";
ctx.lineWidth=3;
ctx.stroke();

}

});

}

function drawUI(text, keypointsCount, confidence){

// Background box
ctx.fillStyle = "rgba(0,0,0,0.6)";
ctx.fillRect(10,10,260,90);

// Text
ctx.fillStyle = "white";
ctx.font = "16px Arial";

ctx.fillText("Status: " + text, 20, 35);
ctx.fillText("Keypoints: " + keypointsCount, 20, 60);
ctx.fillText("Confidence: " + confidence.toFixed(2), 20, 85);

}

async function run(){

const model = poseDetection.SupportedModels.MoveNet;

detector = await poseDetection.createDetector(model,{
modelType:poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING
});

await setupCamera();

canvas.width = 640;
canvas.height = 480;

async function detect(){

ctx.drawImage(video,0,0,640,480);

const poses = await detector.estimatePoses(video);

if(poses.length > 0){

const keypoints = poses[0].keypoints;

// Draw skeleton + joints
const count = drawKeypoints(keypoints);
drawSkeleton(keypoints);

// Confidence (average)
let total = 0;
keypoints.forEach(k=> total += k.score);
const avgConfidence = total / keypoints.length;

// UI text
drawUI("Person Detected", count, avgConfidence);

}else{

drawUI("No Person", 0, 0);

}

requestAnimationFrame(detect);

}

detect();
}

run();
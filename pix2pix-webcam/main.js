let gPose, gModel, gIsRunning;

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.src = src;
  });
}

async function main() {
  console.log("main");
  gModel = await tf.loadGraphModel("./model/model.json");
  gImage = await loadImage("./img/dancer-pose.jpeg");

  // Setup pose detection
  gPose = new Pose({
    locateFile: (file) => {
      return `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`;
    },
  });
  gPose.setOptions({
    modelComplexity: 1,
    smoothLandmarks: true,
    enableSegmentation: true,
    smoothSegmentation: true,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
  gPose.onResults(onPose);

  startCamera();
}

function startCamera() {
  const videoElement = document.getElementById("webcam");
  const camera = new Camera(videoElement, {
    onFrame: async () => {
      await gPose.send({ image: videoElement });
    },
    width: 512,
    height: 512,
  });
  camera.start();
}

async function onPose(results) {
  console.log("onPose", results);
  const drawCanvas = document.getElementById("input");
  const drawCtx = drawCanvas.getContext("2d");
  drawCtx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
  drawConnectors(drawCtx, results.poseLandmarks, POSE_CONNECTIONS, { color: "#00FF00", lineWidth: 4 });
  executePix2pix(drawCanvas);
}

async function executePix2pix(image) {
  console.log("executePix2pix");

  let tensor = tf.expandDims(tf.browser.fromPixels(image), 0);
  tensor = tf.cast(tensor, "float32");
  // Normalize values between -1 and 1
  tensor = tensor.div(tf.scalar(127)).sub(tf.scalar(1));

  let result = await gModel.execute(tensor);
  // Convert results back to 0-1 range
  result = result.mul(tf.scalar(0.5)).add(tf.scalar(0.5));

  const resultCanvas = document.getElementById("output");
  await tf.browser.toPixels(result.squeeze(), resultCanvas);
}

main();

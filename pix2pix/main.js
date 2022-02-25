let gModel, gIsRunning;

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
  const pose = new Pose({
    locateFile: (file) => {
      return `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`;
    },
  });
  pose.setOptions({
    modelComplexity: 1,
    smoothLandmarks: true,
    enableSegmentation: true,
    smoothSegmentation: true,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
  pose.onResults(onPose);
  pose.send({ image: gImage });
  // Convert the dancer to a pose image

  //await generateGrid();
}

async function onPose(results) {
  console.log("onPose", results);

  const drawCanvas = document.createElement("canvas");
  drawCanvas.width = gImage.width;
  drawCanvas.height = gImage.height;
  document.querySelector("#container").appendChild(drawCanvas);
  const drawCtx = drawCanvas.getContext("2d");
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

  const resultCanvas = document.createElement("canvas");
  resultCanvas.width = gImage.width;
  resultCanvas.height = gImage.height;
  document.querySelector("#container").appendChild(resultCanvas);

  await tf.browser.toPixels(result.squeeze(), resultCanvas);
}

main();

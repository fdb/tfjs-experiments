const TFHUB_BASE_URL = "https://tfhub.dev/tensorflow/tfjs-model/deeplab";
const MODEL_NAME = "cityscapes";

const LABELS_CITYSCAPE = [
  "road",
  "sidewalk",
  "building",
  "wall",
  "fence",
  "pole",
  "traffic light",
  "traffic sign",
  "vegetation",
  "terrain",
  "sky",
  "person",
  "rider",
  "car",
  "truck",
  "bus",
  "train",
  "motorcycle",
  "bicycle",
];

const CROP_SIZE = 513;
let shouldStop = false;

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.src = src;
  });
}

async function imageToTensor(image) {
  return tf.tidy(() => {
    const tensor = tf.browser.fromPixels(image);
    const [height, width] = tensor.shape;
    const resizeRatio = CROP_SIZE / Math.max(width, height);
    const targetHeight = Math.round(height * resizeRatio);
    const targetWidth = Math.round(width * resizeRatio);
    const resizedImage = tf.image.resizeBilinear(tensor, [
      targetHeight,
      targetWidth,
    ]);
    return tf.expandDims(resizedImage);
  });
}

function modelUrl(modelName = "cityscapes", quantizationBytes = 4) {
  const modelUrl =
    quantizationBytes === 4
      ? `${modelName}/1/default/1/model.json`
      : `${modelName}/1/quantized/${quantizationBytes}/1/model.json`;
  return `${TFHUB_BASE_URL}/${modelUrl}?tfjs-format=file`;
}

function createVideo(url) {
  return new Promise((resolve) => {
    const video = document.createElement("video");
    video.src = url;
    video.loop = true;
    video.muted = true;
    // video.play();
    // Find the video frame rate.
    const listener = video.addEventListener("canplay", () => {
      video.removeEventListener("canplay", listener);
      resolve(video);
    });
    video.currentTime = 0;
    // video.pause();
  });
}

function copyToCanvas(src, srcCtx, dst, dstCtx) {
  const srcData = srcCtx.getImageData(0, 0, src.width, src.height);
  const dstData = dstCtx.getImageData(0, 0, dst.width, dst.height);
  const srcPixels = srcData.data;
  const dstPixels = dstData.data;
  //   const srcData = imageData.data;
  for (let i = 0; i < srcPixels.length; i += 4) {
    const r = srcPixels[i];
    const g = srcPixels[i + 1];
    const b = srcPixels[i + 2];
    // let a = data[i + 3];

    if (r === 0 && g === 0 && b === 0) {
      srcPixels[i] = dstPixels[i];
      srcPixels[i + 1] = dstPixels[i + 1];
      srcPixels[i + 2] = dstPixels[i + 2];
      //   data[i + 3] = 0;
    }
  }
  dstCtx.putImageData(srcData, 0, 0);

  // animationCtx.globalCompositeOperation = "lighten";
  //animationCanvas.
  // animationCtx.drawImage(canvas, 0, 0);

  //   const ctx = dst.getContext("2d");
  //   ctx.drawImage(src, 0, 0);
}

async function main() {
  let time = 0;
  const model = await tf.loadGraphModel(modelUrl(), 1);
  const selectedLabel = LABELS_CITYSCAPE.findIndex(
    (label) => label === "person"
  );
  const video = await createVideo("../assets/nyc-walk.mp4");

  const canvas = document.getElementById("output");
  const ctx = canvas.getContext("2d");

  const animationCanvas = document.getElementById("animation");
  const animationCtx = animationCanvas.getContext("2d");

  for (;;) {
    console.log(`Time: ${time.toFixed(2)}`);

    await processFrame(
      model,
      selectedLabel,
      video,
      canvas,
      ctx,
      animationCanvas,
      animationCtx
    );

    copyToCanvas(canvas, ctx, animationCanvas, animationCtx);

    // Advance one frame.
    time += 1 / 29.97;
    // time += 0.2;
    video.currentTime = time;

    if (shouldStop) break;
    await new Promise((resolve) => {
      const listener = video.addEventListener("canplay", () => {
        video.removeEventListener("canplay", listener);
        resolve();
      });
    });
  }
}

async function processFrame(
  model,
  selectedLabel,
  video,
  canvas,
  ctx,
  animationCanvas,
  animationCtx
) {
  const tensorImage = await imageToTensor(video);
  const resultWidth = tensorImage.shape[2];
  const resultHeight = tensorImage.shape[1];
  if (canvas.width !== resultWidth || canvas.height !== resultHeight) {
    canvas.width = resultWidth;
    canvas.height = resultHeight;
    animationCanvas.width = resultWidth;
    animationCanvas.height = resultHeight;
    animationCtx.fillRect(0, 0, animationCanvas.width, animationCanvas.height);
  }

  const filtered = await tf.tidy(() => {
    const data = tf.cast(tensorImage, "int32");
    const result = tf.squeeze(model.execute(data));
    // Now filter out all pixels where type = selectedLabel.
    const filtered = tf.where(
      tf.equal(result, tf.scalar(selectedLabel, "int32")),
      tf.scalar(255, "int32"),
      tf.scalar(0, "int32")
    );
    return filtered;
  });
  await tf.browser.toPixels(filtered, canvas);
  filtered.dispose();

  ctx.globalCompositeOperation = "darken";
  ctx.drawImage(video, 0, 0, resultWidth, resultHeight);
  ctx.globalCompositeOperation = "source-over";
}

main();
document.querySelector("#stop").addEventListener("click", () => {
  shouldStop = true;
});

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

async function main() {
  const label = LABELS_CITYSCAPE.findIndex((label) => label === "person");
  const image = await loadImage("../assets/nyc-walk-04.jpeg");
  const tensorImage = await imageToTensor(image);
  const resultWidth = tensorImage.shape[2];
  const resultHeight = tensorImage.shape[1];
  const model = await tf.loadGraphModel(modelUrl());
  //   console.log(model);
  tf.tidy(() => {
    const data = tf.cast(tensorImage, "int32");
    data.shape; // [1, 289, 513, 3]
    console.log(data.rank);
    const result = tf.squeeze(model.execute(data));
    console.log(result.shape, result.dtype);
    // tf.browser.toPixels(result, document.getElementById("output"));
    // Now filter out all pixels with type = 1 or something.
    const filtered = tf.where(
      tf.equal(result, tf.scalar(label, "int32")),
      tf.scalar(255, "int32"),
      tf.scalar(0, "int32")
    );
    const canvas = document.getElementById("output");
    canvas.width = resultWidth;
    canvas.height = resultHeight;
    const ctx = canvas.getContext("2d");
    tf.browser.toPixels(filtered, canvas).then(() => {
      //   ctx.globalCompositeOperation = "source-in";
      ctx.globalCompositeOperation = "darken";

      ctx.drawImage(image, 0, 0, resultWidth, resultHeight);
    });

    // Draw the original image over the white pixels.

    // ctx.blendMode = "screen";
    // ctx.globalCompositeOperation = "source-atop";
    // Now use the filtered image to create a mask.
    // const mask = tf.image.resizeBilinear(filtered, [

    // const imageData = result.dataSync();
    // Convert the resulting 1-dimensional tensor back to an image.
    // const imageData = result.dataSync();
    // Reshape the 1D tensor to a rank-3 tensor with shape [height, width, 3]
    // const result = tf.tensor3d(imageData, [image.height, image.width, 3]);
    // const outputImage = tf.tensor(result, [513, 513, 3]);

    // const canvas = document.getElementById("output");
    // canvas.width = 513;
    // canvas.height = 513;
    // const ctx = canvas.getContext("2d");
    // const imageTensor = outputImage.toFloat();
    // const imageDataTensor = imageTensor.mul(tf.scalar(255)).toInt();
    // const imageDataArray = imageDataTensor.dataSync();
    // const imageDataTyped = new Uint8ClampedArray(imageDataArray);
    // const imageToDraw = new ImageData(imageDataTyped, 513, 513);
    // ctx.putImageData(imageToDraw, 0, 0);

    // console.log(result);
  });
}

main();

// tf.tidy(() => {
//   const data = tf.cast(toInputTensor(input), "int32");
//   tf.squeeze(model.execute(data));
// });

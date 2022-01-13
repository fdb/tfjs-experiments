// function lerp(tensor1, tensor2, t) {
//     return tensor1.mul(1 - t).add(tensor2.mul(t));
//   }

async function loadImage(url) {
  const img = new Image();
  img.src = url;
  return new Promise((resolve) => {
    img.onload = () => resolve(img);
  });
}

async function main() {
  const img = await loadImage("../assets/nyc-walk-01.jpeg");
  const model = await tf.loadGraphModel("../models/mirnet/model.json");
  const tensor = tf.browser.fromPixels(img);
  // Convert image pixels to float
  const tensor2 = tensor.toFloat().div(tf.scalar(255));
  //const resized = tf.image.resizeBilinear(tensor, [224, 224]);
  const input = tf.expandDims(tensor2);
  const result = await model.execute(input);
  const canvas = document.createElement("canvas");
  document.getElementById("container").appendChild(canvas);
  await tf.browser.toPixels(result.squeeze(), canvas);
}

main();

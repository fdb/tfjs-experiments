function lerp(tensor1, tensor2, t) {
  return tensor1.mul(1 - t).add(tensor2.mul(t));
}

let gModel, gIsRunning;

async function main() {
  gModel = await tf.loadGraphModel("https://tfhub.dev/google/progan-128/1", {
    fromTFHub: true,
  });

  await generateGrid();
}

async function generateGrid() {
  if (!gModel) return;
  if (gIsRunning) return;
  gIsRunning = true;
  document.getElementById("container").innerHTML = "";

  // Keep a consistent distance between the two sampled latent spces.
  const space1 = tf.randomNormal([1, 512]);
  const dir = tf.randomNormal([1, 512]).norm();
  const space2 = space1.add(dir.mul(0.2));
  //   const space2 = tf.randomNormal([1, 512]);
  let t = 0;
  const amount = 20 * 20;
  for (let i = 0; i < amount; i += 1) {
    const t = i * (1.0 / amount);
    const space = lerp(space1, space2, t);
    const result = await gModel.execute(space);
    const canvas = document.createElement("canvas");
    document.getElementById("container").appendChild(canvas);
    await tf.browser.toPixels(result.squeeze(), canvas);
  }
  gIsRunning = false;
}

main();
window.addEventListener("keydown", async (e) => {
  if (e.key === "r") {
    await generateGrid();
  }
});

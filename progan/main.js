function lerp(tensor1, tensor2, t) {
  return tensor1.mul(1 - t).add(tensor2.mul(t));
}

async function main() {
  const model = await tf.loadGraphModel(
    "https://tfhub.dev/google/progan-128/1",
    {
      fromTFHub: true,
    }
  );

  console.log(model);
  const space1 = tf.randomNormal([1, 512]);
  const space2 = tf.randomNormal([1, 512]);
  let t = 0;
  const amount = 20 * 20;
  for (let i = 0; i < amount; i += 1) {
    const t = i * (1.0 / amount);
    const space = lerp(space1, space2, t);
    const result = await model.execute(space);
    const canvas = document.createElement("canvas");
    document.getElementById("container").appendChild(canvas);
    await tf.browser.toPixels(result.squeeze(), canvas);
  }
}

main();

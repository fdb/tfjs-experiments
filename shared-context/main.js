let gModel;

async function main() {
  gModel = await tf.loadGraphModel("https://tfhub.dev/google/progan-128/1", {
    fromTFHub: true,
  });

  // Get the tf backend context.
  const backend = tf.backend();
  const gl = backend.gpgpu.gl;

  const space = tf.randomNormal([1, 512]);

  const result = await gModel.execute(space);
  const squeezed = result.squeeze();

  // Get the backing texture from the tensor
  const tex = backend.texData.data.get(result.dataId);

  backend.canvas.width = squeezed.shape[0];
  backend.canvas.height = squeezed.shape[1];

  debugger;

  // Transfer the texture to another canvas?
  // or create a new OffscreenCanvas with the same context?
  backend.canvas;

  const offscreen = new OffscreenCanvas(512, 512);
  var bitmapOne = offscreen.transferToImageBitmap();
  one.transferFromImageBitmap(bitmapOne);

  //   await generateGrid();
  tf.b;
}

main();

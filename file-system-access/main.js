async function canvasToBlob(canvas) {
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (!blob) {
        reject(new Error("Canvas is empty"));
        return;
      }
      resolve(blob);
    });
  });
}

document.getElementById("load-image").addEventListener("click", async () => {
  console.log("clic");
  const [fileHandle] = await window.showOpenFilePicker();
  //   const tttt = await chooseFileSystemEntries();
  const file = await fileHandle.getFile();
  const contents = await file.arrayBuffer();
  var arrayBufferView = new Uint8Array(contents);
  var blob = new Blob([arrayBufferView], { type: "image/jpeg" });
  const blobUrl = URL.createObjectURL(blob);
  const img = document.getElementById("result");
  img.src = blobUrl;
  //   const img = await tf.browser.fromPixels(contents);
});

const canvas = document.getElementById("c");
const ctx = canvas.getContext("2d");
ctx.fillRect(20, 20, 100, 100);

document
  .getElementById("save-image-sequence")
  .addEventListener("click", async () => {
    const dirHandle = await window.showDirectoryPicker();
    // const stream = await dirHandle.createWritable();
    const fileHandle = await dirHandle.getFileHandle("test-01.jpg", {
      create: true,
    });

    const stream = await fileHandle.createWritable();
    const blob = await canvasToBlob(canvas);
    await stream.write(blob);
    await stream.close();
  });

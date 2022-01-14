const TFHUB_BASE_URL = "https://tfhub.dev/tensorflow/tfjs-model/deeplab";
const MODEL_NAME = "ade20k";
const CROP_SIZE = 513;

// prettier-ignore
const LABELS_ADE20K = [ 
  'background', 'wall',         'building',   'sky',         'floor',
  'tree',       'ceiling',      'road',       'bed',         'windowpane',
  'grass',      'cabinet',      'sidewalk',   'person',      'earth',
  'door',       'table',        'mountain',   'plant',       'curtain',
  'chair',      'car',          'water',      'painting',    'sofa',
  'shelf',      'house',        'sea',        'mirror',      'rug',
  'field',      'armchair',     'seat',       'fence',       'desk',
  'rock',       'wardrobe',     'lamp',       'bathtub',     'railing',
  'cushion',    'base',         'box',        'column',      'signboard',
  'chest',      'counter',      'sand',       'sink',        'skyscraper',
  'fireplace',  'refrigerator', 'grandstand', 'path',        'stairs',
  'runway',     'case',         'pool',       'pillow',      'screen',
  'stairway',   'river',        'bridge',     'bookcase',    'blind',
  'coffee',     'toilet',       'flower',     'book',        'hill',
  'bench',      'countertop',   'stove',      'palm',        'kitchen',
  'computer',   'swivel',       'boat',       'bar',         'arcade',
  'hovel',      'bus',          'towel',      'light',       'truck',
  'tower',      'chandelier',   'awning',     'streetlight', 'booth',
  'television', 'airplane',     'dirt',       'apparel',     'pole',
  'land',       'bannister',    'escalator',  'ottoman',     'bottle',
  'buffet',     'poster',       'stage',      'van',         'ship',
  'fountain',   'conveyer',     'canopy',     'washer',      'plaything',
  'swimming',   'stool',        'barrel',     'basket',      'waterfall',
  'tent',       'bag',          'minibike',   'cradle',      'oven',
  'ball',       'food',         'step',       'tank',        'trade',
  'microwave',  'pot',          'animal',     'bicycle',     'lake',
  'dishwasher', 'screen',       'blanket',    'sculpture',   'hood',
  'sconce',     'vase',         'traffic',    'tray',        'ashcan',
  'fan',        'pier',         'screen',     'plate',       'monitor',
  'bulletin',   'shower',       'radiator',   'glass',       'clock',
  'flag',
];

function modelUrl(modelName, quantizationBytes = 4) {
  const modelUrl =
    quantizationBytes === 4
      ? `${modelName}/1/default/1/model.json`
      : `${modelName}/1/quantized/${quantizationBytes}/1/model.json`;
  return `${TFHUB_BASE_URL}/${modelUrl}?tfjs-format=file`;
}

function padZeroes(n, width) {
  n = n + "";
  return n.length >= width ? n : new Array(width - n.length + 1).join("0") + n;
}

async function saveCanvasToFile(canvas, filename) {
  const fileHandle = await gDirHandle.getFileHandle(filename, {
    create: true,
  });
  const stream = await fileHandle.createWritable();
  const blob = await canvasToBlob(canvas);
  await stream.write(blob);
  await stream.close();
}

class VideoPlayer {
  constructor(url) {
    this.video = document.createElement("video");
    this.video.src = url;
    this.video.loop = true;
    this.video.muted = true;
    // video.play();
    // Find the video frame rate.
    this.video.currentTime = 0;
    this.frame = 0;
    this.frameRate = 29.97;
    // document.body.appendChild(this.video);
  }

  load() {
    return new Promise((resolve, reject) => {
      const listener = this.video.addEventListener("canplay", () => {
        this.frameCount = Math.ceil(this.video.duration * this.frameRate);
        this.video.removeEventListener("canplay", listener);
        resolve(this.video);
      });
    });
  }

  async nextFrame() {
    this.frame++;
    this.video.currentTime = this.frame / this.frameRate;
    await new Promise((resolve) => {
      const listener = this.video.addEventListener("canplay", () => {
        this.video.removeEventListener("canplay", listener);
        resolve();
      });
    });
  }

  async toTensor() {
    return tf.tidy(() => {
      const tensor = tf.browser.fromPixels(this.video);
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
}

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

let gPlayer;
let gSelectedLabel = LABELS_ADE20K.findIndex((label) => label === "person");
let gCanvas = document.getElementById("output");
let gModel;
let gStopRequested = false;
let gDirHandle;

async function loadModel() {
  document.getElementById("status").innerHTML = "Loading model...";
  gModel = await tf.loadGraphModel(modelUrl(MODEL_NAME), 4);
  document.getElementById("status").innerHTML = "Model loaded.";
}
createVideoPlayer = async (url) => {
  gPlayer = new VideoPlayer(url);
  await gPlayer.load();
  return gPlayer;
};

async function processVideo() {
  gDirHandle = await window.showDirectoryPicker();

  await createVideoPlayer("../assets/nyc-walk.mp4");
  window.requestAnimationFrame(animate);
}

async function animate() {
  document.getElementById(
    "status"
  ).innerHTML = `Frame ${gPlayer.frame} of ${gPlayer.frameCount}`;

  const tensorImage = await gPlayer.toTensor();
  const filtered = await tf.tidy(() => {
    const data = tf.cast(tensorImage, "int32");
    const result = tf.squeeze(gModel.execute(data));
    // Now filter out all pixels where type = selectedLabel.
    // const filtered = tf.where(
    //   tf.equal(result, tf.scalar(gSelectedLabel, "int32")),
    //   tf.scalar(255, "int32"),
    //   tf.scalar(0, "int32")
    // );
    return result;
  });
  await tf.browser.toPixels(filtered, gCanvas);
  filtered.dispose();
  tensorImage.dispose();

  await saveCanvasToFile(gCanvas, `mask-${padZeroes(gPlayer.frame, 5)}.png`);

  const videoCanvas = document.getElementById("video-canvas");
  if (
    videoCanvas.width !== gCanvas.width ||
    videoCanvas.height !== gCanvas.height
  ) {
    videoCanvas.width = gCanvas.width;
    videoCanvas.height = gCanvas.height;
  }
  const videoCtx = videoCanvas.getContext("2d");
  videoCtx.drawImage(
    gPlayer.video,
    0,
    0,
    videoCanvas.width,
    videoCanvas.height
  );
  await saveCanvasToFile(
    videoCanvas,
    `frame-${padZeroes(gPlayer.frame, 5)}.png`
  );

  if (gPlayer.frame < gPlayer.frameCount) {
    if (gStopRequested) {
      gPlayer.frame = 0;
      gStopRequested = false;
    } else {
      await gPlayer.nextFrame();
      window.requestAnimationFrame(animate);
    }
  }
}

// function popuplateSelection() {
//   const html = LABELS_ADE20K.map((label, index) => {
//     return `<option value="${label}">${label}</option>`;
//   }).join("\n");
//   document.getElementById("category-select").innerHTML = html;
// }

// document.getElementById("category-select").addEventListener("change", (e) => {
//   const labelName = e.target.value;
//   gSelectedLabel = LABELS_ADE20K.findIndex((label) => label === labelName);
//   // gSelectedLabel = e.target.value;
// });

document
  .getElementById("save-sequence")
  .addEventListener("click", processVideo);

document.getElementById("stop").addEventListener("click", () => {
  gStopRequested = true;
});

// popuplateSelection();

loadModel();

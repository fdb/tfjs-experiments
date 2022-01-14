const videoEl = document.querySelector("video");
async function main() {
  const mediaSource = new MediaSource();
  videoEl.src = URL.createObjectURL(mediaSource);
  mediaSource.addEventListener("sourceopen", sourceOpen);

  // Convert buffer to MediaStream

  //   const videoTrack = stream.getVideoTracks()[0];

  //   const stream = await getUserMedia({ video: true });
}

async function sourceOpen(e) {
  URL.revokeObjectURL(videoEl.src);
  const mime = 'video/mp4; codecs="avc1.42E01E, mp4a.40.2"';
  const mediaSource = e.target;
  const sourceBuffer = mediaSource.addSourceBuffer(mime);
  const res = await fetch("../assets/nyc-walk.mp4");
  const buffer = await res.arrayBuffer();
  sourceBuffer.addEventListener("updateend", () => {
    if (!sourceBuffer.updating && mediaSource.readyState === "open") {
      mediaSource.endOfStream();
      videoEl.play();
    }
  });
  sourceBuffer.appendBuffer(buffer);
}

main();

// const stream = await getUserMedia({ video: true });
// const videoTrack = stream.getVideoTracks()[0];

// const videoDecoder = new VideoDecoder({
//   output: processVideo,
//   error: onEncoderError,
// });

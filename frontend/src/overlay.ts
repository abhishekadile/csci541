// Turns a binary mask PNG (white = defect) into a transparent RGBA overlay of the SAME pixel size,
// so it can be stacked on the image inside one transformed stage and stay pixel-aligned.

export type RGB = [number, number, number];

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("mask could not be decoded"));
    img.src = src;
  });
}

export async function maskToOverlayUrl(maskSrc: string, color: RGB): Promise<string> {
  const img = await loadImage(maskSrc);
  const canvas = document.createElement("canvas");
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("canvas unavailable");
  ctx.drawImage(img, 0, 0);
  const data = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const px = data.data;
  for (let i = 0; i < px.length; i += 4) {
    const on = px[i] > 127;
    px[i] = color[0];
    px[i + 1] = color[1];
    px[i + 2] = color[2];
    px[i + 3] = on ? 255 : 0;
  }
  ctx.putImageData(data, 0, 0);
  return canvas.toDataURL("image/png");
}

export const pngDataUrl = (b64: string) => `data:image/png;base64,${b64}`;

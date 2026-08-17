declare module "dom-to-image" {
  interface CaptureOptions {
    width?: number;
    height?: number;
    quality?: number;
  }

  interface DomToImage {
    toPng(node: Node, options?: CaptureOptions): Promise<string>;
  }

  const domToImage: DomToImage;

  export default domToImage;
}

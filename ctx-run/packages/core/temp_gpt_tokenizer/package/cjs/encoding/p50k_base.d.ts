import { GptEncoding } from '../GptEncoding.js';
export * from '../specialTokens.js';
declare const api: GptEncoding;
declare const decode: (inputTokensToDecode: Iterable<number>) => string, decodeAsyncGenerator: (inputTokensToDecode: AsyncIterable<number>) => AsyncGenerator<string, void>, decodeGenerator: (inputTokensToDecode: Iterable<number>) => Generator<string, void, void>, encode: (lineToEncode: string, encodeOptions?: import("../GptEncoding.js").EncodeOptions) => number[], encodeGenerator: (lineToEncode: string, encodeOptions?: import("../GptEncoding.js").EncodeOptions) => Generator<number[], number, undefined>, isWithinTokenLimit: (input: string | Iterable<import("../GptEncoding.js").ChatMessage>, tokenLimit: number) => false | number;
export { decode, decodeAsyncGenerator, decodeGenerator, encode, encodeGenerator, isWithinTokenLimit, };
export default api;

export type RawBytePairRanks = readonly (string | readonly number[])[];
export interface BytePairEncodingConfig {
    mergeableBytePairRanks: RawBytePairRanks;
    specialTokenMapping?: Map<string, number>;
    tokenSplitRegex: RegExp;
}
export declare const decoder: TextDecoder;
export declare class BytePairEncodingCore {
    readonly mergeableBytePairRankCount: number;
    /**
     * an array where the index is the BPE rank,
     * and the value is the string or the array of bytes that it decodes to
     * it may contain holes if token is unused
     */
    private bytePairRankDecoder;
    private bytePairNonUtfRankDecoder;
    private bytePairNonUtfSortedEncoder;
    /**
     * a reverse map of the bytePairRankDecoder,
     * where the key is the string and the value is the rank
     * values that cannot be represented as a string are present in `bytePairNonUtfSortedEncoder`
     */
    private bytePairStringRankEncoder;
    private tokenSplitRegex;
    private specialTokensEncoder;
    private specialTokensDecoder;
    private specialTokenPatternRegex;
    private textEncoder;
    constructor({ mergeableBytePairRanks: bytePairEncoder, specialTokenMapping: specialTokenEncoder, tokenSplitRegex, }: BytePairEncodingConfig);
    encodeNativeGenerator(text: string, allowedSpecial?: Set<string>): Generator<number[], number, undefined>;
    encodeNative(text: string, allowedSpecial?: Set<string>): number[];
    decodeNativeGenerator(tokens: Iterable<number>): Generator<Uint8Array | string, void, void>;
    decodeNative(tokens: Iterable<number>): string;
    decodeNativeAsyncIterable(tokens: AsyncIterable<number>): AsyncGenerator<Uint8Array | string>;
    private getBpeRankFromString;
    private getBpeRankFromStringOrThrow;
    private getBpeRankFromBytes;
    private getBpeRankFromBytesOrThrow;
    private binarySearch;
    private findNextSpecialStartIndex;
    private tryDecodeToken;
    private bytePairEncode;
    private bytePairMerge;
}

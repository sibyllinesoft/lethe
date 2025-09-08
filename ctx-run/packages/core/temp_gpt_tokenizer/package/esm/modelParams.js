import { Cl100KBase } from './encodingParams/Cl100KBase.js';
import { O200KBase } from './encodingParams/O200KBase.js';
import { P50KBase } from './encodingParams/P50KBase.js';
import { P50KEdit } from './encodingParams/P50KEdit.js';
import { R50KBase } from './encodingParams/R50KBase.js';
export const tokenSplitRegex = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;
export function getEncodingParams(encodingName, getMergeableRanks) {
    const mergeableBytePairRanks = getMergeableRanks(encodingName);
    switch (encodingName.toLowerCase()) {
        case 'r50k_base':
            return R50KBase(mergeableBytePairRanks);
        case 'p50k_base':
            return P50KBase(mergeableBytePairRanks);
        case 'p50k_edit':
            return P50KEdit(mergeableBytePairRanks);
        case 'cl100k_base':
            return Cl100KBase(mergeableBytePairRanks);
        case 'o200k_base':
            return O200KBase(mergeableBytePairRanks);
        default:
            throw new Error(`Unknown encoding name: ${encodingName}`);
    }
}
export async function getModelParamsAsync(encodingName, getMergeableRanks) {
    const mergeableBytePairRanks = await getMergeableRanks(encodingName);
    return getEncodingParams(encodingName, () => mergeableBytePairRanks);
}
//# sourceMappingURL=modelParams.js.map
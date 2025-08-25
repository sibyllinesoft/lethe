import { tokenSplitRegex } from '../modelParams.js';
import { EndOfText, FimMiddle, FimPrefix, FimSuffix } from '../specialTokens.js';
export function P50KEdit(mergeableBytePairRanks) {
    const specialTokenMapping = new Map([
        [EndOfText, 50_256],
        [FimPrefix, 50_281],
        [FimMiddle, 50_282],
        [FimSuffix, 50_283],
    ]);
    return {
        tokenSplitRegex,
        mergeableBytePairRanks,
        specialTokenMapping,
    };
}
//# sourceMappingURL=P50KEdit.js.map
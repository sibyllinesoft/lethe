import { tokenSplitRegex } from '../modelParams.js';
import { EndOfText } from '../specialTokens.js';
export function P50KBase(mergeableBytePairRanks) {
    return {
        expectedVocabularySize: 50_281,
        tokenSplitRegex,
        mergeableBytePairRanks,
        specialTokenMapping: new Map([[EndOfText, 50_256]]),
    };
}
//# sourceMappingURL=P50KBase.js.map
import { tokenSplitRegex } from '../modelParams.js';
import { EndOfText } from '../specialTokens.js';
export function R50KBase(mergeableBytePairRanks) {
    return {
        expectedVocabularySize: 50_257,
        tokenSplitRegex,
        mergeableBytePairRanks,
        specialTokenMapping: new Map([[EndOfText, 50_256]]),
    };
}
//# sourceMappingURL=R50KBase.js.map
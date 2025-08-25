"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.R50KBase = R50KBase;
const modelParams_js_1 = require("../modelParams.js");
const specialTokens_js_1 = require("../specialTokens.js");
function R50KBase(mergeableBytePairRanks) {
    return {
        expectedVocabularySize: 50_257,
        tokenSplitRegex: modelParams_js_1.tokenSplitRegex,
        mergeableBytePairRanks,
        specialTokenMapping: new Map([[specialTokens_js_1.EndOfText, 50_256]]),
    };
}
//# sourceMappingURL=R50KBase.js.map
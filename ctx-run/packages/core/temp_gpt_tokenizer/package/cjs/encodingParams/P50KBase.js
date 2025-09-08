"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.P50KBase = P50KBase;
const modelParams_js_1 = require("../modelParams.js");
const specialTokens_js_1 = require("../specialTokens.js");
function P50KBase(mergeableBytePairRanks) {
    return {
        expectedVocabularySize: 50_281,
        tokenSplitRegex: modelParams_js_1.tokenSplitRegex,
        mergeableBytePairRanks,
        specialTokenMapping: new Map([[specialTokens_js_1.EndOfText, 50_256]]),
    };
}
//# sourceMappingURL=P50KBase.js.map
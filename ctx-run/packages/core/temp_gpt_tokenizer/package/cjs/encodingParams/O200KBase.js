"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.O200KBase = O200KBase;
const specialTokens_js_1 = require("../specialTokens.js");
function O200KBase(mergeableBytePairRanks) {
    const specialTokenMapping = new Map([
        [specialTokens_js_1.EndOfText, 199_999],
        [specialTokens_js_1.FimPrefix, 200_000],
        [specialTokens_js_1.FimMiddle, 200_001],
        [specialTokens_js_1.FimSuffix, 200_002],
        [specialTokens_js_1.ImStart, 200_003],
        [specialTokens_js_1.ImEnd, 200_004],
        [specialTokens_js_1.ImSep, 200_005],
        [specialTokens_js_1.EndOfPrompt, 200_006],
    ]);
    return {
        tokenSplitRegex: /(?:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+/giu,
        mergeableBytePairRanks,
        specialTokenMapping,
    };
}
//# sourceMappingURL=O200KBase.js.map
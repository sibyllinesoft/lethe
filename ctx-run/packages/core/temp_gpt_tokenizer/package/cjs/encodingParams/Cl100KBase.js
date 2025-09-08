"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.Cl100KBase = Cl100KBase;
const specialTokens_js_1 = require("../specialTokens.js");
function Cl100KBase(mergeableBytePairRanks) {
    const specialTokenMapping = new Map([
        [specialTokens_js_1.EndOfText, 100_257],
        [specialTokens_js_1.FimPrefix, 100_258],
        [specialTokens_js_1.FimMiddle, 100_259],
        [specialTokens_js_1.FimSuffix, 100_260],
        [specialTokens_js_1.ImStart, 100_264],
        [specialTokens_js_1.ImEnd, 100_265],
        [specialTokens_js_1.ImSep, 100_266],
        [specialTokens_js_1.EndOfPrompt, 100_276],
    ]);
    return {
        tokenSplitRegex: /(?:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+/giu,
        mergeableBytePairRanks,
        specialTokenMapping,
    };
}
//# sourceMappingURL=Cl100KBase.js.map
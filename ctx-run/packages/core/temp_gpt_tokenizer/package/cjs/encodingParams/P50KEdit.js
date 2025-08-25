"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.P50KEdit = P50KEdit;
const modelParams_js_1 = require("../modelParams.js");
const specialTokens_js_1 = require("../specialTokens.js");
function P50KEdit(mergeableBytePairRanks) {
    const specialTokenMapping = new Map([
        [specialTokens_js_1.EndOfText, 50_256],
        [specialTokens_js_1.FimPrefix, 50_281],
        [specialTokens_js_1.FimMiddle, 50_282],
        [specialTokens_js_1.FimSuffix, 50_283],
    ]);
    return {
        tokenSplitRegex: modelParams_js_1.tokenSplitRegex,
        mergeableBytePairRanks,
        specialTokenMapping,
    };
}
//# sourceMappingURL=P50KEdit.js.map
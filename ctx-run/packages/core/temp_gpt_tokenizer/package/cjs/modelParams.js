"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.tokenSplitRegex = void 0;
exports.getEncodingParams = getEncodingParams;
exports.getModelParamsAsync = getModelParamsAsync;
const Cl100KBase_js_1 = require("./encodingParams/Cl100KBase.js");
const O200KBase_js_1 = require("./encodingParams/O200KBase.js");
const P50KBase_js_1 = require("./encodingParams/P50KBase.js");
const P50KEdit_js_1 = require("./encodingParams/P50KEdit.js");
const R50KBase_js_1 = require("./encodingParams/R50KBase.js");
exports.tokenSplitRegex = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;
function getEncodingParams(encodingName, getMergeableRanks) {
    const mergeableBytePairRanks = getMergeableRanks(encodingName);
    switch (encodingName.toLowerCase()) {
        case 'r50k_base':
            return (0, R50KBase_js_1.R50KBase)(mergeableBytePairRanks);
        case 'p50k_base':
            return (0, P50KBase_js_1.P50KBase)(mergeableBytePairRanks);
        case 'p50k_edit':
            return (0, P50KEdit_js_1.P50KEdit)(mergeableBytePairRanks);
        case 'cl100k_base':
            return (0, Cl100KBase_js_1.Cl100KBase)(mergeableBytePairRanks);
        case 'o200k_base':
            return (0, O200KBase_js_1.O200KBase)(mergeableBytePairRanks);
        default:
            throw new Error(`Unknown encoding name: ${encodingName}`);
    }
}
async function getModelParamsAsync(encodingName, getMergeableRanks) {
    const mergeableBytePairRanks = await getMergeableRanks(encodingName);
    return getEncodingParams(encodingName, () => mergeableBytePairRanks);
}
//# sourceMappingURL=modelParams.js.map
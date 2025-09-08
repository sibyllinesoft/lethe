"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __exportStar = (this && this.__exportStar) || function(m, exports) {
    for (var p in m) if (p !== "default" && !Object.prototype.hasOwnProperty.call(exports, p)) __createBinding(exports, m, p);
};
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.openDb = openDb;
exports.migrate = migrate;
exports.loadVectorExtension = loadVectorExtension;
const better_sqlite3_1 = __importDefault(require("better-sqlite3"));
const fs_1 = require("fs");
const path_1 = require("path");
function openDb(path) {
    const db = new better_sqlite3_1.default(path);
    db.pragma('journal_mode = WAL');
    return db;
}
async function migrate(db) {
    const schema = (0, fs_1.readFileSync)((0, path_1.join)(__dirname, 'schema.sql'), 'utf-8');
    db.exec(schema);
}
async function loadVectorExtension(db) {
    try {
        // Try to load sqlite-vec
        db.loadExtension('sqlite-vec');
        console.log('sqlite-vec extension loaded.');
        return true;
    }
    catch (error) {
        console.warn('Could not load sqlite-vec extension. Trying sqlite-vss.');
        try {
            // Try to load sqlite-vss
            db.loadExtension('sqlite-vss');
            console.log('sqlite-vss extension loaded.');
            return true;
        }
        catch (error) {
            console.warn('Could not load any SQLite vector extension. Vector search will be disabled.');
            return false;
        }
    }
}
__exportStar(require("./chunks"), exports);
//# sourceMappingURL=migrate.js.map
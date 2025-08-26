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
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __exportStar = (this && this.__exportStar) || function(m, exports) {
    for (var p in m) if (p !== "default" && !Object.prototype.hasOwnProperty.call(exports, p)) __createBinding(exports, m, p);
};
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.openDb = openDb;
exports.migrate = migrate;
exports.loadVectorExtension = loadVectorExtension;
exports.upsertConfig = upsertConfig;
exports.getConfig = getConfig;
exports.getState = getState;
exports.setState = setState;
exports.insertMessages = insertMessages;
exports.insertChunks = insertChunks;
exports.insertEmbeddings = insertEmbeddings;
exports.vectorSearch = vectorSearch;
exports.refreshVectorIndex = refreshVectorIndex;
exports.getChunksBySession = getChunksBySession;
exports.getDFIdf = getDFIdf;
exports.getChunkById = getChunkById;
const better_sqlite3_1 = __importDefault(require("better-sqlite3"));
const fs_1 = require("fs");
const path_1 = require("path");
// Milestone 2: Adaptive Planning Policy - New Exports
__exportStar(require("./atoms-types.js"), exports);
__exportStar(require("./adaptive-planning.js"), exports);
__exportStar(require("./adaptive-atoms-db.js"), exports);
__exportStar(require("./grid-search.js"), exports);
__exportStar(require("./atoms-db.js"), exports);
__exportStar(require("./entity-extraction.js"), exports);
__exportStar(require("./session-idf.js"), exports);
// Core functions as per package contract
function openDb(path) {
    return new better_sqlite3_1.default(path);
}
async function migrate(db) {
    const schemaPath = (0, path_1.join)(__dirname, '..', 'schema.sql');
    const schema = (0, fs_1.readFileSync)(schemaPath, 'utf-8');
    db.exec(schema);
}
async function loadVectorExtension(db) {
    try {
        // Try loading sqlite-vec first
        db.exec("SELECT load_extension('sqlite-vec')");
        return true;
    }
    catch {
        try {
            // Try sqlite-vss as fallback
            db.exec("SELECT load_extension('sqlite-vss')");
            return true;
        }
        catch {
            return false;
        }
    }
}
// CRUD helpers
function upsertConfig(db, key, value) {
    const stmt = db.prepare('INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)');
    stmt.run(key, JSON.stringify(value));
}
function getConfig(db, key) {
    const stmt = db.prepare('SELECT value FROM config WHERE key = ?');
    const result = stmt.get(key);
    return result ? JSON.parse(result.value) : undefined;
}
function getState(db, sessionId) {
    const stmt = db.prepare('SELECT json FROM state WHERE session_id = ?');
    const result = stmt.get(sessionId);
    return result ? JSON.parse(result.json) : null;
}
function setState(db, sessionId, json) {
    const stmt = db.prepare('INSERT OR REPLACE INTO state (session_id, json) VALUES (?, ?)');
    stmt.run(sessionId, JSON.stringify(json));
}
function insertMessages(db, messages) {
    const stmt = db.prepare('INSERT OR REPLACE INTO messages (id, session_id, turn, role, text, ts, meta) VALUES (?, ?, ?, ?, ?, ?, ?)');
    const transaction = db.transaction((msgs) => {
        for (const msg of msgs) {
            stmt.run(msg.id, msg.session_id, msg.turn, msg.role, msg.text, msg.ts, msg.meta ? JSON.stringify(msg.meta) : null);
        }
    });
    transaction(messages);
}
function insertChunks(db, chunks) {
    const stmt = db.prepare('INSERT OR REPLACE INTO chunks (id, message_id, session_id, offset_start, offset_end, kind, text, tokens) VALUES (?, ?, ?, ?, ?, ?, ?, ?)');
    const transaction = db.transaction((chks) => {
        for (const chunk of chks) {
            stmt.run(chunk.id, chunk.message_id, chunk.session_id, chunk.offset_start, chunk.offset_end, chunk.kind, chunk.text, chunk.tokens);
        }
    });
    transaction(chunks);
}
function insertEmbeddings(db, rows) {
    const stmt = db.prepare('INSERT OR REPLACE INTO embeddings (chunk_id, dim, vec) VALUES (?, ?, ?)');
    const transaction = db.transaction((embeddings) => {
        for (const row of embeddings) {
            stmt.run(row.chunk_id, row.dim, row.vec);
        }
    });
    transaction(rows);
}
// Global vector index instances
let nativeVectorSupported = null;
let wasmIndex = null; // HnswWasm instance (imported dynamically)
async function vectorSearch(db, qVec, k) {
    // Check if native vector extension is available
    if (nativeVectorSupported === null) {
        nativeVectorSupported = await loadVectorExtension(db);
    }
    if (nativeVectorSupported) {
        // Use native vector extension if available
        try {
            const stmt = db.prepare('SELECT chunk_id as docId, distance as score FROM vec_index WHERE vec MATCH ? ORDER BY distance LIMIT ?');
            const results = stmt.all(qVec, k);
            // Convert distance to similarity score (1 - distance for cosine)
            return results.map(r => ({ docId: r.docId, score: 1 - r.score }));
        }
        catch (error) {
            console.warn('Native vector search failed, falling back to WASM:', error);
            nativeVectorSupported = false;
        }
    }
    // Fall back to WASM implementation
    if (!wasmIndex) {
        // Import WASM dynamically to avoid circular dependencies
        const { HnswWasm } = await Promise.resolve().then(() => __importStar(require('@lethe/wasm')));
        wasmIndex = new HnswWasm();
        // Load all embeddings into WASM index
        const stmt = db.prepare('SELECT chunk_id, vec FROM embeddings');
        const embeddings = stmt.all();
        for (const embedding of embeddings) {
            const vector = new Float32Array(embedding.vec.buffer, embedding.vec.byteOffset, embedding.vec.byteLength / 4);
            wasmIndex.addVector(embedding.chunk_id, vector);
        }
        console.log(`Loaded ${embeddings.length} embeddings into WASM vector index`);
    }
    return wasmIndex.search(qVec, k);
}
// Function to refresh vector index when new embeddings are added
function refreshVectorIndex() {
    if (wasmIndex) {
        wasmIndex.clear();
        wasmIndex = null;
    }
}
function getChunksBySession(db, sessionId) {
    const stmt = db.prepare('SELECT * FROM chunks WHERE session_id = ?');
    return stmt.all(sessionId);
}
function getDFIdf(db, sessionId) {
    const stmt = db.prepare('SELECT * FROM dfidf WHERE session_id = ?');
    return stmt.all(sessionId);
}
function getChunkById(db, chunkId) {
    const stmt = db.prepare('SELECT * FROM chunks WHERE id = ?');
    const result = stmt.get(chunkId);
    return result || null;
}
//# sourceMappingURL=index.js.map
import Database from 'better-sqlite3';
import { readFileSync } from 'fs';
import { join } from 'path';
import { DB } from './types';

export function openDb(path: string): DB {
  const db = new Database(path);
  db.pragma('journal_mode = WAL');
  return db;
}

export async function migrate(db: DB): Promise<void> {
  const schema = readFileSync(join(__dirname, 'schema.sql'), 'utf-8');
  db.exec(schema);
}

export async function loadVectorExtension(db: DB): Promise<boolean> {
  try {
    // Try to load sqlite-vec
    db.loadExtension('sqlite-vec');
    console.log('sqlite-vec extension loaded.');
    return true;
  } catch (error) {
    console.warn('Could not load sqlite-vec extension. Trying sqlite-vss.');
    try {
      // Try to load sqlite-vss
      db.loadExtension('sqlite-vss');
      console.log('sqlite-vss extension loaded.');
      return true;
    } catch (error) {
      console.warn('Could not load any SQLite vector extension. Vector search will be disabled.');
      return false;
    }
  }
}

export * from './chunks';

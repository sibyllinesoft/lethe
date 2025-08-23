import Database from 'better-sqlite3';

export type DB = Database.Database;

export interface Message {
    id: string;
    session_id: string;
    turn: number;
    role: string;
    text: string;
    ts: number;
    meta?: any;
}

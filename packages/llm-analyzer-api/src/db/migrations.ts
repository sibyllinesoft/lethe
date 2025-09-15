import type { Database } from 'better-sqlite3';

export interface Migration {
  version: number;
  description: string;
  up: (db: Database.Database) => void;
  down?: (db: Database.Database) => void;
}

export const migrations: Migration[] = [
  {
    version: 1,
    description: 'Initial schema',
    up: (db) => {
      // Schema is already applied in database.ts
      // This is just a placeholder for the initial version
    }
  }
  // Add future migrations here
];

export class MigrationManager {
  constructor(private db: Database.Database) {}

  getCurrentVersion(): number {
    try {
      const result = this.db.prepare('SELECT MAX(version) as version FROM schema_migrations').get() as { version: number | null };
      return result.version || 0;
    } catch {
      return 0;
    }
  }

  migrate(): void {
    const currentVersion = this.getCurrentVersion();
    
    for (const migration of migrations) {
      if (migration.version > currentVersion) {
        console.log(`Running migration ${migration.version}: ${migration.description}`);
        
        try {
          migration.up(this.db);
          
          // Record migration
          this.db.prepare('INSERT INTO schema_migrations (version) VALUES (?)').run(migration.version);
          
          console.log(`Migration ${migration.version} completed`);
        } catch (error) {
          console.error(`Migration ${migration.version} failed:`, error);
          throw error;
        }
      }
    }
  }

  rollback(targetVersion: number): void {
    const currentVersion = this.getCurrentVersion();
    
    for (let i = migrations.length - 1; i >= 0; i--) {
      const migration = migrations[i];
      
      if (migration.version > targetVersion && migration.version <= currentVersion) {
        if (!migration.down) {
          throw new Error(`Migration ${migration.version} does not support rollback`);
        }
        
        console.log(`Rolling back migration ${migration.version}: ${migration.description}`);
        
        try {
          migration.down(this.db);
          
          // Remove migration record
          this.db.prepare('DELETE FROM schema_migrations WHERE version = ?').run(migration.version);
          
          console.log(`Rollback ${migration.version} completed`);
        } catch (error) {
          console.error(`Rollback ${migration.version} failed:`, error);
          throw error;
        }
      }
    }
  }
}
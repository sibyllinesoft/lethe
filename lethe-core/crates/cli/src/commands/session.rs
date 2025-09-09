use async_trait::async_trait;
use clap::{Args, Subcommand};
use lethe_shared::Result;
use crate::utils::AppContext;
use super::Command;

#[derive(Debug, Args)]
pub struct SessionCommand {
    #[command(subcommand)]
    action: SessionAction,
}

#[derive(Debug, Subcommand)]
enum SessionAction {
    /// List all sessions
    List {
        /// Limit number of results
        #[arg(long, short, default_value = "10")]
        limit: usize,
    },
    /// Create a new session
    Create {
        /// Session ID to create
        session_id: String,
        /// Optional metadata
        #[arg(long)]
        metadata: Option<String>,
    },
    /// Show session details
    Show {
        /// Session ID to show
        session_id: String,
    },
    /// Delete a session
    Delete {
        /// Session ID to delete
        session_id: String,
        /// Force deletion without confirmation
        #[arg(long)]
        force: bool,
    },
}

#[async_trait]
impl Command for SessionCommand {
    async fn execute(&self, context: &AppContext) -> Result<()> {
        use lethe_infrastructure::{DatabaseManager, PgSessionRepository};
        use std::sync::Arc;

        let db_url = context.database_url.as_ref()
            .ok_or("Database URL is required for session management")?;
        let db_manager = Arc::new(DatabaseManager::new(db_url).await?);
        let session_repo = Arc::new(PgSessionRepository::new(db_manager.pool().clone()));

        match &self.action {
            SessionAction::List { limit } => {
                let sessions = session_repo.find_recent(*limit).await?;
                
                if sessions.is_empty() {
                    if !context.quiet {
                        println!("No sessions found");
                    }
                    return Ok(());
                }

                match &context.output_format {
                    crate::utils::OutputFormat::Json => {
                        println!("{}", serde_json::to_string_pretty(&sessions)?);
                    }
                    _ => {
                        println!("📋 Sessions:");
                        for session in sessions {
                            println!("  🆔 ID: {}", session.id);
                            if let Some(meta) = &session.meta {
                                println!("     📝 Meta: {}", serde_json::to_string(meta)?);
                            }
                            println!("     🕒 Created: {}", session.created_at);
                            println!();
                        }
                    }
                }
            }
            SessionAction::Create { session_id, metadata } => {
                let meta = if let Some(metadata_str) = metadata {
                    Some(serde_json::from_str(metadata_str)?)
                } else {
                    None
                };

                let session = lethe_shared::Session {
                    id: session_id.clone(),
                    created_at: chrono::Utc::now(),
                    updated_at: chrono::Utc::now(),
                    meta,
                };

                session_repo.create(&session).await?;
                
                if !context.quiet {
                    println!("✅ Created session: {}", session_id);
                }
            }
            SessionAction::Show { session_id } => {
                let session = session_repo.find_by_id(session_id).await?
                    .ok_or_else(|| format!("Session not found: {}", session_id))?;

                match &context.output_format {
                    crate::utils::OutputFormat::Json => {
                        println!("{}", serde_json::to_string_pretty(&session)?);
                    }
                    _ => {
                        println!("📋 Session Details:");
                        println!("  🆔 ID: {}", session.id);
                        println!("  🕒 Created: {}", session.created_at);
                        println!("  🔄 Updated: {}", session.updated_at);
                        if let Some(meta) = &session.meta {
                            println!("  📝 Meta: {}", serde_json::to_string_pretty(meta)?);
                        }
                    }
                }
            }
            SessionAction::Delete { session_id, force } => {
                if !force && !context.quiet {
                    use dialoguer::Confirm;
                    if !Confirm::new()
                        .with_prompt(format!("Delete session '{}'?", session_id))
                        .interact()? 
                    {
                        println!("Cancelled");
                        return Ok(());
                    }
                }

                session_repo.delete(session_id).await?;
                
                if !context.quiet {
                    println!("✅ Deleted session: {}", session_id);
                }
            }
        }

        Ok(())
    }
}
use super::Command;
use crate::utils::AppContext;
use async_trait::async_trait;
use clap::{Args, Subcommand};
use lethe_shared::Result;

#[derive(Debug, Args)]
pub struct SessionCommand {
    #[command(subcommand)]
    action: SessionAction,
}

#[derive(Debug, Subcommand)]
enum SessionAction {
    /// List sessions
    List {
        /// Maximum number of sessions to show
        #[arg(long, default_value = "20")]
        limit: usize,
    },
    /// Create a new session with optional metadata (JSON)
    Create {
        /// Session ID to create
        session_id: String,
        /// Optional metadata JSON blob
        #[arg(long)]
        metadata: Option<String>,
    },
    /// Show session details
    Show { session_id: String },
    /// Delete a session
    Delete {
        session_id: String,
        /// Force deletion without confirmation
        #[arg(long)]
        force: bool,
    },
}

#[async_trait]
impl Command for SessionCommand {
    async fn execute(&self, context: &AppContext) -> Result<()> {
        use lethe_infrastructure::{DatabaseManager, PgSessionRepository, SessionRepository};
        use std::sync::Arc;

        let db_url = context.database_url.as_ref().ok_or_else(|| {
            lethe_shared::LetheError::config("Database URL is required for session management")
        })?;

        let db_manager = Arc::new(DatabaseManager::new(db_url).await?);
        let session_repo = Arc::new(PgSessionRepository::new(db_manager.pool().clone()));

        match &self.action {
            SessionAction::List { limit } => {
                let sessions = session_repo
                    .list_sessions(Some(*limit as i32), Some(0))
                    .await?;

                if sessions.is_empty() {
                    println!("No sessions found.");
                    return Ok(());
                }

                match &context.output_format {
                    crate::utils::OutputFormat::Json => {
                        let json = serde_json::to_string_pretty(&sessions)?;
                        println!("{}", json);
                    }
                    _ => {
                        println!("📋 Sessions (showing up to {}):", limit);
                        for session in sessions.iter() {
                            println!("  • {} (updated {})", session.id, session.updated_at);
                        }
                    }
                }
            }
            SessionAction::Create {
                session_id,
                metadata,
            } => {
                let metadata_json = match metadata {
                    Some(raw) => Some(serde_json::from_str(raw)?),
                    None => None,
                };

                let session = lethe_infrastructure::Session {
                    id: session_id.clone(),
                    created_at: chrono::Utc::now(),
                    updated_at: chrono::Utc::now(),
                    metadata: metadata_json,
                };

                session_repo.create_session(&session).await?;
                println!("✅ Created session '{}'.", session_id);
            }
            SessionAction::Show { session_id } => {
                match session_repo.get_session(session_id).await? {
                    Some(session) => match &context.output_format {
                        crate::utils::OutputFormat::Json => {
                            let json = serde_json::to_string_pretty(&session)?;
                            println!("{}", json);
                        }
                        _ => {
                            println!("📋 Session {}", session.id);
                            println!("   • Created: {}", session.created_at);
                            println!("   • Updated: {}", session.updated_at);
                            if let Some(metadata) = session.metadata {
                                println!(
                                    "   • Metadata: {}",
                                    serde_json::to_string_pretty(&metadata)?
                                );
                            }
                        }
                    },
                    None => println!("Session '{}' not found.", session_id),
                }
            }
            SessionAction::Delete { session_id, force } => {
                if !force && !context.quiet {
                    use dialoguer::Confirm;
                    let proceed = Confirm::new()
                        .with_prompt(format!(
                            "Delete session '{}' and related state?",
                            session_id
                        ))
                        .interact()
                        .map_err(|e| {
                            lethe_shared::LetheError::internal(format!("Prompt failed: {}", e))
                        })?;
                    if !proceed {
                        println!("Cancelled");
                        return Ok(());
                    }
                }

                let removed = session_repo.delete_session(session_id).await?;
                if removed {
                    println!("✅ Deleted session '{}'.", session_id);
                } else {
                    println!("Session '{}' did not exist.", session_id);
                }
            }
        }

        Ok(())
    }
}

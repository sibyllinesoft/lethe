use super::Command;
use crate::utils::AppContext;
use async_trait::async_trait;
use clap::{Args, Subcommand};
use lethe_shared::Result;

#[derive(Debug, Args)]
pub struct MessageCommand {
    #[command(subcommand)]
    action: MessageAction,
}

#[derive(Debug, Subcommand)]
enum MessageAction {
    /// List messages for a session
    List {
        /// Session ID (required)
        #[arg(long)]
        session_id: Option<String>,
        /// Number of messages to display
        #[arg(long, default_value = "20")]
        limit: usize,
    },
    /// Show a single message by ID
    Show {
        /// Message ID (UUID)
        message_id: String,
    },
    /// Delete a message by ID
    Delete {
        /// Message ID (UUID)
        message_id: String,
        /// Force deletion without confirmation
        #[arg(long)]
        force: bool,
    },
}

#[async_trait]
impl Command for MessageCommand {
    async fn execute(&self, context: &AppContext) -> Result<()> {
        use lethe_infrastructure::{DatabaseManager, MessageRepository, PgMessageRepository};
        use std::sync::Arc;
        use uuid::Uuid;

        let db_url = context.database_url.as_ref().ok_or_else(|| {
            lethe_shared::LetheError::config("Database URL is required for message management")
        })?;

        let db_manager = Arc::new(DatabaseManager::new(db_url).await?);
        let message_repo = Arc::new(PgMessageRepository::new(db_manager.pool().clone()));

        match &self.action {
            MessageAction::List { session_id, limit } => {
                let session_id = session_id.as_deref().ok_or_else(|| {
                    lethe_shared::LetheError::validation(
                        "session_id",
                        "Parameter is required for message listing",
                    )
                })?;

                let messages = message_repo
                    .get_messages_by_session(session_id, Some(*limit as i32))
                    .await?;

                if messages.is_empty() {
                    println!("No messages found for session '{}'.", session_id);
                    return Ok(());
                }

                match &context.output_format {
                    crate::utils::OutputFormat::Json => {
                        let json = serde_json::to_string_pretty(&messages)?;
                        println!("{}", json);
                    }
                    _ => {
                        println!("📨 Messages for session '{}':", session_id);
                        for message in messages.iter() {
                            println!(
                                "  • {} [{}] — {}",
                                message.id,
                                message.role,
                                truncate(&message.text, 120)
                            );
                        }
                    }
                }
            }
            MessageAction::Show { message_id } => {
                let message_id = Uuid::parse_str(message_id).map_err(|e| {
                    lethe_shared::LetheError::validation(
                        "message_id",
                        format!("Invalid UUID: {}", e),
                    )
                })?;

                match message_repo.get_message(&message_id).await? {
                    Some(message) => match &context.output_format {
                        crate::utils::OutputFormat::Json => {
                            let json = serde_json::to_string_pretty(&message)?;
                            println!("{}", json);
                        }
                        _ => {
                            println!("📨 Message {}", message.id);
                            println!("   • Session: {}", message.session_id);
                            println!("   • Role: {}", message.role);
                            println!("   • Turn: {}", message.turn);
                            println!("   • Timestamp: {}", message.ts);
                            println!("   • Text:\n{}", message.text);
                            if let Some(meta) = &message.meta {
                                println!("   • Metadata: {}", serde_json::to_string_pretty(meta)?);
                            }
                        }
                    },
                    None => println!("Message not found."),
                }
            }
            MessageAction::Delete { message_id, force } => {
                let message_id = Uuid::parse_str(message_id).map_err(|e| {
                    lethe_shared::LetheError::validation(
                        "message_id",
                        format!("Invalid UUID: {}", e),
                    )
                })?;

                if !force && !context.quiet {
                    use dialoguer::Confirm;
                    let proceed = Confirm::new()
                        .with_prompt("Delete this message?")
                        .interact()
                        .map_err(|e| {
                            lethe_shared::LetheError::internal(format!("Prompt failed: {}", e))
                        })?;
                    if !proceed {
                        println!("Cancelled");
                        return Ok(());
                    }
                }

                let removed = message_repo.delete_message(&message_id).await?;
                if removed {
                    println!("✅ Deleted message {}", message_id);
                } else {
                    println!("Message {} did not exist", message_id);
                }
            }
        }

        Ok(())
    }
}

fn truncate(text: &str, limit: usize) -> String {
    if text.len() <= limit {
        text.to_string()
    } else {
        format!("{}...", &text[..limit.saturating_sub(3)])
    }
}

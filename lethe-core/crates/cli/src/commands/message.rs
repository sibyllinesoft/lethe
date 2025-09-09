use async_trait::async_trait;
use clap::{Args, Subcommand};
use lethe_shared::Result;
use crate::utils::AppContext;
use super::Command;

#[derive(Debug, Args)]
pub struct MessageCommand {
    #[command(subcommand)]
    action: MessageAction,
}

#[derive(Debug, Subcommand)]
enum MessageAction {
    /// List messages
    List {
        /// Session ID to filter by
        #[arg(long)]
        session_id: Option<String>,
        /// Limit number of results
        #[arg(long, short, default_value = "10")]
        limit: usize,
    },
    /// Show message details
    Show {
        /// Message ID to show
        message_id: String,
    },
    /// Delete a message
    Delete {
        /// Message ID to delete
        message_id: String,
        /// Force deletion without confirmation
        #[arg(long)]
        force: bool,
    },
}

#[async_trait]
impl Command for MessageCommand {
    async fn execute(&self, context: &AppContext) -> Result<()> {
        use lethe_infrastructure::{DatabaseManager, PgMessageRepository};
        use std::sync::Arc;

        let db_url = context.database_url.as_ref()
            .ok_or("Database URL is required for message management")?;
        let db_manager = Arc::new(DatabaseManager::new(db_url).await?);
        let message_repo = Arc::new(PgMessageRepository::new(db_manager.pool().clone()));

        match &self.action {
            MessageAction::List { session_id, limit } => {
                let messages = if let Some(session_id) = session_id {
                    message_repo.find_by_session(session_id).await?
                } else {
                    message_repo.find_recent(*limit).await?
                };

                match &context.output_format {
                    crate::utils::OutputFormat::Json => {
                        println!("{}", serde_json::to_string_pretty(&messages)?);
                    }
                    _ => {
                        if messages.is_empty() {
                            println!("No messages found");
                        } else {
                            println!("📨 Messages ({})", messages.len());
                            for msg in messages {
                                println!("  🆔 {}: {} - {}", msg.id, msg.role, 
                                    if msg.text.len() > 60 { 
                                        format!("{}...", &msg.text[..57]) 
                                    } else { 
                                        msg.text.clone() 
                                    }
                                );
                            }
                        }
                    }
                }
            }
            MessageAction::Show { message_id } => {
                let message_uuid = uuid::Uuid::parse_str(message_id)?;
                let message = message_repo.find_by_id(&message_uuid).await?
                    .ok_or_else(|| format!("Message not found: {}", message_id))?;

                match &context.output_format {
                    crate::utils::OutputFormat::Json => {
                        println!("{}", serde_json::to_string_pretty(&message)?);
                    }
                    _ => {
                        println!("📨 Message: {}", message.id);
                        println!("   Session: {}", message.session_id);
                        println!("   Role: {}", message.role);
                        println!("   Turn: {}", message.turn);
                        println!("   Time: {}", message.ts);
                        println!("   Text:\n{}", message.text);
                        if let Some(meta) = &message.meta {
                            println!("   Meta: {}", serde_json::to_string_pretty(meta)?);
                        }
                    }
                }
            }
            MessageAction::Delete { message_id, force } => {
                if !force && !context.quiet {
                    use dialoguer::Confirm;
                    if !Confirm::new()
                        .with_prompt(format!("Delete message '{}'?", message_id))
                        .interact()? 
                    {
                        println!("Cancelled");
                        return Ok(());
                    }
                }

                let message_uuid = uuid::Uuid::parse_str(message_id)?;
                message_repo.delete(&message_uuid).await?;
                
                if !context.quiet {
                    println!("✅ Deleted message: {}", message_id);
                }
            }
        }

        Ok(())
    }
}
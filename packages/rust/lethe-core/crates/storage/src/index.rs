use crate::bloom::SimpleBloom;
use crate::corpus::ChunkBloomExport;
use lethe_shared::{Chunk, LetheError, Message, RepositoryConfig, Result};
use parquet::{
    column::writer::ColumnWriter,
    data_type::ByteArray,
    file::properties::WriterProperties,
    file::writer::SerializedFileWriter,
    schema::{parser::parse_message_type, types::ColumnPath},
};
use std::{collections::HashSet, fs::File, path::Path, sync::Arc};
use tantivy::{
    schema::{Schema, TantivyDocument, STORED, TEXT},
    Index, Term,
};

pub fn write_parquet_chunks(session_dir: &Path, session_id: &str, chunks: &[Chunk]) -> Result<()> {
    if chunks.is_empty() {
        return Ok(());
    }

    const CHUNK_SCHEMA: &str = "
    message chunk_record {
      REQUIRED BINARY id (UTF8);
      REQUIRED BINARY message_id (UTF8);
      REQUIRED BINARY session_id (UTF8);
      REQUIRED INT32 offset_start;
      REQUIRED INT32 offset_end;
      REQUIRED BINARY kind (UTF8);
      REQUIRED BINARY text (UTF8);
      REQUIRED INT32 tokens;
    }
    ";

    let path = session_dir.join("chunks.parquet");
    let file = File::create(&path).map_err(|e| {
        LetheError::internal(format!(
            "Failed to create parquet file {}: {}",
            path.display(),
            e
        ))
    })?;

    let schema = Arc::new(
        parse_message_type(CHUNK_SCHEMA)
            .map_err(|e| LetheError::internal(format!("Failed to parse chunk schema: {}", e)))?,
    );

    let props = Arc::new(
        WriterProperties::builder()
            .set_column_bloom_filter_enabled(ColumnPath::from(vec!["text".to_string()]), true)
            .build(),
    );

    let mut writer = SerializedFileWriter::new(file, schema, props)
        .map_err(|e| LetheError::internal(format!("Failed to initialise parquet writer: {}", e)))?;

    let ids: Vec<ByteArray> = chunks
        .iter()
        .map(|chunk| ByteArray::from(chunk.id.as_bytes()))
        .collect();
    let message_ids: Vec<ByteArray> = chunks
        .iter()
        .map(|chunk| ByteArray::from(chunk.message_id.to_string().into_bytes()))
        .collect();
    let session_ids: Vec<ByteArray> = chunks
        .iter()
        .map(|_| ByteArray::from(session_id.as_bytes()))
        .collect();
    let offset_start: Vec<i32> = chunks
        .iter()
        .map(|chunk| chunk.offset_start as i32)
        .collect();
    let offset_end: Vec<i32> = chunks.iter().map(|chunk| chunk.offset_end as i32).collect();
    let kinds: Vec<ByteArray> = chunks
        .iter()
        .map(|chunk| ByteArray::from(chunk.kind.as_bytes()))
        .collect();
    let texts: Vec<ByteArray> = chunks
        .iter()
        .map(|chunk| ByteArray::from(chunk.text.as_bytes()))
        .collect();
    let tokens: Vec<i32> = chunks.iter().map(|chunk| chunk.tokens).collect();

    let mut row_group = writer
        .next_row_group()
        .map_err(|e| LetheError::internal(format!("Failed to create row group: {}", e)))?;
    let mut column_index = 0;

    while let Some(mut column_writer) = row_group
        .next_column()
        .map_err(|e| LetheError::internal(format!("Failed to obtain column writer: {}", e)))?
    {
        let untyped = column_writer.untyped();
        match column_index {
            0 => {
                if let ColumnWriter::ByteArrayColumnWriter(ref mut writer) = untyped {
                    writer.write_batch(&ids, None, None).map_err(|e| {
                        LetheError::internal(format!("Failed to write id column: {}", e))
                    })?;
                } else {
                    return Err(LetheError::internal(
                        "Unexpected column writer type for chunk id",
                    ));
                }
            }
            1 => {
                if let ColumnWriter::ByteArrayColumnWriter(ref mut writer) = untyped {
                    writer.write_batch(&message_ids, None, None).map_err(|e| {
                        LetheError::internal(format!("Failed to write message_id column: {}", e))
                    })?;
                } else {
                    return Err(LetheError::internal(
                        "Unexpected column writer type for message_id",
                    ));
                }
            }
            2 => {
                if let ColumnWriter::ByteArrayColumnWriter(ref mut writer) = untyped {
                    writer.write_batch(&session_ids, None, None).map_err(|e| {
                        LetheError::internal(format!("Failed to write session_id column: {}", e))
                    })?;
                } else {
                    return Err(LetheError::internal(
                        "Unexpected column writer type for session_id",
                    ));
                }
            }
            3 => {
                if let ColumnWriter::Int32ColumnWriter(ref mut writer) = untyped {
                    writer.write_batch(&offset_start, None, None).map_err(|e| {
                        LetheError::internal(format!("Failed to write offset_start column: {}", e))
                    })?;
                } else {
                    return Err(LetheError::internal(
                        "Unexpected column writer type for offset_start",
                    ));
                }
            }
            4 => {
                if let ColumnWriter::Int32ColumnWriter(ref mut writer) = untyped {
                    writer.write_batch(&offset_end, None, None).map_err(|e| {
                        LetheError::internal(format!("Failed to write offset_end column: {}", e))
                    })?;
                } else {
                    return Err(LetheError::internal(
                        "Unexpected column writer type for offset_end",
                    ));
                }
            }
            5 => {
                if let ColumnWriter::ByteArrayColumnWriter(ref mut writer) = untyped {
                    writer.write_batch(&kinds, None, None).map_err(|e| {
                        LetheError::internal(format!("Failed to write kind column: {}", e))
                    })?;
                } else {
                    return Err(LetheError::internal(
                        "Unexpected column writer type for kind",
                    ));
                }
            }
            6 => {
                if let ColumnWriter::ByteArrayColumnWriter(ref mut writer) = untyped {
                    writer.write_batch(&texts, None, None).map_err(|e| {
                        LetheError::internal(format!("Failed to write text column: {}", e))
                    })?;
                } else {
                    return Err(LetheError::internal(
                        "Unexpected column writer type for text",
                    ));
                }
            }
            7 => {
                if let ColumnWriter::Int32ColumnWriter(ref mut writer) = untyped {
                    writer.write_batch(&tokens, None, None).map_err(|e| {
                        LetheError::internal(format!("Failed to write tokens column: {}", e))
                    })?;
                } else {
                    return Err(LetheError::internal(
                        "Unexpected column writer type for tokens",
                    ));
                }
            }
            _ => {}
        }

        column_writer
            .close()
            .map_err(|e| LetheError::internal(format!("Failed to close chunk column: {}", e)))?;
        column_index += 1;
    }

    row_group
        .close()
        .map_err(|e| LetheError::internal(format!("Failed to close chunk row group: {}", e)))?;
    writer
        .close()
        .map_err(|e| LetheError::internal(format!("Failed to close chunk writer: {}", e)))?;

    Ok(())
}

pub fn write_parquet_messages(session_dir: &Path, messages: &[Message]) -> Result<()> {
    const MESSAGE_SCHEMA: &str = "
    message message_record {
      REQUIRED BINARY id (UTF8);
      REQUIRED BINARY session_id (UTF8);
      REQUIRED INT32 turn;
      REQUIRED BINARY role (UTF8);
      REQUIRED BINARY text (UTF8);
      OPTIONAL BINARY metadata (UTF8);
      REQUIRED BINARY timestamp (UTF8);
    }
    ";

    if messages.is_empty() {
        let path = session_dir.join("messages.parquet");
        if path.exists() {
            std::fs::remove_file(&path).ok();
        }
        return Ok(());
    }

    let path = session_dir.join("messages.parquet");
    let file = File::create(&path).map_err(|e| {
        LetheError::internal(format!(
            "Failed to create message parquet file {}: {}",
            path.display(),
            e
        ))
    })?;

    let schema = Arc::new(
        parse_message_type(MESSAGE_SCHEMA)
            .map_err(|e| LetheError::internal(format!("Failed to parse message schema: {}", e)))?,
    );

    let props = Arc::new(WriterProperties::builder().build());

    let mut writer = SerializedFileWriter::new(file, schema, props)
        .map_err(|e| LetheError::internal(format!("Failed to initialise message writer: {}", e)))?;

    let ids: Vec<ByteArray> = messages
        .iter()
        .map(|message| ByteArray::from(message.id.to_string().into_bytes()))
        .collect();
    let session_ids: Vec<ByteArray> = messages
        .iter()
        .map(|message| ByteArray::from(message.session_id.as_bytes()))
        .collect();
    let turns: Vec<i32> = messages.iter().map(|message| message.turn).collect();
    let roles: Vec<ByteArray> = messages
        .iter()
        .map(|message| ByteArray::from(message.role.as_bytes()))
        .collect();
    let texts: Vec<ByteArray> = messages
        .iter()
        .map(|message| ByteArray::from(message.text.as_bytes()))
        .collect();
    let meta_values: Vec<Option<ByteArray>> = messages
        .iter()
        .map(|message| {
            message
                .meta
                .as_ref()
                .and_then(|meta| serde_json::to_vec(meta).ok())
                .map(ByteArray::from)
        })
        .collect();
    let meta_packed: Vec<ByteArray> = meta_values
        .iter()
        .filter_map(|value| value.clone())
        .collect();
    let meta_def_levels: Vec<i16> = meta_values
        .iter()
        .map(|value| if value.is_some() { 1 } else { 0 })
        .collect();
    let timestamps: Vec<ByteArray> = messages
        .iter()
        .map(|message| ByteArray::from(message.ts.to_rfc3339().into_bytes()))
        .collect();

    let mut row_group = writer
        .next_row_group()
        .map_err(|e| LetheError::internal(format!("Failed to create message row group: {}", e)))?;
    let mut column_index = 0;

    while let Some(mut column_writer) = row_group.next_column().map_err(|e| {
        LetheError::internal(format!("Failed to obtain message column writer: {}", e))
    })? {
        let untyped = column_writer.untyped();
        match column_index {
            0 => {
                if let ColumnWriter::ByteArrayColumnWriter(ref mut writer) = untyped {
                    writer.write_batch(&ids, None, None).map_err(|e| {
                        LetheError::internal(format!("Failed to write message id column: {}", e))
                    })?;
                }
            }
            1 => {
                if let ColumnWriter::ByteArrayColumnWriter(ref mut writer) = untyped {
                    writer.write_batch(&session_ids, None, None).map_err(|e| {
                        LetheError::internal(format!(
                            "Failed to write message session column: {}",
                            e
                        ))
                    })?;
                }
            }
            2 => {
                if let ColumnWriter::Int32ColumnWriter(ref mut writer) = untyped {
                    writer.write_batch(&turns, None, None).map_err(|e| {
                        LetheError::internal(format!("Failed to write message turn column: {}", e))
                    })?;
                }
            }
            3 => {
                if let ColumnWriter::ByteArrayColumnWriter(ref mut writer) = untyped {
                    writer.write_batch(&roles, None, None).map_err(|e| {
                        LetheError::internal(format!("Failed to write message role column: {}", e))
                    })?;
                }
            }
            4 => {
                if let ColumnWriter::ByteArrayColumnWriter(ref mut writer) = untyped {
                    writer.write_batch(&texts, None, None).map_err(|e| {
                        LetheError::internal(format!("Failed to write message text column: {}", e))
                    })?;
                }
            }
            5 => {
                if let ColumnWriter::ByteArrayColumnWriter(ref mut writer) = untyped {
                    writer
                        .write_batch(&meta_packed, Some(&meta_def_levels), None)
                        .map_err(|e| {
                            LetheError::internal(format!(
                                "Failed to write message metadata column: {}",
                                e
                            ))
                        })?;
                }
            }
            6 => {
                if let ColumnWriter::ByteArrayColumnWriter(ref mut writer) = untyped {
                    writer.write_batch(&timestamps, None, None).map_err(|e| {
                        LetheError::internal(format!(
                            "Failed to write message timestamp column: {}",
                            e
                        ))
                    })?;
                }
            }
            _ => {}
        }

        column_writer
            .close()
            .map_err(|e| LetheError::internal(format!("Failed to close message column: {}", e)))?;
        column_index += 1;
    }

    row_group
        .close()
        .map_err(|e| LetheError::internal(format!("Failed to close message row group: {}", e)))?;
    writer
        .close()
        .map_err(|e| LetheError::internal(format!("Failed to close message writer: {}", e)))?;

    Ok(())
}

fn tokenize_for_bloom(text: &str) -> HashSet<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_lowercase())
        .collect()
}

fn write_chunk_bloom(session_dir: &Path, chunks: &[Chunk]) -> Result<()> {
    if chunks.is_empty() {
        let path = session_dir.join("chunks.bloom");
        if path.exists() {
            std::fs::remove_file(path).ok();
        }
        return Ok(());
    }

    let mut exports = Vec::with_capacity(chunks.len());
    for chunk in chunks {
        let tokens = tokenize_for_bloom(&chunk.text);
        let mut bloom = SimpleBloom::new(tokens.len().max(1), 0.01);
        for token in tokens {
            bloom.insert(&token);
        }
        exports.push(ChunkBloomExport {
            chunk_id: chunk.id.clone(),
            filter: bloom,
        });
    }

    let data = bincode::serialize(&exports)
        .map_err(|e| LetheError::internal(format!("Failed to serialise bloom filters: {}", e)))?;
    std::fs::write(session_dir.join("chunks.bloom"), data)
        .map_err(|e| LetheError::internal(format!("Failed to write bloom filters: {}", e)))?;
    Ok(())
}

fn build_schema() -> Schema {
    let mut builder = Schema::builder();
    builder.add_text_field("session_id", TEXT | STORED);
    builder.add_text_field("repository_path", TEXT | STORED);
    builder.add_text_field("doc_id", TEXT | STORED);
    builder.add_text_field("kind", TEXT | STORED);
    builder.add_text_field("text", TEXT | STORED);
    builder.add_text_field("metadata", STORED);
    builder.build()
}

fn write_tantivy_index(
    session_dir: &Path,
    session_id: &str,
    repo: &RepositoryConfig,
    chunks: &[Chunk],
) -> Result<()> {
    let schema = build_schema();
    let index = if session_dir.join("meta.json").exists() {
        Index::open_in_dir(session_dir)
            .map_err(|e| LetheError::internal(format!("Failed to open taintivy index: {}", e)))?
    } else {
        Index::create_in_dir(session_dir, schema.clone())
            .map_err(|e| LetheError::internal(format!("Failed to create taintivy index: {}", e)))?
    };

    let schema = index.schema();
    let session_field = schema
        .get_field("session_id")
        .map_err(|_| LetheError::internal("taintivy schema missing session_id field"))?;
    let repo_field = schema
        .get_field("repository_path")
        .map_err(|_| LetheError::internal("taintivy schema missing repository_path field"))?;
    let doc_id_field = schema
        .get_field("doc_id")
        .map_err(|_| LetheError::internal("taintivy schema missing doc_id field"))?;
    let kind_field = schema
        .get_field("kind")
        .map_err(|_| LetheError::internal("taintivy schema missing kind field"))?;
    let text_field = schema
        .get_field("text")
        .map_err(|_| LetheError::internal("taintivy schema missing text field"))?;
    let metadata_field = schema
        .get_field("metadata")
        .map_err(|_| LetheError::internal("taintivy schema missing metadata field"))?;

    let mut writer = index
        .writer(50_000_000)
        .map_err(|e| LetheError::internal(format!("Failed to create taintivy writer: {}", e)))?;

    writer.delete_term(Term::from_field_text(session_field, session_id));

    for chunk in chunks {
        let mut document = TantivyDocument::new();
        document.add_text(session_field, session_id);
        document.add_text(repo_field, &repo.path);
        document.add_text(doc_id_field, &chunk.id);
        document.add_text(kind_field, &chunk.kind);
        document.add_text(text_field, &chunk.text);
        document.add_text(
            metadata_field,
            serde_json::to_string(chunk)
                .map_err(|e| LetheError::internal(format!("Failed to serialize chunk: {}", e)))?,
        );
        let _ = writer.add_document(document);
    }

    writer
        .commit()
        .map_err(|e| LetheError::internal(format!("Failed to commit taintivy index: {}", e)))?;

    Ok(())
}

pub fn write_session_artifacts(
    session_dir: &Path,
    session_id: &str,
    repo: &RepositoryConfig,
    chunks: &[Chunk],
    messages: &[Message],
) -> Result<()> {
    write_parquet_chunks(session_dir, session_id, chunks)?;
    write_chunk_bloom(session_dir, chunks)?;
    write_parquet_messages(session_dir, messages)?;
    write_tantivy_index(session_dir, session_id, repo, chunks)?;
    Ok(())
}

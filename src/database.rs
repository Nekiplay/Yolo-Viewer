use rusqlite::{Connection, Result};
use serde::{Serialize, Deserialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ModelSettings {
    pub probability_threshold: f32,
    pub iou_threshold: f32,
    pub model_path: Option<String>,
}

impl Default for ModelSettings {
    fn default() -> Self {
        Self {
            probability_threshold: 0.25,
            iou_threshold: 0.45,
            model_path: None,
        }
    }
}

pub struct SettingsDatabase {
    connection: Connection,
}

impl SettingsDatabase {
    pub fn new() -> Result<Self> {
        // Create database file in the same directory as the executable
        let db_path = "yolo_settings.db";
        let connection = Connection::open(db_path)?;

        // Initialize database tables
        Self::initialize_database(&connection)?;

        Ok(Self { connection })
    }

    fn initialize_database(conn: &Connection) -> Result<()> {
        conn.execute(
            "CREATE TABLE IF NOT EXISTS settings (
                id INTEGER PRIMARY KEY,
                probability_threshold REAL NOT NULL,
                iou_threshold REAL NOT NULL,
                model_path TEXT
            )",
            [],
        )?;

        // Check if we have any settings, if not, insert defaults
        let count: i64 = conn.query_row("SELECT COUNT(*) FROM settings", [], |row: &rusqlite::Row| row.get(0))?;

        if count == 0 {
            let default_settings = ModelSettings::default();
            conn.execute(
                "INSERT INTO settings (id, probability_threshold, iou_threshold, model_path) VALUES (1, ?1, ?2, ?3)",
                rusqlite::params![
                    default_settings.probability_threshold,
                    default_settings.iou_threshold,
                    default_settings.model_path
                ],
            )?;
        }

        Ok(())
    }

    pub fn get_settings(&self) -> Result<ModelSettings> {
        self.connection.query_row(
            "SELECT probability_threshold, iou_threshold, model_path FROM settings WHERE id = 1",
            [],
            |row: &rusqlite::Row| {
                Ok(ModelSettings {
                    probability_threshold: row.get(0)?,
                    iou_threshold: row.get(1)?,
                    model_path: row.get(2)?,
                })
            },
        )
    }

    pub fn save_settings(&self, settings: &ModelSettings) -> Result<()> {
        self.connection.execute(
            "UPDATE settings SET probability_threshold = ?1, iou_threshold = ?2, model_path = ?3 WHERE id = 1",
            rusqlite::params![
                settings.probability_threshold,
                settings.iou_threshold,
                settings.model_path
            ],
        )?;
        Ok(())
    }
}

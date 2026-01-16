pub mod app;
pub mod utils;
pub mod inference;
pub mod loading;
pub mod database;

use image::{DynamicImage};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, mpsc};
use yolo_rs::YoloEntityOutput;

use crate::database::ModelSettings;

#[derive(Clone)]
pub struct Candidate {
    pub det: YoloEntityOutput,
    pub mask_coeffs: Vec<f32>,
    pub class_id: usize,
}

pub enum ImageInput {
    File(PathBuf),
    Pixels(DynamicImage),
}

#[derive(Clone)]
pub struct DetectionResult {
    pub texture_data: egui::ColorImage,
    pub mask_texture: Option<egui::ColorImage>,
    pub detections: Vec<YoloEntityOutput>,
    pub img_size: egui::Vec2,
}

#[derive(Clone)]
pub enum AppMessage {
    Success(DetectionResult),
    Error(String),
}

pub struct YoloGuiApp {
    pub model_session: Option<Arc<Mutex<yolo_rs::model::YoloModelSession>>>,
    pub model_input_size: (u32, u32),
    pub class_names: Arc<Vec<String>>,
    pub texture: Option<egui::TextureHandle>,
    pub mask_texture: Option<egui::TextureHandle>,
    pub detections: Vec<YoloEntityOutput>,
    pub img_size: egui::Vec2,
    pub zoom: f32,
    pub pan: egui::Vec2,
    pub fit_to_screen_req: bool,
    pub hovered_idx: Option<usize>,
    pub tx: mpsc::Sender<AppMessage>,
    pub rx: mpsc::Receiver<AppMessage>,
    pub is_processing: bool,
    pub status: String,
    pub settings: ModelSettings,
    pub database: Option<Arc<Mutex<crate::database::SettingsDatabase>>>,
    pub show_settings_window: bool,
}

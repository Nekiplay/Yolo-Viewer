use eframe::egui;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, mpsc};

use crate::{DetectionResult, ImageInput, AppMessage, YoloGuiApp, Candidate};
use crate::utils;
use crate::database::{SettingsDatabase, ModelSettings};

impl YoloGuiApp {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        // Initialize database and load settings first
        let database = match SettingsDatabase::new() {
            Ok(db) => Some(Arc::new(Mutex::<SettingsDatabase>::new(db))),
            Err(e) => {
                eprintln!("Failed to initialize database: {}", e);
                None
            }
        };

        let settings = if let Some(db) = &database {
            match db.lock().unwrap().get_settings() {
                Ok(settings) => settings,
                Err(e) => {
                    eprintln!("Failed to load settings: {}", e);
                    ModelSettings::default()
                }
            }
        } else {
            ModelSettings::default()
        };

        let default_models = [
            "best.onnx",
            "yolo11n-seg.onnx",
            "yolo11n.onnx",
            "yolov8n-seg.onnx",
            "yolov8n.onnx",
        ];
        let mut session = None;
        let mut loaded_path = None;

        for name in default_models {
            let path = PathBuf::from(name);
            if path.exists() {
                println!("Auto-loading: {:?}", path);
                if let Ok(mut s) = yolo_rs::model::YoloModelSession::from_filename_v8(&path) {
                    // Use settings from database
                    s.probability_threshold = Some(settings.probability_threshold);
                    s.iou_threshold = Some(settings.iou_threshold);
                    session = Some(Arc::new(Mutex::new(s)));
                    loaded_path = Some(name.to_string());
                    break;
                }
            }
        }

        let (input_size, class_names) = if let Some(s) = &session {
            let guard = s.lock().unwrap();
            let size = if let Some(input) = guard.session.inputs.first() {
                if let ort::value::ValueType::Tensor { shape, .. } = &input.input_type {
                    (shape[3] as u32, shape[2] as u32)
                } else {
                    (640, 640)
                }
            } else {
                (640, 640)
            };

            let mut names = Vec::new();
            if let Ok(meta) = guard.session.metadata() {
                if let Ok(Some(raw)) = meta.custom("names") {
                    let fixed = raw
                        .replace('\'', "\"")
                        .replace("{", "{\"")
                        .replace(": ", "\": ")
                        .replace(", ", ", \"");
                    if let Ok(map) = serde_json::from_str::<std::collections::HashMap<String, String>>(&fixed) {
                        let mut keys: Vec<u32> =
                            map.keys().filter_map(|k| k.parse().ok()).collect();
                        keys.sort();
                        let l: Vec<String> = keys
                            .iter()
                            .filter_map(|k| map.get(&k.to_string()).cloned())
                            .collect();
                        if !l.is_empty() {
                            names = l;
                        }
                    }
                }
            }
            (size, Arc::new(names))
        } else {
            ((640, 640), Arc::new(Vec::new()))
        };

        let (tx, rx) = mpsc::channel();

        let status = loaded_path
            .map(|n| format!("Model: {}", n))
            .unwrap_or_else(|| "Drag and drop the .onnx file".to_string());

        Self {
            model_session: session,
            model_input_size: input_size,
            class_names: class_names,
            texture: None,
            mask_texture: None,
            detections: Vec::new(),
            img_size: egui::Vec2::ZERO,
            zoom: 1.0,
            pan: egui::Vec2::ZERO,
            fit_to_screen_req: false,
            hovered_idx: None,
            tx,
            rx,
            is_processing: false,
            status,
            settings,
            database,
            show_settings_window: false,
        }
    }

    pub fn load_model(&mut self, path: PathBuf) {
        println!("Загрузка: {:?}", path);
        match yolo_rs::model::YoloModelSession::from_filename_v8(&path) {
            Ok(mut s) => {
                // Use settings from database
                s.probability_threshold = Some(self.settings.probability_threshold);
                s.iou_threshold = Some(self.settings.iou_threshold);
                let size = if let Some(input) = s.session.inputs.first() {
                    if let ort::value::ValueType::Tensor { shape, .. } = &input.input_type {
                        (shape[3] as u32, shape[2] as u32)
                    } else {
                        (640, 640)
                    }
                } else {
                    (640, 640)
                };

                let mut names = Vec::new();
                if let Ok(meta) = s.session.metadata() {
                    if let Ok(Some(raw)) = meta.custom("names") {
                        let fixed = raw
                            .replace('\'', "\"")
                            .replace("{", "{\"")
                            .replace(": ", "\": ")
                            .replace(", ", ", \"");
                        if let Ok(map) = serde_json::from_str::<std::collections::HashMap<String, String>>(&fixed) {
                            let mut keys: Vec<u32> =
                                map.keys().filter_map(|k| k.parse().ok()).collect();
                            keys.sort();
                            let l: Vec<String> = keys
                                .iter()
                                .filter_map(|k| map.get(&k.to_string()).cloned())
                                .collect();
                            if !l.is_empty() {
                                names = l;
                            }
                        }
                    }
                }
                self.model_session = Some(Arc::new(Mutex::new(s)));
                self.model_input_size = size;
                self.class_names = Arc::new(names);
                self.status = format!("Loaded: {:?}", path.file_name().unwrap());
                println!("Вход модели: {:?}", size);
            }
            Err(e) => {
                self.status = format!("Error: {:?}", e);
            }
        }
    }

    pub fn run_worker(&self, input: ImageInput, ctx: egui::Context) {
        let model_arc: Arc<Mutex<yolo_rs::model::YoloModelSession>> = match &self.model_session {
            Some(m) => Arc::clone(m),
            None => return,
        };
        let names_arc: Arc<Vec<String>> = Arc::clone(&self.class_names);
        let tx = self.tx.clone();
        let (tw, th) = self.model_input_size;

        std::thread::spawn(move || {
            let process = || -> Result<DetectionResult, String> {
                let original_img = match input {
                    ImageInput::File(path) => image::open(&path).map_err(|e| e.to_string())?,
                    ImageInput::Pixels(img) => img,
                };
                let (img_w, img_h) = (original_img.width() as f32, original_img.height() as f32);

                let resized =
                    original_img.resize_exact(tw, th, image::imageops::FilterType::Triangle);
                let rgb = resized.to_rgb8();
                let mut flat_data = vec![0.0f32; (3 * th * tw) as usize];
                let area = (th * tw) as usize;
                for (x, y, pixel) in rgb.enumerate_pixels() {
                    let idx = y as usize * tw as usize + x as usize;
                    flat_data[0 * area + idx] = pixel[0] as f32 / 255.0;
                    flat_data[1 * area + idx] = pixel[1] as f32 / 255.0;
                    flat_data[2 * area + idx] = pixel[2] as f32 / 255.0;
                }

                let input_tensor =
                    ort::value::Value::from_array((vec![1usize, 3, th as usize, tw as usize], flat_data))
                        .map_err(|e| e.to_string())?;

                // INFERENCE
                let (output0_shape, output0_data, output1_opt) = {
                    let mut m = model_arc.lock().unwrap();
                    let outputs = m
                        .session
                        .run(ort::inputs![input_tensor])
                        .map_err(|e| e.to_string())?;
                    let (s0, d0) = outputs[0]
                            .try_extract_tensor::<f32>()
                            .map_err(|e: ort::Error| e.to_string())?;

                    let out1 = if outputs.len() > 1 {
                        let (s1, d1) = outputs[1]
                            .try_extract_tensor::<f32>()
                            .map_err(|e: ort::Error| e.to_string())?;
                        let mask_dims = (s1[3] as usize, s1[2] as usize);
                        Some((mask_dims, d1.to_vec()))
                    } else {
                        None
                    };
                    (s0.to_vec(), d0.to_vec(), out1)
                };

                // PARSING
                let num_rows = output0_shape[1] as usize;
                let num_anchors = output0_shape[2] as usize;
                let has_masks = output1_opt.is_some();

                let num_classes = if has_masks {
                    if num_rows > 36 { num_rows - 4 - 32 } else { 0 }
                } else {
                    if num_rows > 4 { num_rows - 4 } else { 0 }
                };
                let mask_start_idx = if has_masks { num_rows - 32 } else { 0 };

                let mut candidates = Vec::new();

                for i in 0..num_anchors {
                    let mut max_conf = 0.0f32;
                    let mut class_id = 0;
                    for c in 0..num_classes {
                        let conf = output0_data[(4 + c) * num_anchors + i];
                        if conf > max_conf {
                            max_conf = conf;
                            class_id = c;
                        }
                    }
                    if max_conf > 0.25 {
                        let cx = output0_data[0 * num_anchors + i];
                        let cy = output0_data[1 * num_anchors + i];
                        let w = output0_data[2 * num_anchors + i];
                        let h = output0_data[3 * num_anchors + i];

                        let label = if class_id < names_arc.len() {
                            names_arc[class_id].clone()
                        } else {
                            format!("Class {}", class_id)
                        };

                        let mut mask_coeffs = Vec::new();
                        if has_masks {
                            for m in 0..32 {
                                let idx = (mask_start_idx + m) * num_anchors + i;
                                if idx < output0_data.len() {
                                    mask_coeffs.push(output0_data[idx]);
                                }
                            }
                        }
                        if has_masks && mask_coeffs.len() != 32 {
                            continue;
                        }

                        candidates.push(Candidate {
                            det: yolo_rs::YoloEntityOutput {
                                bounding_box: yolo_rs::BoundingBox {
                                    x1: cx - w / 2.0,
                                    y1: cy - h / 2.0,
                                    x2: cx + w / 2.0,
                                    y2: cy + h / 2.0,
                                },
                                confidence: max_conf,
                                label: arcstr::ArcStr::from(label),
                            },
                            mask_coeffs,
                            class_id,
                        });
                    }
                }

                let kept_indices = utils::perform_nms_indices(&candidates, 0.45);
                let final_detections: Vec<yolo_rs::YoloEntityOutput> = kept_indices
                    .iter()
                    .map(|&i| candidates[i].det.clone())
                    .collect();

                println!("Objects found: {}", final_detections.len());

                let mask_image = if has_masks && !kept_indices.is_empty() {
                    if let Some((mask_dims, proto_data)) = output1_opt {
                        let kept_candidates: Vec<&Candidate> =
                            kept_indices.iter().map(|&i| &candidates[i]).collect();
                        Some(utils::process_masks(
                            &kept_candidates,
                            &proto_data,
                            (tw as usize, th as usize),
                            mask_dims,
                        ))
                    } else {
                        None
                    }
                } else {
                    None
                };

                Ok(DetectionResult {
                    texture_data: utils::load_egui_image(&original_img),
                    mask_texture: mask_image,
                    detections: final_detections,
                    img_size: egui::vec2(img_w, img_h),
                })
            };

            match process() {
                Ok(res) => {
                    let _ = tx.send(AppMessage::Success(res));
                }
                Err(e) => {
                    eprintln!("Error: {}", e);
                    let _ = tx.send(AppMessage::Error(e));
                }
            }
            ctx.request_repaint();
        });
    }

    pub fn handle_clipboard(&mut self, ctx: &egui::Context) {
        if let Ok(mut cb) = arboard::Clipboard::new() {
            if let Ok(img_data) = cb.get_image() {
                let bytes = img_data.bytes.into_owned();
                if let Some(rgba) =
                    image::RgbaImage::from_raw(img_data.width as u32, img_data.height as u32, bytes)
                {
                    self.is_processing = true;
                    self.status = "Processing...".to_string();
                    self.run_worker(
                        ImageInput::Pixels(image::DynamicImage::ImageRgba8(rgba)),
                        ctx.clone(),
                    );
                }
            } else if let Ok(text) = cb.get_text() {
                let path = PathBuf::from(text.trim_matches('"').trim());
                if path.exists() {
                    self.is_processing = true;
                    self.status = "File...".to_string();
                    self.run_worker(ImageInput::File(path), ctx.clone());
                }
            }
        }
    }
}

impl eframe::App for YoloGuiApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if let Ok(msg) = self.rx.try_recv() {
            match msg {
                AppMessage::Success(res) => {
                    self.texture =
                        Some(ctx.load_texture("img", res.texture_data, Default::default()));
                    // Текстура теперь высокого разрешения (1280x1280), ставим LINEAR чтобы при уменьшении было красиво
                    self.mask_texture = res
                        .mask_texture
                        .map(|m| ctx.load_texture("masks", m, egui::TextureOptions::LINEAR));
                    self.detections = res.detections;
                    self.img_size = res.img_size;
                    self.status = format!(
                        "Founded: {} | Mask: {}",
                        self.detections.len(),
                        if self.mask_texture.is_some() {
                            "OK"
                        } else {
                            "-"
                        }
                    );
                    self.fit_to_screen_req = true;
                }
                AppMessage::Error(e) => self.status = format!("Error: {}", e),
            }
            self.is_processing = false;
        }

        if !ctx.input(|i| i.raw.dropped_files.is_empty()) {
            let dropped = ctx.input(|i| i.raw.dropped_files.first().cloned());
            if let Some(d) = dropped {
                if let Some(path) = d.path {
                    if let Some(ext) = path.extension() {
                        let ext_str = ext.to_string_lossy().to_lowercase();
                        if ext_str == "onnx" {
                            self.load_model(path);
                        } else if self.model_session.is_some() {
                            self.is_processing = true;
                            self.run_worker(ImageInput::File(path), ctx.clone());
                        }
                    }
                }
            }
        }

        if ctx.input_mut(|i| {
            i.consume_shortcut(&egui::KeyboardShortcut::new(
                egui::Modifiers::COMMAND,
                egui::Key::V,
            ))
        }) {
            if self.model_session.is_some() {
                self.handle_clipboard(ctx);
            }
        }

        egui::SidePanel::right("side")
            .width_range(160.0..=350.0)
            .show(ctx, |ui| {
                ui.heading("Objects");
                ui.separator();
                egui::ScrollArea::vertical().show(ui, |ui| {
                    let mut lh = None;
                    for (i, det) in self.detections.iter().enumerate() {
                        let col = utils::get_color(&det.label);
                        ui.horizontal(|ui| {
                            let (r, _) = ui
                                .allocate_exact_size(egui::vec2(10.0, 10.0), egui::Sense::hover());
                            ui.painter().circle_filled(r.center(), 4.0, col);
                            if ui
                                .selectable_label(
                                    self.hovered_idx == Some(i),
                                    format!("{} {:.0}%", det.label, det.confidence * 100.0),
                                )
                                .hovered()
                            {
                                lh = Some(i);
                            }
                        });
                    }
                    if lh.is_some() {
                        self.hovered_idx = lh;
                    }
                });
            });

        egui::TopBottomPanel::bottom("status").show(ctx, |ui| {
            ui.horizontal(|ui| {
                // Add settings icon button on the left
                if ui.button("⚙").clicked() {
                    self.show_settings_window = !self.show_settings_window;
                }

                if self.is_processing {
                    ui.spinner();
                }
                ui.label(" ".to_owned() + &self.status);
            });
        });

        // Settings window
        if self.show_settings_window {
            egui::Window::new("Model Settings")
                .open(&mut self.show_settings_window)
                .show(ctx, |ui| {
                    ui.label("Detection Thresholds:");
                    ui.horizontal(|ui| {
                        ui.label("Probability:");
                        let mut prob = self.settings.probability_threshold;
                        if ui.add(egui::Slider::new(&mut prob, 0.0..=1.0)).changed() {
                            self.settings.probability_threshold = prob;
                            if let Some(db) = &self.database {
                                if let Ok(db_lock) = db.lock() {
                                    let _ = db_lock.save_settings(&self.settings);
                                }
                            }
                        }
                        ui.label(format!("{:.2}", prob));
                    });

                    ui.horizontal(|ui| {
                        ui.label("IoU:");
                        let mut iou = self.settings.iou_threshold;
                        if ui.add(egui::Slider::new(&mut iou, 0.0..=1.0)).changed() {
                            self.settings.iou_threshold = iou;
                            if let Some(db) = &self.database {
                                if let Ok(db_lock) = db.lock() {
                                    let _ = db_lock.save_settings(&self.settings);
                                }
                            }
                        }
                        ui.label(format!("{:.2}", iou));
                    });

                    if ui.button("Reset to Defaults").clicked() {
                        self.settings = ModelSettings::default();
                        if let Some(db) = &self.database {
                            if let Ok(db_lock) = db.lock() {
                                let _ = db_lock.save_settings(&self.settings);
                            }
                        }
                    }
                });
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            if self.model_session.is_none() {
                ui.centered_and_justified(|ui| {
                    ui.heading("Upload the model (drag and drop .onnx)");
                });
                return;
            }

            if let Some(tex) = &self.texture {
                let (response, painter) =
                    ui.allocate_painter(ui.available_size(), egui::Sense::click_and_drag());
                let viewport_rect = response.rect;

                if self.fit_to_screen_req {
                    let w_ratio = viewport_rect.width() / self.img_size.x;
                    let h_ratio = viewport_rect.height() / self.img_size.y;
                    self.zoom = w_ratio.min(h_ratio) * 0.95;
                    let content_size = self.img_size * self.zoom;
                    self.pan =
                        viewport_rect.min.to_vec2() + (viewport_rect.size() - content_size) / 2.0;
                    self.fit_to_screen_req = false;
                }

                let scroll_delta = ctx.input(|i| i.raw_scroll_delta);
                if scroll_delta.y != 0.0 {
                    if let Some(mouse_pos) = ctx.input(|i| i.pointer.hover_pos()) {
                        if viewport_rect.contains(mouse_pos) {
                            let zoom_factor = if scroll_delta.y > 0.0 { 1.1 } else { 0.9 };
                            let old_zoom = self.zoom;
                            let new_zoom = (self.zoom * zoom_factor).clamp(0.05, 50.0);
                            let mouse_vec = mouse_pos.to_vec2();
                            self.pan = mouse_vec - (mouse_vec - self.pan) * (new_zoom / old_zoom);
                            self.zoom = new_zoom;
                        }
                    }
                }

                if response.dragged_by(egui::PointerButton::Primary)
                    || response.dragged_by(egui::PointerButton::Middle)
                {
                    self.pan += response.drag_delta();
                }

                let displayed_rect =
                    egui::Rect::from_min_size(self.pan.to_pos2(), self.img_size * self.zoom);
                let painter = painter.with_clip_rect(viewport_rect);

                painter.image(
                    tex.id(),
                    displayed_rect,
                    egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                    egui::Color32::WHITE,
                );

                if let Some(m) = &self.mask_texture {
                    painter.image(
                        m.id(),
                        displayed_rect,
                        egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                        egui::Color32::WHITE,
                    );
                }

                let mut h_id = None;
                if let Some(ptr) = response.hover_pos() {
                    let relative_pos = (ptr - self.pan) / self.zoom;
                    if relative_pos.x >= 0.0
                        && relative_pos.y >= 0.0
                        && relative_pos.x <= self.img_size.x
                        && relative_pos.y <= self.img_size.y
                    {
                        let sx = self.img_size.x / self.model_input_size.0 as f32;
                        let sy = self.img_size.y / self.model_input_size.1 as f32;
                        for (i, d) in self.detections.iter().enumerate().rev() {
                            let b = &d.bounding_box;
                            let bx1 = b.x1 * sx;
                            let by1 = b.y1 * sy;
                            let bx2 = b.x2 * sx;
                            let by2 = b.y2 * sy;
                            if relative_pos.x >= bx1
                                && relative_pos.x <= bx2
                                && relative_pos.y >= by1
                                && relative_pos.y <= by2
                            {
                                h_id = Some(i);
                                break;
                            }
                        }
                    }
                }
                if h_id.is_some() {
                    self.hovered_idx = h_id;
                }

                let sx = (self.img_size.x / self.model_input_size.0 as f32) * self.zoom;
                let sy = (self.img_size.y / self.model_input_size.1 as f32) * self.zoom;

                for (i, d) in self.detections.iter().enumerate() {
                    let b = &d.bounding_box;
                    let col = utils::get_color(&d.label);
                    let is_h = self.hovered_idx == Some(i);
                    let c = if is_h { col.linear_multiply(1.5) } else { col };
                    let r = egui::Rect::from_min_max(
                        self.pan.to_pos2() + egui::vec2(b.x1 * sx, b.y1 * sy),
                        self.pan.to_pos2() + egui::vec2(b.x2 * sx, b.y2 * sy),
                    );
                    if painter.clip_rect().intersects(r) {
                        painter.rect_stroke(
                            r,
                            0.0,
                            egui::Stroke::new(if is_h { 3.0 } else { 1.5 }, c),
                        );
                        if is_h || self.zoom > 0.4 {
                            let t = format!("{} {:.0}%", d.label, d.confidence * 100.0);
                            let f =
                                egui::FontId::proportional((13.0f32 * self.zoom).clamp(10.0f32, 24.0f32));
                            let g = painter.layout_no_wrap(t, f, egui::Color32::WHITE);
                            let mut lp = r.min;
                            if lp.y - g.size().y - 6.0 < displayed_rect.min.y {
                                lp.y += 2.0;
                            } else {
                                lp.y -= g.size().y + 4.0;
                            }
                            let lr = egui::Rect::from_min_size(lp, g.size() + egui::vec2(8.0, 4.0));
                            painter.rect_filled(lr, egui::Rounding::same(4.0), c);
                            painter.galley(lr.min + egui::vec2(4.0, 2.0), g, egui::Color32::WHITE);
                        }
                    }
                }
            } else {
                ui.centered_and_justified(|ui| ui.label("Drag and drop the image"));
            }
        });
    }
}

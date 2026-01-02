use eframe::egui;
use image::{DynamicImage, RgbaImage};
use std::sync::{Arc, Mutex, mpsc};
use std::path::PathBuf;
use std::collections::HashMap;
use ort::value::{Value, ValueType};
// Убедитесь, что yolo_rs и arcstr добавлены в Cargo.toml
// [dependencies]
// eframe = "0.28" (или ваша версия)
// image = "0.24"
// ort = { version = "2.0.0-rc.4", features = ["cuda"] } # Или другая версия ORT
// yolo-rs = "0.1" (или ваша библиотека)
// arcstr = "1.1"
// arboard = "3.2"
// serde_json = "1.0"
use yolo_rs::{YoloEntityOutput, model, BoundingBox};

enum ImageInput {
    File(PathBuf),
    Pixels(DynamicImage),
}

struct DetectionResult {
    texture_data: egui::ColorImage,
    detections: Vec<YoloEntityOutput>,
    img_size: egui::Vec2,
}

struct YoloGuiApp {
    model_session: Arc<Mutex<model::YoloModelSession>>,
    model_input_size: (u32, u32),
    class_names: Arc<Vec<String>>, 
    
    // Состояние изображения
    texture: Option<egui::TextureHandle>,
    detections: Vec<YoloEntityOutput>,
    img_size: egui::Vec2,
    
    // Зум и навигация
    zoom: f32,
    hovered_idx: Option<usize>,
    
    tx: mpsc::Sender<DetectionResult>,
    rx: mpsc::Receiver<DetectionResult>,
    
    is_processing: bool,
    status: String,
}

impl YoloGuiApp {
    fn new(_cc: &eframe::CreationContext<'_>, model_path: PathBuf) -> Self {
        let mut session = model::YoloModelSession::from_filename_v8(&model_path)
            .expect("Failed to load model");

        // Извлечение имен классов
        let mut class_names = get_coco_names();
        if let Ok(metadata) = session.session.metadata() {
            if let Ok(Some(names_raw)) = metadata.custom("names") {
                let fixed = names_raw.replace('\'', "\"").replace("{", "{\"").replace(": ", "\": ").replace(", ", ", \"");
                if let Ok(map) = serde_json::from_str::<HashMap<String, String>>(&fixed) {
                    let mut keys: Vec<u32> = map.keys().filter_map(|k| k.parse().ok()).collect();
                    keys.sort();
                    let new_names: Vec<String> = keys.iter().filter_map(|k| map.get(&k.to_string()).cloned()).collect();
                    if !new_names.is_empty() { class_names = new_names; }
                }
            }
        }

        let input_size = if let Some(input) = session.session.inputs.first() {
            if let ValueType::Tensor { shape, .. } = &input.input_type {
                (shape[3] as u32, shape[2] as u32)
            } else { (640, 640) }
        } else { (640, 640) };

        let (tx, rx) = mpsc::channel();
        session.probability_threshold = Some(0.4);
        session.iou_threshold = Some(0.4);

        Self {
            model_session: Arc::new(Mutex::new(session)),
            model_input_size: input_size,
            class_names: Arc::new(class_names),
            texture: None,
            detections: Vec::new(),
            img_size: egui::Vec2::ZERO,
            zoom: 1.0,
            hovered_idx: None,
            tx, rx,
            is_processing: false,
            status: "Готов. Ctrl+V или Drag&Drop.".to_string(),
        }
    }

    fn run_worker(&self, input: ImageInput, ctx: egui::Context) {
        let model_arc = Arc::clone(&self.model_session);
        let names_arc = Arc::clone(&self.class_names);
        let tx = self.tx.clone();
        let (tw, th) = self.model_input_size;
        
        std::thread::spawn(move || {
            let process = || -> Result<(), String> {
                let original_img = match input {
                    ImageInput::File(path) => image::open(&path).map_err(|e| e.to_string())?,
                    ImageInput::Pixels(img) => img,
                };
                let (img_w, img_h) = (original_img.width() as f32, original_img.height() as f32);

                let resized = original_img.resize_exact(tw, th, image::imageops::FilterType::Triangle);
                let rgb = resized.to_rgb8();
                
                let mut flat_data = vec![0.0f32; (3 * th * tw) as usize];
                let area = (th * tw) as usize;
                for (x, y, pixel) in rgb.enumerate_pixels() {
                    let idx = y as usize * tw as usize + x as usize;
                    flat_data[0 * area + idx] = pixel[0] as f32 / 255.0;
                    flat_data[1 * area + idx] = pixel[1] as f32 / 255.0;
                    flat_data[2 * area + idx] = pixel[2] as f32 / 255.0;
                }

                let input_tensor = Value::from_array((vec![1usize, 3, th as usize, tw as usize], flat_data)).map_err(|e| e.to_string())?;
                
                let (shape, data_vec) = {
                    let mut m = model_arc.lock().unwrap();
                    let outputs = m.session.run(ort::inputs![input_tensor]).map_err(|e| e.to_string())?;
                    let (shape, slice) = outputs[0].try_extract_tensor::<f32>().map_err(|e| e.to_string())?;
                    (shape.to_vec(), slice.to_vec()) 
                }; 

                let num_rows = shape[1] as usize;   
                let num_anchors = shape[2] as usize;
                let mut candidates = Vec::new();
                
                for i in 0..num_anchors {
                    let mut max_conf = 0.0f32;
                    let mut class_id = 0;
                    for c in 4..num_rows {
                        let conf = data_vec[c * num_anchors + i];
                        if conf > max_conf { max_conf = conf; class_id = c - 4; }
                    }
                    
                    if max_conf > 0.4 {
                        let cx = data_vec[0 * num_anchors + i];
                        let cy = data_vec[1 * num_anchors + i];
                        let w = data_vec[2 * num_anchors + i];
                        let h = data_vec[3 * num_anchors + i];
                        
                        let label = names_arc.get(class_id).cloned().unwrap_or_else(|| format!("{}", class_id)); // ID если нет имени
                        candidates.push(YoloEntityOutput {
                            bounding_box: BoundingBox { x1: cx - w / 2.0, y1: cy - h / 2.0, x2: cx + w / 2.0, y2: cy + h / 2.0 },
                            confidence: max_conf,
                            label: arcstr::ArcStr::from(label),
                        });
                    }
                }

                let detections = perform_nms(candidates, 0.45);

                let _ = tx.send(DetectionResult {
                    texture_data: load_egui_image(&original_img),
                    detections,
                    img_size: egui::vec2(img_w, img_h),
                });
                Ok(())
            };
            if let Err(e) = process() { eprintln!("Inference Error: {}", e); }
            ctx.request_repaint();
        });
    }

    fn handle_clipboard(&mut self, ctx: &egui::Context) {
        if let Ok(mut cb) = arboard::Clipboard::new() {
            if let Ok(img_data) = cb.get_image() {
                let rgba = RgbaImage::from_raw(img_data.width as u32, img_data.height as u32, img_data.bytes.to_vec());
                if let Some(rgba) = rgba {
                    self.is_processing = true;
                    self.run_worker(ImageInput::Pixels(DynamicImage::ImageRgba8(rgba)), ctx.clone());
                    return;
                }
            }
            if let Ok(text) = cb.get_text() {
                let path = PathBuf::from(text.trim_matches('"').trim());
                if path.exists() && path.is_file() {
                    self.is_processing = true;
                    self.run_worker(ImageInput::File(path), ctx.clone());
                }
            }
        }
    }
}

impl eframe::App for YoloGuiApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // 1. Сначала обрабатываем получение результатов от потока
        if let Ok(res) = self.rx.try_recv() {
            self.texture = Some(ctx.load_texture("img", res.texture_data, Default::default()));
            self.detections = res.detections;
            self.img_size = res.img_size;
            self.is_processing = false;
            self.zoom = 1.0;
            self.status = format!("Готово. Найдено объектов: {}", self.detections.len());
        }

        // 2. Логика ввода (Ctrl+V и Drag & Drop)
        // Мы собираем намерение загрузить файл в переменную, чтобы избежать конфликтов borrowing
        let mut next_input = None;

        // A) Проверка Ctrl+V
        if ctx.input_mut(|i| i.consume_shortcut(&egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, egui::Key::V))) {
            if let Ok(mut cb) = arboard::Clipboard::new() {
                let mut found = false;
                // Пробуем как картинку
                if let Ok(img_data) = cb.get_image() {
                    if let Some(rgba) = RgbaImage::from_raw(img_data.width as u32, img_data.height as u32, img_data.bytes.into_owned()) {
                        next_input = Some(ImageInput::Pixels(DynamicImage::ImageRgba8(rgba)));
                        found = true;
                    }
                }
                // Если не картинка, пробуем как текст (путь к файлу)
                if !found {
                    if let Ok(text) = cb.get_text() {
                        let path = PathBuf::from(text.trim_matches('"').trim());
                        if path.exists() && path.is_file() {
                            next_input = Some(ImageInput::File(path));
                            found = true;
                        }
                    }
                }
                
                if !found {
                    self.status = "Буфер обмена пуст или формат не поддерживается".to_string();
                }
            } else {
                self.status = "Ошибка доступа к буферу обмена".to_string();
            }
        }

        // B) Проверка Drag & Drop
        // Важно: проверяем dropped_files до отрисовки UI
        if !ctx.input(|i| i.raw.dropped_files.is_empty()) {
            ctx.input(|i| {
                if let Some(dropped) = i.raw.dropped_files.first() {
                    if let Some(path) = &dropped.path {
                        next_input = Some(ImageInput::File(path.clone()));
                    }
                }
            });
        }

        // 3. Запуск обработки, если что-то пришло
        if let Some(input) = next_input {
            self.is_processing = true;
            self.status = "Обработка...".to_string();
            self.run_worker(input, ctx.clone());
        }

        // 4. Обычная логика зума
        let z_delta = ctx.input(|i| i.zoom_delta());
        if z_delta != 1.0 {
            self.zoom = (self.zoom * z_delta).clamp(0.1, 20.0);
        }

        // --- ОТРИСОВКА ИНТЕРФЕЙСА ---

        // Боковая панель
        egui::SidePanel::right("side").width_range(160.0..=250.0).show(ctx, |ui| {
            ui.heading(format!("Найдено: {}", self.detections.len()));
            ui.separator();
            egui::ScrollArea::vertical().show(ui, |ui| {
                let mut hovered_in_list = None;
                for (i, det) in self.detections.iter().enumerate() {
                    let color = get_color(&det.label);
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 8.0;
                        let (rect, _resp) = ui.allocate_exact_size(egui::vec2(10.0, 10.0), egui::Sense::hover());
                        ui.painter().circle_filled(rect.center(), 4.0, color);

                        let text = format!("{}  {:.0}%", det.label, det.confidence * 100.0);
                        let is_selected = self.hovered_idx == Some(i);
                        let res = ui.selectable_label(is_selected, text);
                        if res.hovered() { hovered_in_list = Some(i); }
                    });
                }
                if hovered_in_list.is_some() {
                    self.hovered_idx = hovered_in_list;
                }
            });
        });

        // Статус бар
        egui::TopBottomPanel::bottom("status").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if self.is_processing { ui.spinner(); }
                ui.label(&self.status);
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("Сброс (1:1)").clicked() { self.zoom = 1.0; }
                    ui.label(format!("Zoom: {:.0}%", self.zoom * 100.0));
                });
            });
        });

        // Центральная панель
        egui::CentralPanel::default().show(ctx, |ui| {
            if let Some(tex) = &self.texture {
                let display_size = self.img_size * self.zoom;
                
                egui::ScrollArea::both().show(ui, |ui| {
                    let (rect, response) = ui.allocate_exact_size(display_size, egui::Sense::hover());
                    
                    ui.painter().image(tex.id(), rect, egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)), egui::Color32::WHITE);
                    
                    let mut hovered_on_canvas = None;
                    if let Some(ptr_pos) = response.hover_pos() {
                        let scale_x = (self.img_size.x / self.model_input_size.0 as f32) * self.zoom;
                        let scale_y = (self.img_size.y / self.model_input_size.1 as f32) * self.zoom;
                        
                        for (i, det) in self.detections.iter().enumerate().rev() {
                            let b = &det.bounding_box;
                            let screen_rect = egui::Rect::from_min_max(
                                egui::pos2(b.x1 * scale_x, b.y1 * scale_y) + rect.min.to_vec2(),
                                egui::pos2(b.x2 * scale_x, b.y2 * scale_y) + rect.min.to_vec2(),
                            );
                            if screen_rect.contains(ptr_pos) {
                                hovered_on_canvas = Some(i);
                                break; 
                            }
                        }
                    }
                    if hovered_on_canvas.is_some() { self.hovered_idx = hovered_on_canvas; }
                    
                    let painter = ui.painter();
                    let scale_x = (self.img_size.x / self.model_input_size.0 as f32) * self.zoom;
                    let scale_y = (self.img_size.y / self.model_input_size.1 as f32) * self.zoom;

                    for (i, det) in self.detections.iter().enumerate() {
                        let b = &det.bounding_box;
                        let base_color = get_color(&det.label);
                        let is_hovered = self.hovered_idx == Some(i);
                        let color = if is_hovered { base_color.linear_multiply(1.5) } else { base_color };
                        
                        let screen_rect = egui::Rect::from_min_max(
                            egui::pos2(b.x1 * scale_x, b.y1 * scale_y) + rect.min.to_vec2(),
                            egui::pos2(b.x2 * scale_x, b.y2 * scale_y) + rect.min.to_vec2(),
                        );

                        painter.rect_filled(screen_rect, 4.0, color.linear_multiply(0.15));
                        let stroke_width = if is_hovered { 3.0 } else { 1.5 };
                        painter.rect_stroke(screen_rect, 4.0, egui::Stroke::new(stroke_width, color));
                        
                        if is_hovered || self.zoom > 0.4 {
                            let label_text = format!("{} {:.0}%", det.label, det.confidence * 100.0);
                            let font_size = (13.0 * self.zoom).clamp(10.0, 24.0);
                            let font_id = egui::FontId::proportional(font_size);
                            let galley = painter.layout_no_wrap(label_text, font_id, egui::Color32::WHITE);
                            
                            let mut label_pos = screen_rect.min;
                            if label_pos.y - galley.size().y - 6.0 < rect.min.y {
                                label_pos.y += 2.0; 
                            } else {
                                label_pos.y -= galley.size().y + 4.0;
                            }

                            let label_rect = egui::Rect::from_min_size(label_pos, galley.size() + egui::vec2(8.0, 4.0));
                            painter.rect_filled(label_rect, 4.0, color);
                            painter.galley(label_rect.min + egui::vec2(4.0, 2.0), galley, egui::Color32::WHITE);
                        }
                    }
                });
            } else {
                ui.centered_and_justified(|ui| {
                    ui.label(egui::RichText::new("Перетащите картинку сюда или нажмите Ctrl+V").size(24.0).color(egui::Color32::GRAY));
                });
            }
        });
    }
}

// Хелперы
fn get_color(label: &str) -> egui::Color32 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    let mut hasher = DefaultHasher::new();
    label.hash(&mut hasher);
    let h = hasher.finish();
    
    let hue = (h % 360) as f32 / 360.0;
    egui::ecolor::Hsva::new(hue, 0.85, 0.85, 1.0).into()
}

fn perform_nms(mut candidates: Vec<YoloEntityOutput>, iou_threshold: f32) -> Vec<YoloEntityOutput> {
    candidates.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
    let mut selected: Vec<YoloEntityOutput> = Vec::new();
    let mut active = vec![true; candidates.len()];

    for i in 0..candidates.len() {
        if !active[i] { continue; }
        let a = &candidates[i];
        selected.push(a.clone());
        
        for j in (i + 1)..candidates.len() {
            if !active[j] { continue; }
            let b = &candidates[j];
            
            let b1 = &a.bounding_box; 
            let b2 = &b.bounding_box;
            
            let inter_x1 = b1.x1.max(b2.x1);
            let inter_y1 = b1.y1.max(b2.y1);
            let inter_x2 = b1.x2.min(b2.x2);
            let inter_y2 = b1.y2.min(b2.y2);

            let inter_area = (inter_x2 - inter_x1).max(0.0) * (inter_y2 - inter_y1).max(0.0);
            let area1 = (b1.x2 - b1.x1) * (b1.y2 - b1.y1);
            let area2 = (b2.x2 - b2.x1) * (b2.y2 - b2.y1);
            
            let union_area = area1 + area2 - inter_area;
            let iou = if union_area <= 0.0 { 0.0 } else { inter_area / union_area };
            
            if iou > iou_threshold {
                active[j] = false;
            }
        }
    }
    selected
}

fn load_egui_image(img: &DynamicImage) -> egui::ColorImage {
    let size = [img.width() as _, img.height() as _];
    let rgba = img.to_rgba8();
    egui::ColorImage::from_rgba_unmultiplied(size, rgba.as_flat_samples().as_slice())
}

fn get_coco_names() -> Vec<String> {
    vec!["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
        .iter().map(|s| s.to_string()).collect()
}

fn main() -> eframe::Result<()> {
    let model_path = PathBuf::from("yolov8n.onnx"); 
    eframe::run_native(
        "YOLO Ultimate Viewer",
        eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default()
                .with_inner_size([1200.0, 850.0])
                .with_drag_and_drop(true),
            ..Default::default()
        },
        Box::new(|cc| Ok(Box::new(YoloGuiApp::new(cc, model_path)))),
    )
}
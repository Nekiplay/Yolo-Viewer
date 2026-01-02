use eframe::egui;
use image::{DynamicImage, RgbaImage, ImageBuffer, Rgba};
use std::sync::{Arc, Mutex, mpsc};
use std::path::PathBuf;
use std::collections::HashMap;
use ort::value::{Value, ValueType};
use yolo_rs::{YoloEntityOutput, model, BoundingBox};

// --- КОНСТАНТЫ ---
const MASK_COEFFS_NUM: usize = 32;

// --- ТИПЫ ---
enum ImageInput {
    File(PathBuf),
    Pixels(DynamicImage),
}

struct DetectionResult {
    texture_data: egui::ColorImage,
    mask_texture: Option<egui::ColorImage>,
    detections: Vec<YoloEntityOutput>,
    img_size: egui::Vec2,
}

#[derive(Clone)]
struct Candidate {
    det: YoloEntityOutput,
    mask_coeffs: Vec<f32>,
    class_id: usize,
}

enum AppMessage {
    Success(DetectionResult),
    Error(String),
}

struct YoloGuiApp {
    model_session: Option<Arc<Mutex<model::YoloModelSession>>>,
    model_input_size: (u32, u32),
    class_names: Arc<Vec<String>>, 
    
    texture: Option<egui::TextureHandle>,
    mask_texture: Option<egui::TextureHandle>,
    detections: Vec<YoloEntityOutput>,
    img_size: egui::Vec2,
    
    zoom: f32,
    hovered_idx: Option<usize>,
    
    tx: mpsc::Sender<AppMessage>,
    rx: mpsc::Receiver<AppMessage>,
    
    is_processing: bool,
    status: String,
}

impl YoloGuiApp {
    fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        // Добавил best.onnx в список автозагрузки
        let default_models = ["best.onnx", "yolo11n-seg.onnx", "yolo11n.onnx", "yolov8n-seg.onnx", "yolov8n.onnx"];
        let mut session = None;
        let mut loaded_path = None;

        for name in default_models {
            let path = PathBuf::from(name);
            if path.exists() {
                println!("Auto-loading: {:?}", path);
                if let Ok(mut s) = model::YoloModelSession::from_filename_v8(&path) {
                    s.probability_threshold = Some(0.25);
                    s.iou_threshold = Some(0.45);
                    session = Some(Arc::new(Mutex::new(s)));
                    loaded_path = Some(name.to_string());
                    break;
                }
            }
        }

        // Пытаемся достать размер входа и имена классов
        let (input_size, class_names) = if let Some(s) = &session {
            let guard = s.lock().unwrap();
            
            // Определяем размер входа (часто 640, но может быть 512 и т.д.)
            let size = if let Some(input) = guard.session.inputs.first() {
                if let ValueType::Tensor { shape, .. } = &input.input_type {
                    (shape[3] as u32, shape[2] as u32)
                } else { (640, 640) }
            } else { (640, 640) };
            
            // Пытаемся достать имена, иначе генерируем заглушки
            let mut names = Vec::new(); 
            if let Ok(meta) = guard.session.metadata() {
                 if let Ok(Some(raw)) = meta.custom("names") {
                    let fixed = raw.replace('\'', "\"").replace("{", "{\"").replace(": ", "\": ").replace(", ", ", \"");
                    if let Ok(map) = serde_json::from_str::<HashMap<String, String>>(&fixed) {
                         let mut keys: Vec<u32> = map.keys().filter_map(|k| k.parse().ok()).collect();
                         keys.sort();
                         let loaded_names: Vec<String> = keys.iter().filter_map(|k| map.get(&k.to_string()).cloned()).collect();
                         if !loaded_names.is_empty() { names = loaded_names; }
                    }
                 }
            }
            // Если имен нет, создаем "Class 0", "Class 1"... позже, динамически
            (size, Arc::new(names))
        } else {
            ((640, 640), Arc::new(Vec::new()))
        };

        let (tx, rx) = mpsc::channel();
        let status = loaded_path.map(|n| format!("Модель: {}", n)).unwrap_or_else(|| "Перетащите .onnx файл".to_string());

        Self {
            model_session: session,
            model_input_size: input_size,
            class_names: class_names,
            texture: None,
            mask_texture: None,
            detections: Vec::new(),
            img_size: egui::Vec2::ZERO,
            zoom: 1.0,
            hovered_idx: None,
            tx, rx,
            is_processing: false,
            status,
        }
    }

    fn load_model(&mut self, path: PathBuf) {
        println!("Загрузка: {:?}", path);
        match model::YoloModelSession::from_filename_v8(&path) {
            Ok(mut s) => {
                s.probability_threshold = Some(0.25);
                s.iou_threshold = Some(0.45);
                
                let size = if let Some(input) = s.session.inputs.first() {
                     if let ValueType::Tensor { shape, .. } = &input.input_type {
                        (shape[3] as u32, shape[2] as u32)
                     } else { (640, 640) }
                } else { (640, 640) };
                
                let mut names = Vec::new();
                if let Ok(meta) = s.session.metadata() {
                     if let Ok(Some(raw)) = meta.custom("names") {
                        let fixed = raw.replace('\'', "\"").replace("{", "{\"").replace(": ", "\": ").replace(", ", ", \"");
                        if let Ok(map) = serde_json::from_str::<HashMap<String, String>>(&fixed) {
                             let mut keys: Vec<u32> = map.keys().filter_map(|k| k.parse().ok()).collect();
                             keys.sort();
                             let l: Vec<String> = keys.iter().filter_map(|k| map.get(&k.to_string()).cloned()).collect();
                             if !l.is_empty() { names = l; }
                        }
                     }
                }
                
                self.model_session = Some(Arc::new(Mutex::new(s)));
                self.model_input_size = size;
                self.class_names = Arc::new(names);
                self.status = format!("Загружено: {:?}", path.file_name().unwrap());
                println!("Размер входа модели: {:?}", size);
            }
            Err(e) => { 
                self.status = format!("Ошибка: {:?}", e); 
                eprintln!("Load Error: {:?}", e);
            }
        }
    }

    fn run_worker(&self, input: ImageInput, ctx: egui::Context) {
        let model_arc = match &self.model_session {
            Some(m) => Arc::clone(m),
            None => return,
        };
        let names_arc = Arc::clone(&self.class_names);
        let tx = self.tx.clone();
        let (tw, th) = self.model_input_size;
        
        std::thread::spawn(move || {
            let process = || -> Result<DetectionResult, String> {
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

                let input_tensor = Value::from_array((vec![1usize, 3, th as usize, tw as usize], flat_data))
                    .map_err(|e| e.to_string())?;
                
                // --- INFERENCE ---
                let (output0_shape, output0_data, output1_opt) = {
                    let mut m = model_arc.lock().unwrap();
                    let outputs = m.session.run(ort::inputs![input_tensor]).map_err(|e| e.to_string())?;
                    let (s0, d0) = outputs[0].try_extract_tensor::<f32>().map_err(|e| e.to_string())?;
                    
                    let out1 = if outputs.len() > 1 {
                         let (s1, d1) = outputs[1].try_extract_tensor::<f32>().map_err(|e| e.to_string())?;
                         // Сохраняем размеры масок (важно для 512x512!)
                         let mask_dims = (s1[3] as usize, s1[2] as usize); // [batch, 32, h, w] -> (w, h)
                         Some((mask_dims, d1.to_vec()))
                    } else { None };
                    (s0.to_vec(), d0.to_vec(), out1)
                }; 

                // --- PARSING (АВТОМАТИЧЕСКОЕ ОПРЕДЕЛЕНИЕ СТРУКТУРЫ) ---
                let num_rows = output0_shape[1] as usize; 
                let num_anchors = output0_shape[2] as usize;
                
                // Определяем, есть ли маски, по наличию второго выхода
                let has_masks = output1_opt.is_some();
                
                // ВАЖНО: Вычисляем количество классов математически, чтобы не зависеть от JSON
                // Структура rows: [4 box coordinates] + [N classes] + [32 mask coeffs (optional)]
                let num_classes = if has_masks {
                    if num_rows > 36 { num_rows - 4 - 32 } else { 0 }
                } else {
                    if num_rows > 4 { num_rows - 4 } else { 0 }
                };

                // Индекс, где начинаются маски (последние 32 элемента)
                let mask_start_idx = if has_masks { num_rows - 32 } else { 0 };

                let mut candidates = Vec::new();

                for i in 0..num_anchors {
                    let mut max_conf = 0.0f32;
                    let mut class_id = 0;
                    
                    // Классы идут с индекса 4
                    for c in 0..num_classes {
                        let conf = output0_data[(4 + c) * num_anchors + i];
                        if conf > max_conf { max_conf = conf; class_id = c; }
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
                            // Безопасное чтение масок
                            for m in 0..MASK_COEFFS_NUM {
                                let data_idx = (mask_start_idx + m) * num_anchors + i;
                                if data_idx < output0_data.len() {
                                    mask_coeffs.push(output0_data[data_idx]);
                                }
                            }
                        }

                        // Если маски должны быть, но их не считали - не добавляем кандидата (чтобы не упасть)
                        if has_masks && mask_coeffs.len() != MASK_COEFFS_NUM {
                            continue;
                        }

                        candidates.push(Candidate {
                            det: YoloEntityOutput {
                                bounding_box: BoundingBox { x1: cx - w / 2.0, y1: cy - h / 2.0, x2: cx + w / 2.0, y2: cy + h / 2.0 },
                                confidence: max_conf,
                                label: arcstr::ArcStr::from(label),
                            },
                            mask_coeffs,
                            class_id,
                        });
                    }
                }

                let kept_indices = perform_nms_indices(&candidates, 0.45);
                let final_detections: Vec<YoloEntityOutput> = kept_indices.iter().map(|&i| candidates[i].det.clone()).collect();
                
                println!("Objects found: {}", final_detections.len());

                // --- GENERATE MASKS ---
                let mask_image = if has_masks && !kept_indices.is_empty() {
                    if let Some((mask_dims, proto_data)) = output1_opt {
                        let kept_candidates: Vec<&Candidate> = kept_indices.iter().map(|&i| &candidates[i]).collect();
                        
                        // Передаем динамические размеры маски (для 512x512 это будет 128x128)
                        Some(process_masks(
                            &kept_candidates, 
                            &proto_data, 
                            (tw as usize, th as usize), // Model input size (e.g. 512, 512)
                            mask_dims                   // Mask proto size (e.g. 128, 128)
                        ))
                    } else { None }
                } else { None };

                Ok(DetectionResult {
                    texture_data: load_egui_image(&original_img),
                    mask_texture: mask_image,
                    detections: final_detections,
                    img_size: egui::vec2(img_w, img_h),
                })
            };

            match process() {
                Ok(res) => { let _ = tx.send(AppMessage::Success(res)); },
                Err(e) => { 
                    eprintln!("Worker Panic/Error: {}", e);
                    let _ = tx.send(AppMessage::Error(e)); 
                }
            }
            ctx.request_repaint();
        });
    }

    fn handle_clipboard(&mut self, ctx: &egui::Context) {
        if let Ok(mut cb) = arboard::Clipboard::new() {
             if let Ok(img_data) = cb.get_image() {
                  let bytes = img_data.bytes.into_owned();
                  if let Some(rgba) = RgbaImage::from_raw(img_data.width as u32, img_data.height as u32, bytes) {
                      self.is_processing = true;
                      self.status = "Обработка...".to_string();
                      self.run_worker(ImageInput::Pixels(DynamicImage::ImageRgba8(rgba)), ctx.clone());
                  }
             } else if let Ok(text) = cb.get_text() {
                 let path = PathBuf::from(text.trim_matches('"').trim());
                 if path.exists() {
                     self.is_processing = true;
                     self.status = "Файл...".to_string();
                     self.run_worker(ImageInput::File(path), ctx.clone());
                 }
             }
        }
   }
}

// --- ОТРИСОВКА МАСОК (ДИНАМИЧЕСКИЙ РАЗМЕР) ---
fn process_masks(
    candidates: &[&Candidate], 
    proto_data: &[f32], 
    model_size: (usize, usize), // (512, 512) or (640, 640)
    mask_proto_dim: (usize, usize) // (128, 128) or (160, 160)
) -> egui::ColorImage {
    let (mw, mh) = mask_proto_dim;
    let proto_len = mw * mh;

    // Буфер в разрешении масок
    let mut mask_buffer = ImageBuffer::<Rgba<u8>, Vec<u8>>::new(mw as u32, mh as u32);
    
    let colors: Vec<[u8; 4]> = candidates.iter().map(|c| {
        let col = get_color_raw(&c.det.label);
        [col[0], col[1], col[2], 220] 
    }).collect();

    // Коэффициент масштаба (например, 128 / 512 = 0.25)
    let scale_x = mw as f32 / model_size.0 as f32;
    let scale_y = mh as f32 / model_size.1 as f32;

    for (idx, cand) in candidates.iter().enumerate() {
        // Пропускаем, если коэффициенты не совпали (защита от краша)
        if cand.mask_coeffs.len() != MASK_COEFFS_NUM { continue; }

        let b = &cand.det.bounding_box;
        let color = colors[idx];
        let coeffs = &cand.mask_coeffs;

        // BBox в координатах модели -> координаты маски
        let x1 = (b.x1 * scale_x).floor().max(0.0) as u32;
        let y1 = (b.y1 * scale_y).floor().max(0.0) as u32;
        let x2 = (b.x2 * scale_x).ceil().min(mw as f32) as u32;
        let y2 = (b.y2 * scale_y).ceil().min(mh as f32) as u32;

        if x2 <= x1 || y2 <= y1 { continue; }

        for y in y1..y2 {
            for x in x1..x2 {
                let offset = (y as usize) * mw + (x as usize);
                
                // Защита от выхода за границы массива прототипов
                if offset >= proto_data.len() / MASK_COEFFS_NUM { continue; }

                let mut sum = 0.0f32;
                for k in 0..MASK_COEFFS_NUM {
                    // Индекс в плоском массиве прототипов [32, H, W]
                    // k-й канал, пиксель offset
                    let proto_idx = k * proto_len + offset;
                    if proto_idx < proto_data.len() {
                        sum += coeffs[k] * proto_data[proto_idx];
                    }
                }
                
                let mask_val = 1.0 / (1.0 + (-sum).exp());

                if mask_val > 0.5 {
                    mask_buffer.put_pixel(x, y, Rgba(color));
                }
            }
        }
    }
    
    egui::ColorImage::from_rgba_unmultiplied(
        [mask_buffer.width() as _, mask_buffer.height() as _], 
        mask_buffer.as_flat_samples().as_slice()
    )
}

fn perform_nms_indices(candidates: &Vec<Candidate>, iou_threshold: f32) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..candidates.len()).collect();
    indices.sort_by(|&i, &j| candidates[j].det.confidence.partial_cmp(&candidates[i].det.confidence).unwrap());
    let mut active = vec![true; candidates.len()];
    let mut kept = Vec::new();
    for &i in &indices {
        if !active[i] { continue; }
        kept.push(i);
        let a = &candidates[i].det.bounding_box;
        let area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
        for &j in &indices {
            if i == j || !active[j] { continue; }
            let b = &candidates[j].det.bounding_box;
            let ix1 = a.x1.max(b.x1); let iy1 = a.y1.max(b.y1);
            let ix2 = a.x2.min(b.x2); let iy2 = a.y2.min(b.y2);
            if ix2 < ix1 || iy2 < iy1 { continue; }
            let inter = (ix2 - ix1) * (iy2 - iy1);
            let union = area_a + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
            if union > 0.0 && (inter / union) > iou_threshold { active[j] = false; }
        }
    }
    kept
}

fn get_color_raw(label: &str) -> [u8; 3] {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    label.hash(&mut h);
    let hue = (h.finish() % 360) as f32 / 360.0;
    let (s, v) = (1.0, 1.0);
    let c = v * s;
    let x = c * (1.0 - ((hue * 6.0) % 2.0 - 1.0).abs());
    let m = v - c;
    let (r, g, b) = match (hue * 6.0) as i32 {
        0 => (c, x, 0.0), 1 => (x, c, 0.0), 2 => (0.0, c, x),
        3 => (0.0, x, c), 4 => (x, 0.0, c), _ => (c, 0.0, x),
    };
    [((r + m) * 255.0) as u8, ((g + m) * 255.0) as u8, ((b + m) * 255.0) as u8]
}

fn get_color(label: &str) -> egui::Color32 {
    let rgb = get_color_raw(label);
    egui::Color32::from_rgb(rgb[0], rgb[1], rgb[2])
}

fn load_egui_image(img: &DynamicImage) -> egui::ColorImage {
    let size = [img.width() as _, img.height() as _];
    egui::ColorImage::from_rgba_unmultiplied(size, img.to_rgba8().as_flat_samples().as_slice())
}

fn get_coco_names() -> Vec<String> {
    vec!["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
        .iter().map(|s| s.to_string()).collect()
}

impl eframe::App for YoloGuiApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if let Ok(msg) = self.rx.try_recv() {
            match msg {
                AppMessage::Success(res) => {
                    self.texture = Some(ctx.load_texture("img", res.texture_data, Default::default()));
                    self.mask_texture = res.mask_texture.map(|m| ctx.load_texture("masks", m, egui::TextureOptions::LINEAR));
                    self.detections = res.detections;
                    self.img_size = res.img_size;
                    self.status = format!("Найдено: {} | Маски: {}", self.detections.len(), if self.mask_texture.is_some() {"OK"} else {"-"});
                },
                AppMessage::Error(e) => self.status = format!("Ошибка: {}", e),
            }
            self.is_processing = false;
        }

        if !ctx.input(|i| i.raw.dropped_files.is_empty()) {
            let dropped = ctx.input(|i| i.raw.dropped_files.first().cloned());
             if let Some(d) = dropped {
                 if let Some(path) = d.path {
                    if let Some(ext) = path.extension() {
                        if ext.to_string_lossy().to_lowercase() == "onnx" {
                            self.load_model(path);
                        } else if self.model_session.is_some() {
                            self.is_processing = true;
                            self.run_worker(ImageInput::File(path), ctx.clone());
                        }
                    }
                 }
             }
        }
        if ctx.input_mut(|i| i.consume_shortcut(&egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, egui::Key::V))) {
            if self.model_session.is_some() { self.handle_clipboard(ctx); }
        }

        let z_delta = ctx.input(|i| i.zoom_delta());
        if z_delta != 1.0 { self.zoom = (self.zoom * z_delta).clamp(0.1, 20.0); }

        egui::SidePanel::right("side").width_range(160.0..=250.0).show(ctx, |ui| {
            ui.heading("Объекты");
            ui.separator();
            egui::ScrollArea::vertical().show(ui, |ui| {
                let mut lh = None;
                for (i, det) in self.detections.iter().enumerate() {
                    let col = get_color(&det.label);
                    ui.horizontal(|ui| {
                        let (r, _) = ui.allocate_exact_size(egui::vec2(10.0, 10.0), egui::Sense::hover());
                        ui.painter().circle_filled(r.center(), 4.0, col);
                        if ui.selectable_label(self.hovered_idx == Some(i), format!("{} {:.0}%", det.label, det.confidence * 100.0)).hovered() {
                            lh = Some(i);
                        }
                    });
                }
                if lh.is_some() { self.hovered_idx = lh; }
            });
        });

        egui::TopBottomPanel::bottom("status").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if self.is_processing { ui.spinner(); }
                ui.label(&self.status);
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            if self.model_session.is_none() {
                ui.centered_and_justified(|ui| { ui.heading("Загрузите модель (перетащите .onnx)"); });
                return;
            }

            if let Some(tex) = &self.texture {
                let display_size = self.img_size * self.zoom;
                egui::ScrollArea::both().show(ui, |ui| {
                    let (rect, resp) = ui.allocate_exact_size(display_size, egui::Sense::hover());
                    
                    // 1. Рисуем фото
                    ui.painter().image(tex.id(), rect, egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)), egui::Color32::WHITE);
                    
                    // 2. Рисуем МАСКИ (Overlay)
                    if let Some(m) = &self.mask_texture {
                        ui.painter().image(m.id(), rect, egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)), egui::Color32::WHITE);
                    }

                    let mut h_id = None;
                    if let Some(ptr) = resp.hover_pos() {
                        let sx = (self.img_size.x / self.model_input_size.0 as f32) * self.zoom;
                        let sy = (self.img_size.y / self.model_input_size.1 as f32) * self.zoom;
                        for (i, d) in self.detections.iter().enumerate().rev() {
                            let b = &d.bounding_box;
                            let r = egui::Rect::from_min_max(
                                egui::pos2(b.x1 * sx, b.y1 * sy) + rect.min.to_vec2(),
                                egui::pos2(b.x2 * sx, b.y2 * sy) + rect.min.to_vec2(),
                            );
                            if r.contains(ptr) { h_id = Some(i); break; }
                        }
                    }
                    if h_id.is_some() { self.hovered_idx = h_id; }

                    // 3. Рисуем ОБВОДКУ И ЗАЛИВКУ БОКСА
                    let p = ui.painter();
                    let sx = (self.img_size.x / self.model_input_size.0 as f32) * self.zoom;
                    let sy = (self.img_size.y / self.model_input_size.1 as f32) * self.zoom;
                    
                    for (i, d) in self.detections.iter().enumerate() {
                        let b = &d.bounding_box;
                        let col = get_color(&d.label);
                        let is_h = self.hovered_idx == Some(i);
                        let c = if is_h { col.linear_multiply(1.5) } else { col };
                        
                        let r = egui::Rect::from_min_max(
                            egui::pos2(b.x1 * sx, b.y1 * sy) + rect.min.to_vec2(),
                            egui::pos2(b.x2 * sx, b.y2 * sy) + rect.min.to_vec2(),
                        );
                        
                        // А) ЗАЛИВКА БОКСА (Возвращена, 15% прозрачности)
                        p.rect_filled(r, 0.0, c.linear_multiply(0.15));
                        
                        // Б) ОБВОДКА
                        p.rect_stroke(r, 0.0, egui::Stroke::new(if is_h {4.0} else {2.0}, c));

                        // В) ТЕКСТ (Сверху)
                        if is_h || self.zoom > 0.4 {
                             let t = format!("{} {:.0}%", d.label, d.confidence * 100.0);
                             let f = egui::FontId::proportional((13.0*self.zoom).clamp(10.0, 24.0));
                             let g = p.layout_no_wrap(t, f, egui::Color32::WHITE);
                             let mut lp = r.min;
                             if lp.y - g.size().y - 6.0 < rect.min.y { lp.y += 2.0; } else { lp.y -= g.size().y + 4.0; }
                             let lr = egui::Rect::from_min_size(lp, g.size() + egui::vec2(8.0, 4.0));
                             
                             let rounding = egui::Rounding { nw: 4.0, ne: 4.0, sw: 0.0, se: 0.0 };
                             p.rect_filled(lr, rounding, c);
                             p.galley(lr.min + egui::vec2(4.0, 2.0), g, egui::Color32::WHITE);
                        }
                    }
                });
            } else {
                ui.centered_and_justified(|ui| ui.label("Перетащите картинку"));
            }
        });
    }
}

fn main() -> eframe::Result<()> {
    eframe::run_native(
        "YOLO Ultimate Viewer",
        eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default().with_inner_size([1200.0, 850.0]).with_drag_and_drop(true),
            ..Default::default()
        },
        Box::new(|cc| Ok(Box::new(YoloGuiApp::new(cc)))),
    )
}
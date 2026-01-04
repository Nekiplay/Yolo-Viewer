#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

use eframe::egui;
use image::{DynamicImage, RgbaImage, ImageBuffer, Rgba, Luma, imageops::FilterType};
use std::sync::{Arc, Mutex, mpsc};
use std::path::PathBuf;
use std::collections::HashMap;
use ort::value::{Value, ValueType};
use yolo_rs::{YoloEntityOutput, model, BoundingBox};

const MASK_PROTO_SIZE: usize = 160;
const MASK_COEFFS_NUM: usize = 32;

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
    #[allow(dead_code)]
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
    pan: egui::Vec2,
    fit_to_screen_req: bool,

    hovered_idx: Option<usize>,
    
    tx: mpsc::Sender<AppMessage>,
    rx: mpsc::Receiver<AppMessage>,
    
    is_processing: bool,
    status: String,
}

impl YoloGuiApp {
    fn new(_cc: &eframe::CreationContext<'_>) -> Self {
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

        let (input_size, class_names) = if let Some(s) = &session {
            let guard = s.lock().unwrap();
            let size = if let Some(input) = guard.session.inputs.first() {
                if let ValueType::Tensor { shape, .. } = &input.input_type {
                    (shape[3] as u32, shape[2] as u32)
                } else { (640, 640) }
            } else { (640, 640) };
            
            let mut names = Vec::new();
            if let Ok(meta) = guard.session.metadata() {
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
            pan: egui::Vec2::ZERO,
            fit_to_screen_req: false,
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
                println!("Вход модели: {:?}", size);
            }
            Err(e) => { 
                self.status = format!("Ошибка: {:?}", e); 
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
                
                // INFERENCE
                let (output0_shape, output0_data, output1_opt) = {
                    let mut m = model_arc.lock().unwrap();
                    let outputs = m.session.run(ort::inputs![input_tensor]).map_err(|e| e.to_string())?;
                    let (s0, d0) = outputs[0].try_extract_tensor::<f32>().map_err(|e| e.to_string())?;
                    
                    let out1 = if outputs.len() > 1 {
                         let (s1, d1) = outputs[1].try_extract_tensor::<f32>().map_err(|e| e.to_string())?;
                         let mask_dims = (s1[3] as usize, s1[2] as usize); 
                         Some((mask_dims, d1.to_vec()))
                    } else { None };
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
                            for m in 0..MASK_COEFFS_NUM {
                                let idx = (mask_start_idx + m) * num_anchors + i;
                                if idx < output0_data.len() {
                                    mask_coeffs.push(output0_data[idx]);
                                }
                            }
                        }
                        if has_masks && mask_coeffs.len() != MASK_COEFFS_NUM { continue; }

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
                
                println!("Найдены объекты: {}", final_detections.len());

                let mask_image = if has_masks && !kept_indices.is_empty() {
                    if let Some((mask_dims, proto_data)) = output1_opt {
                        let kept_candidates: Vec<&Candidate> = kept_indices.iter().map(|&i| &candidates[i]).collect();
                        Some(process_masks(&kept_candidates, &proto_data, (tw as usize, th as usize), mask_dims))
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
                    eprintln!("Error: {}", e);
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

fn process_masks(
    candidates: &[&Candidate], 
    proto_data: &[f32], 
    model_size: (usize, usize), 
    mask_proto_dim: (usize, usize) 
) -> egui::ColorImage {
    let (mw, mh) = mask_proto_dim;
    let (tw, th) = model_size;
    let proto_len = mw * mh;

    const SCALE_FACTOR: f32 = 4.0; 
    let super_w = (tw as f32 * SCALE_FACTOR) as u32;
    let super_h = (th as f32 * SCALE_FACTOR) as u32;

    let mut final_buffer = ImageBuffer::<Rgba<u8>, Vec<u8>>::new(super_w, super_h);
  
    let scale_x_proto = mw as f32 / tw as f32;
    let scale_y_proto = mh as f32 / th as f32;

    for cand in candidates.iter() {
        if cand.mask_coeffs.len() != MASK_COEFFS_NUM { continue; }

        let b = &cand.det.bounding_box;
        
        let mx1 = (b.x1 * scale_x_proto).floor().max(0.0) as u32;
        let my1 = (b.y1 * scale_y_proto).floor().max(0.0) as u32;
        let mx2 = (b.x2 * scale_x_proto).ceil().min(mw as f32) as u32;
        let my2 = (b.y2 * scale_y_proto).ceil().min(mh as f32) as u32;

        if mx2 <= mx1 || my2 <= my1 { continue; }
        
        let patch_w = mx2 - mx1;
        let patch_h = my2 - my1;

        let mut float_patch = ImageBuffer::<Luma<f32>, Vec<f32>>::new(patch_w, patch_h);
        
        for py in 0..patch_h {
            for px in 0..patch_w {
                let y = my1 + py;
                let x = mx1 + px;
                let offset = (y as usize) * mw + (x as usize);
                
                if offset * MASK_COEFFS_NUM >= proto_data.len() { continue; }

                let mut sum = 0.0f32;
                for k in 0..MASK_COEFFS_NUM {
                    let proto_idx = k * proto_len + offset;
                    sum += cand.mask_coeffs[k] * proto_data[proto_idx];
                }
                let val = 1.0 / (1.0 + (-sum).exp());
                float_patch.put_pixel(px, py, Luma([val]));
            }
        }

        let target_x1 = (b.x1 * SCALE_FACTOR).max(0.0) as u32;
        let target_y1 = (b.y1 * SCALE_FACTOR).max(0.0) as u32;
        
        let target_x2 = (b.x2 * SCALE_FACTOR).min(super_w as f32) as u32;
        let target_y2 = (b.y2 * SCALE_FACTOR).min(super_h as f32) as u32;
        
        let target_w = target_x2.saturating_sub(target_x1);
        let target_h = target_y2.saturating_sub(target_y1);

        if target_w == 0 || target_h == 0 { continue; }

        let resized_patch = image::imageops::resize(
            &float_patch, 
            target_w, 
            target_h, 
            FilterType::Triangle 
        );

        let col_rgb = get_color_raw(&cand.det.label);
        
        for py in 0..target_h {
            for px in 0..target_w {
                let val = resized_patch.get_pixel(px, py)[0];
                
                if val > 0.5 {
                    let gx = target_x1 + px;
                    let gy = target_y1 + py;
                    
                    if gx < super_w && gy < super_h {
                        // Anti-aliasing
                        let alpha = ((val - 0.5) * 2.0 * 180.0).clamp(0.0, 180.0) as u8;
                        
                        let pixel = final_buffer.get_pixel_mut(gx, gy);
                        if pixel[3] < alpha {
                            *pixel = Rgba([col_rgb[0], col_rgb[1], col_rgb[2], alpha]);
                        }
                    }
                }
            }
        }
    }
    
    egui::ColorImage::from_rgba_unmultiplied(
        [final_buffer.width() as _, final_buffer.height() as _], 
        final_buffer.as_flat_samples().as_slice()
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

impl eframe::App for YoloGuiApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if let Ok(msg) = self.rx.try_recv() {
            match msg {
                AppMessage::Success(res) => {
                    self.texture = Some(ctx.load_texture("img", res.texture_data, Default::default()));
                    // Текстура теперь высокого разрешения (1280x1280), ставим LINEAR чтобы при уменьшении было красиво
                    self.mask_texture = res.mask_texture.map(|m| ctx.load_texture("masks", m, egui::TextureOptions::LINEAR));
                    self.detections = res.detections;
                    self.img_size = res.img_size;
                    self.status = format!("Найдено: {} | Маски: {}", self.detections.len(), if self.mask_texture.is_some() {"OK"} else {"-"});
                    self.fit_to_screen_req = true;
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
        
        if ctx.input_mut(|i| i.consume_shortcut(&egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, egui::Key::V))) {
            if self.model_session.is_some() { self.handle_clipboard(ctx); }
        }

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
                let (response, painter) = ui.allocate_painter(ui.available_size(), egui::Sense::click_and_drag());
                let viewport_rect = response.rect;

                if self.fit_to_screen_req {
                    let w_ratio = viewport_rect.width() / self.img_size.x;
                    let h_ratio = viewport_rect.height() / self.img_size.y;
                    self.zoom = w_ratio.min(h_ratio) * 0.95;
                    let content_size = self.img_size * self.zoom;
                    self.pan = viewport_rect.min.to_vec2() + (viewport_rect.size() - content_size) / 2.0;
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

                if response.dragged_by(egui::PointerButton::Primary) || response.dragged_by(egui::PointerButton::Middle) {
                    self.pan += response.drag_delta();
                }

                let displayed_rect = egui::Rect::from_min_size(self.pan.to_pos2(), self.img_size * self.zoom);
                let painter = painter.with_clip_rect(viewport_rect);

                painter.image(tex.id(), displayed_rect, egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)), egui::Color32::WHITE);
                
                if let Some(m) = &self.mask_texture {
                    painter.image(m.id(), displayed_rect, egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)), egui::Color32::WHITE);
                }

                let mut h_id = None;
                if let Some(ptr) = response.hover_pos() {
                    let relative_pos = (ptr - self.pan) / self.zoom;
                    if relative_pos.x >= 0.0 && relative_pos.y >= 0.0 && relative_pos.x <= self.img_size.x && relative_pos.y <= self.img_size.y {
                           let sx = self.img_size.x / self.model_input_size.0 as f32;
                           let sy = self.img_size.y / self.model_input_size.1 as f32;
                           for (i, d) in self.detections.iter().enumerate().rev() {
                               let b = &d.bounding_box;
                               let bx1 = b.x1 * sx; let by1 = b.y1 * sy;
                               let bx2 = b.x2 * sx; let by2 = b.y2 * sy;
                               if relative_pos.x >= bx1 && relative_pos.x <= bx2 && relative_pos.y >= by1 && relative_pos.y <= by2 {
                                   h_id = Some(i);
                                   break;
                               }
                           }
                       }
                }
                if h_id.is_some() { self.hovered_idx = h_id; }

                let sx = (self.img_size.x / self.model_input_size.0 as f32) * self.zoom;
                let sy = (self.img_size.y / self.model_input_size.1 as f32) * self.zoom;

                for (i, d) in self.detections.iter().enumerate() {
                    let b = &d.bounding_box;
                    let col = get_color(&d.label);
                    let is_h = self.hovered_idx == Some(i);
                    let c = if is_h { col.linear_multiply(1.5) } else { col };
                    let r = egui::Rect::from_min_max(
                        self.pan.to_pos2() + egui::vec2(b.x1 * sx, b.y1 * sy),
                        self.pan.to_pos2() + egui::vec2(b.x2 * sx, b.y2 * sy),
                    );
                    if painter.clip_rect().intersects(r) {
                        painter.rect_stroke(r, 0.0, egui::Stroke::new(if is_h {3.0} else {1.5}, c));
                        if is_h || self.zoom > 0.4 {
                             let t = format!("{} {:.0}%", d.label, d.confidence * 100.0);
                             let f = egui::FontId::proportional((13.0*self.zoom).clamp(10.0, 24.0));
                             let g = painter.layout_no_wrap(t, f, egui::Color32::WHITE);
                             let mut lp = r.min;
                             if lp.y - g.size().y - 6.0 < displayed_rect.min.y { lp.y += 2.0; } else { lp.y -= g.size().y + 4.0; }
                             let lr = egui::Rect::from_min_size(lp, g.size() + egui::vec2(8.0, 4.0));
                             painter.rect_filled(lr, egui::Rounding::same(4.0), c);
                             painter.galley(lr.min + egui::vec2(4.0, 2.0), g, egui::Color32::WHITE);
                        }
                    }
                }
            } else {
                ui.centered_and_justified(|ui| ui.label("Перетащите картинку или Ctrl+V"));
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
use image::{DynamicImage, ImageBuffer, Luma, Rgba, imageops::FilterType};
use egui;
use crate::Candidate;

const MASK_COEFFS_NUM: usize = 32;

pub fn process_masks(
    candidates: &[&Candidate],
    proto_data: &[f32],
    model_size: (usize, usize),
    mask_proto_dim: (usize, usize),
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
        if cand.mask_coeffs.len() != MASK_COEFFS_NUM {
            continue;
        }

        let b = &cand.det.bounding_box;

        let mx1 = (b.x1 * scale_x_proto).floor().max(0.0) as u32;
        let my1 = (b.y1 * scale_y_proto).floor().max(0.0) as u32;
        let mx2 = (b.x2 * scale_x_proto).ceil().min(mw as f32) as u32;
        let my2 = (b.y2 * scale_y_proto).ceil().min(mh as f32) as u32;

        if mx2 <= mx1 || my2 <= my1 {
            continue;
        }

        let patch_w = mx2 - mx1;
        let patch_h = my2 - my1;

        let mut float_patch = ImageBuffer::<Luma<f32>, Vec<f32>>::new(patch_w, patch_h);

        for py in 0..patch_h {
            for px in 0..patch_w {
                let y = my1 + py;
                let x = mx1 + px;
                let offset = (y as usize) * mw + (x as usize);

                if offset * MASK_COEFFS_NUM >= proto_data.len() {
                    continue;
                }

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

        if target_w == 0 || target_h == 0 {
            continue;
        }

        let resized_patch =
            image::imageops::resize(&float_patch, target_w, target_h, FilterType::Triangle);

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
        final_buffer.as_flat_samples().as_slice(),
    )
}

pub fn perform_nms_indices(candidates: &Vec<Candidate>, iou_threshold: f32) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..candidates.len()).collect();
    indices.sort_by(|&i, &j| {
        candidates[j]
            .det
            .confidence
            .partial_cmp(&candidates[i].det.confidence)
            .unwrap()
    });
    let mut active = vec![true; candidates.len()];
    let mut kept = Vec::new();
    for &i in &indices {
        if !active[i] {
            continue;
        }
        kept.push(i);
        let a = &candidates[i].det.bounding_box;
        let area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
        for &j in &indices {
            if i == j || !active[j] {
                continue;
            }
            let b = &candidates[j].det.bounding_box;
            let ix1 = a.x1.max(b.x1);
            let iy1 = a.y1.max(b.y1);
            let ix2 = a.x2.min(b.x2);
            let iy2 = a.y2.min(b.y2);
            if ix2 < ix1 || iy2 < iy1 {
                continue;
            }
            let inter = (ix2 - ix1) * (iy2 - iy1);
            let union = area_a + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
            if union > 0.0 && (inter / union) > iou_threshold {
                active[j] = false;
            }
        }
    }
    kept
}

pub fn get_color_raw(label: &str) -> [u8; 3] {
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
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    [
        ((r + m) * 255.0) as u8,
        ((g + m) * 255.0) as u8,
        ((b + m) * 255.0) as u8,
    ]
}

pub fn get_color(label: &str) -> egui::Color32 {
    let rgb = get_color_raw(label);
    egui::Color32::from_rgb(rgb[0], rgb[1], rgb[2])
}

pub fn load_egui_image(img: &DynamicImage) -> egui::ColorImage {
    let size = [img.width() as _, img.height() as _];
    egui::ColorImage::from_rgba_unmultiplied(size, img.to_rgba8().as_flat_samples().as_slice())
}

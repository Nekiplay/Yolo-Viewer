#![windows_subsystem = "windows"]

use yolo_viewer::YoloGuiApp;

fn main() -> eframe::Result<()> {
    eframe::run_native(
        "YOLO Ultimate Viewer",
        eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default()
                .with_inner_size([1200.0, 850.0])
                .with_drag_and_drop(true),
            ..Default::default()
        },
        Box::new(|cc| Ok(Box::new(YoloGuiApp::new(cc)))),
    )
}

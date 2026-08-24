use std::path::PathBuf;
use std::time::Instant;

use anyhow::Context;

use egui_wgpu::{Renderer as EguiRenderer, ScreenDescriptor};
use egui_winit::State as EguiWinitState;

use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};

use egui::ViewportId;

use crate::{converter, export, mesh_loader};

pub fn run_gui() -> anyhow::Result<()> {
    env_logger::init();

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Wait);

    let mut app = App::default();
    event_loop.run_app(&mut app)?;

    Ok(())
}

#[derive(Default)]
struct App {
    window: Option<Window>,
    window_id: Option<WindowId>,

    // wgpu
    instance: Option<wgpu::Instance>,
    surface: Option<wgpu::Surface<'static>>,
    adapter: Option<wgpu::Adapter>,
    device: Option<wgpu::Device>,
    queue: Option<wgpu::Queue>,
    config: Option<wgpu::SurfaceConfiguration>,

    // egui
    egui_ctx: egui::Context,
    egui_state: Option<EguiWinitState>,
    egui_renderer: Option<EguiRenderer>,

    // state
    input_path: Option<PathBuf>,
    output_dir: Option<PathBuf>,
    density: f32,
    scale: f32,
    export_ply: bool,
    export_splat: bool,
    status: String,
    error: Option<String>,
    last_ms: Option<f32>,
    last_count: Option<usize>,

    gaussians: Option<Vec<converter::Gaussian>>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = event_loop
            .create_window(
                Window::default_attributes()
                    .with_title("Mesh2Splat (macOS)")
                    .with_inner_size(PhysicalSize::new(1200, 800)),
            )
            .expect("create window");

        let window_id = window.id();

        self.density = if self.density == 0.0 { 1.0 } else { self.density };
        self.scale = if self.scale == 0.0 { 0.65 } else { self.scale };
        self.export_ply = true;
        self.export_splat = true;

        if let Err(e) = self.init_gpu_and_egui(&window) {
            self.error = Some(format!("{e:?}"));
            self.status = "Failed to init GPU".to_string();
        } else {
            self.status = "Ready. Drop a .glb/.gltf or click Open…".to_string();
        }

        self.window_id = Some(window_id);
        self.window = Some(window);

        event_loop.set_control_flow(ControlFlow::Wait);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        if Some(window_id) != self.window_id {
            return;
        }

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
                return;
            }
            WindowEvent::Resized(size) => {
                self.resize(size);
            }
            WindowEvent::DroppedFile(path) => {
                self.input_path = Some(path.clone());
                self.error = None;
                self.status = format!("Selected: {}", path.display());
            }
            WindowEvent::RedrawRequested => {
                let window = match self.window.as_ref() {
                    Some(w) => w,
                    None => return,
                };

                if let Some(egui_state) = self.egui_state.as_mut() {
                    // Feed the event to egui (for completeness you’d route all events; ok for now)
                    // NOTE: for full input handling, call egui_state.on_window_event on every event.
                    let _ = egui_state.on_window_event(window, &WindowEvent::RedrawRequested);
                }

                if let Err(e) = self.render(window) {
                    self.error = Some(format!("{e:?}"));
                    self.status = "Render error".to_string();
                }
                return;
            }
            _ => {}
        }

        // For proper egui input, forward every event:
        if let (Some(window), Some(egui_state)) = (self.window.as_ref(), self.egui_state.as_mut()) {
            let _ = egui_state.on_window_event(window, &event);
            window.request_redraw();
        }
    }
}

impl App {
    fn init_gpu_and_egui(&mut self, window: &Window) -> anyhow::Result<()> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());

        // wgpu 22 pattern: create_surface(window)
        let surface = unsafe { instance.create_surface(window) }?;

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .context("No suitable GPU adapter found")?;

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
            },
            None,
        ))?;

        let size = window.inner_size();
        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: caps.present_modes[0],
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        // egui-winit 0.29.1 signature (your compiler output)
        let viewport_id = ViewportId::ROOT;
        let egui_state = EguiWinitState::new(
            self.egui_ctx.clone(),
            viewport_id,
            window,
            None,
            None,
            None,
        );

        // egui-wgpu 0.29.1 signature (your compiler output: extra bool)
        let egui_renderer = EguiRenderer::new(&device, format, None, 1, false);

        self.instance = Some(instance);
        self.surface = Some(surface);
        self.adapter = Some(adapter);
        self.device = Some(device);
        self.queue = Some(queue);
        self.config = Some(config);

        self.egui_state = Some(egui_state);
        self.egui_renderer = Some(egui_renderer);

        Ok(())
    }

    fn resize(&mut self, size: PhysicalSize<u32>) {
        let (Some(surface), Some(device), Some(config)) =
            (self.surface.as_ref(), self.device.as_ref(), self.config.as_mut())
        else {
            return;
        };

        config.width = size.width.max(1);
        config.height = size.height.max(1);
        surface.configure(device, config);
    }

    fn render(&mut self, window: &Window) -> anyhow::Result<()> {
        let device = self.device.as_ref().unwrap();
        let queue = self.queue.as_ref().unwrap();
        let surface = self.surface.as_ref().unwrap();
        let config = self.config.as_ref().unwrap();

        let egui_state = self.egui_state.as_mut().unwrap();
        let egui_renderer = self.egui_renderer.as_mut().unwrap();

        let frame = match surface.get_current_texture() {
            Ok(f) => f,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                self.resize(window.inner_size());
                return Ok(());
            }
            Err(e) => return Err(anyhow::anyhow!(e)),
        };

        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Start egui pass (0.29 uses begin_frame/end_frame; yes deprecated warnings are fine)
        let raw_input = egui_state.take_egui_input(window);
        self.egui_ctx.begin_frame(raw_input);

        // Collect actions first (avoid borrow checker issues)
        enum Action {
            PickInput,
            PickOutput,
            Convert,
            Export,
        }
        let mut actions: Vec<Action> = Vec::new();

        egui::TopBottomPanel::top("top").show(&self.egui_ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("Mesh2Splat");
                ui.separator();

                if ui.button("Open…").clicked() {
                    actions.push(Action::PickInput);
                }
                if ui.button("Output folder…").clicked() {
                    actions.push(Action::PickOutput);
                }

                ui.separator();
                ui.checkbox(&mut self.export_ply, "PLY");
                ui.checkbox(&mut self.export_splat, "SPLAT");
            });
        });

        egui::SidePanel::left("left").default_width(360.0).show(&self.egui_ctx, |ui| {
            ui.label("Input:");
            match &self.input_path {
                Some(p) => {
                    ui.monospace(p.display().to_string());
                }
                None => {
                    ui.colored_label(egui::Color32::GRAY, "No file selected");
                }
            }

            ui.add_space(10.0);
            ui.add(egui::Slider::new(&mut self.density, 0.1..=2.0).text("Density"));
            ui.add(egui::Slider::new(&mut self.scale, 0.1..=2.0).text("Scale"));

            ui.add_space(10.0);

            let can_convert = self.input_path.is_some();
            if ui.add_enabled(can_convert, egui::Button::new("Convert")).clicked() {
                actions.push(Action::Convert);
            }

            let can_export = self.gaussians.is_some();
            if ui.add_enabled(can_export, egui::Button::new("Export")).clicked() {
                actions.push(Action::Export);
            }

            ui.separator();
            ui.label("Status:");
            ui.label(&self.status);

            if let Some(ms) = self.last_ms {
                ui.label(format!("Last: {:.2} ms", ms));
            }
            if let Some(n) = self.last_count {
                ui.label(format!("Gaussians: {}", n));
            }
            if let Some(err) = &self.error {
                ui.add_space(8.0);
                ui.colored_label(egui::Color32::LIGHT_RED, err);
            }
        });

        egui::CentralPanel::default().show(&self.egui_ctx, |ui| {
            ui.heading("Preview");
            ui.colored_label(egui::Color32::GRAY, "Preview placeholder.");
            ui.add_space(6.0);
            ui.label("Drop a file, Convert, then Export.");
        });

        // Apply actions (now we can mutably touch self safely)
        for a in actions {
            match a {
                Action::PickInput => {
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("glTF/GLB", &["glb", "gltf"])
                        .pick_file()
                    {
                        self.input_path = Some(path.clone());
                        self.error = None;
                        self.status = format!("Selected: {}", path.display());
                    }
                }
                Action::PickOutput => {
                    if let Some(dir) = rfd::FileDialog::new().pick_folder() {
                        self.output_dir = Some(dir.clone());
                        self.status = format!("Output dir: {}", dir.display());
                    }
                }
                Action::Convert => {
                    if let Err(e) = self.convert_now(device, queue) {
                        self.error = Some(format!("{e:?}"));
                        self.status = "Conversion failed".to_string();
                    }
                }
                Action::Export => {
                    if let Err(e) = self.export_now() {
                        self.error = Some(format!("{e:?}"));
                        self.status = "Export failed".to_string();
                    }
                }
            }
        }

        let full_output = self.egui_ctx.end_frame();
        egui_state.handle_platform_output(window, full_output.platform_output);

        let paint_jobs = self
            .egui_ctx
            .tessellate(full_output.shapes, full_output.pixels_per_point);

        let screen_desc = ScreenDescriptor {
            size_in_pixels: [config.width, config.height],
            pixels_per_point: window.scale_factor() as f32,
        };

        for (id, image_delta) in &full_output.textures_delta.set {
            egui_renderer.update_texture(device, queue, *id, image_delta);
        }

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("encoder"),
        });

        // Render pass (egui-wgpu wants an existing RenderPass) [web:175]
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("render-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.08,
                            g: 0.08,
                            b: 0.10,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            egui_renderer.update_buffers(device, queue, &mut encoder, &paint_jobs, &screen_desc);
            egui_renderer.render(&mut rpass, &paint_jobs, &screen_desc);
        }

        queue.submit(Some(encoder.finish()));
        frame.present();

        for id in &full_output.textures_delta.free {
            egui_renderer.free_texture(id);
        }

        Ok(())
    }

    fn convert_now(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) -> anyhow::Result<()> {
        let Some(input_path) = self.input_path.clone() else {
            anyhow::bail!("No input selected");
        };

        self.error = None;
        self.status = "Loading mesh…".to_string();

        // IMPORTANT: map this to your actual loader function name.
        // Your compile error says load_mesh_from_path doesn't exist.
        // Search in src/mesh_loader.rs for the correct function and replace this call.
        let mesh = mesh_loader::load_mesh(&input_path)
            .with_context(|| format!("Failed to load mesh: {}", input_path.display()))?;

        self.status = "Converting…".to_string();
        let start = Instant::now();

        let gaussians = converter::convert_mesh_to_gaussians(
            device,
            queue,
            &mesh,
            self.density,
            self.scale,
        )
        .map_err(|e| anyhow::anyhow!("{e}"))?; // avoid Box<dyn Error> Send/Sync issues

        self.last_ms = Some(start.elapsed().as_secs_f32() * 1000.0);
        self.last_count = Some(gaussians.len());
        self.gaussians = Some(gaussians);

        self.status = "Done.".to_string();
        Ok(())
    }

    fn export_now(&mut self) -> anyhow::Result<()> {
        let Some(gaussians) = self.gaussians.as_ref() else {
            anyhow::bail!("Nothing to export (convert first)");
        };

        let out_dir = self.output_dir.clone().unwrap_or_else(|| {
            std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
        });

        let stem = self
            .input_path
            .as_ref()
            .and_then(|p| p.file_stem())
            .and_then(|s| s.to_str())
            .unwrap_or("output");

        if self.export_ply {
            let ply_path = out_dir.join(format!("{stem}.ply"));
            export::ply::write_ply(gaussians, &ply_path)
                .with_context(|| format!("Failed to write {}", ply_path.display()))?;
            self.status = format!("Wrote {}", ply_path.display());
        }

        if self.export_splat {
            let splat_path = out_dir.join(format!("{stem}.splat"));
            export::splat::write_splat(gaussians, &splat_path)
                .with_context(|| format!("Failed to write {}", splat_path.display()))?;
            self.status = format!("Wrote {}", splat_path.display());
        }

        Ok(())
    }
}

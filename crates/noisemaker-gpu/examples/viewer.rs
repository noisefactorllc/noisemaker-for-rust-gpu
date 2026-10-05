//! A live viewer: a DSL program animated in real time in a window.
//!
//! ```text
//! cargo run --release -p noisemaker-for-rust-gpu --example viewer -- PROGRAM.dsl \
//!     [--size N] [--media IMAGE.png] [--exit-after SECONDS] [--screenshot OUT.png]
//! ```
//!
//! The program is loaded as the reference demo page loads it ([`DemoHost`])
//! and rendered every display refresh at the reference's normalized loop
//! time (`CanvasRenderer._renderLoop`: `(elapsed % loopDuration) /
//! loopDuration`). Each frame's render surface is presented to the window
//! the way the reference presents to its canvas ([`Presenter`]: the
//! `present()` blit, nearest-sampled when the window's drawable has the
//! render size and linearly filtered when it scales, onto a `bgra8unorm`
//! surface configured as the page configures its canvas). While the window
//! is hidden nothing renders, as `requestAnimationFrame` stops for a hidden
//! page; the loop time keeps running. Saving the DSL file recompiles it; a
//! program that does not compile is reported on standard error with the DSL
//! error formatter while the last good program keeps running. Escape or
//! closing the window quits.
//!
//! `--exit-after SECONDS` quits after that long, for scripted runs;
//! `--screenshot` then renders the frame of that moment and writes what the
//! window shows, read back through the same blit.

use std::path::PathBuf;
use std::process::ExitCode;
use std::rc::Rc;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

use noisemaker_gpu::demo::{DemoHost, DemoHostOptions};
use noisemaker_gpu::dsl::error_formatter::format_compile_error;
use noisemaker_gpu::host::{CanvasRenderer, CanvasRendererOptions};
use noisemaker_gpu::png_io::{read_png_rgba8, write_png_rgba8};
use noisemaker_gpu::{GpuDevice, Presenter, RenderError};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

const USAGE: &str = "usage: viewer PROGRAM.dsl [--size N] [--media IMAGE.png] [--exit-after SECONDS] [--screenshot OUT.png]";

/// How often the DSL file is checked for changes.
const WATCH_INTERVAL: Duration = Duration::from_millis(250);

struct Args {
    program: PathBuf,
    size: u32,
    media: Option<PathBuf>,
    exit_after: Option<Duration>,
    screenshot: Option<PathBuf>,
}

fn parse_args() -> Result<Args, String> {
    let mut program = None;
    let mut args = Args {
        program: PathBuf::new(),
        size: 512,
        media: None,
        exit_after: None,
        screenshot: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut value = |name: &str| it.next().ok_or_else(|| format!("{name} needs a value"));
        match arg.as_str() {
            "--size" => {
                args.size = value("--size")?
                    .parse()
                    .map_err(|e| format!("--size: {e}"))?
            }
            "--media" => args.media = Some(PathBuf::from(value("--media")?)),
            "--exit-after" => {
                let seconds: f64 = value("--exit-after")?
                    .parse()
                    .map_err(|e| format!("--exit-after: {e}"))?;
                args.exit_after = Some(
                    Duration::try_from_secs_f64(seconds)
                        .map_err(|e| format!("--exit-after: {e}"))?,
                );
            }
            "--screenshot" => args.screenshot = Some(PathBuf::from(value("--screenshot")?)),
            "-h" | "--help" => return Err(String::new()),
            _ if program.is_none() => program = Some(PathBuf::from(arg)),
            _ => return Err(format!("unexpected argument '{arg}'")),
        }
    }
    args.program = program.ok_or("no DSL file given")?;
    if args.screenshot.is_some() && args.exit_after.is_none() {
        return Err("--screenshot needs --exit-after".into());
    }
    Ok(args)
}

/// Print why a program did not load: compile errors with the DSL formatter.
fn report(source: &str, error: &RenderError) {
    match error.dsl_error() {
        Some(e) => eprintln!("{}", format_compile_error(source, e)),
        None => eprintln!("viewer: {error}"),
    }
}

/// The window, its surface and the program running in it.
struct Running {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    device: GpuDevice,
    host: DemoHost,
    presenter: Presenter,
    source: String,
    /// The program's last load failed.
    failed: bool,
    modified: Option<SystemTime>,
    last_check: Instant,
    started: Instant,
    /// The window can show frames (not occluded or minimized).
    visible: bool,
    /// Frames presented to the window.
    presented: u64,
}

impl Running {
    fn start(event_loop: &ActiveEventLoop, args: &Args) -> Result<Running, String> {
        let title = format!("noisemaker viewer: {}", args.program.display());
        let window = event_loop
            .create_window(
                Window::default_attributes()
                    .with_title(title)
                    .with_inner_size(LogicalSize::new(args.size, args.size)),
            )
            .map_err(|e| format!("window: {e}"))?;
        let window = Arc::new(window);
        let instance = wgpu::Instance::default();
        let surface = instance
            .create_surface(window.clone())
            .map_err(|e| format!("surface: {e}"))?;
        let device = GpuDevice::create_for_surface(instance, &surface, &Default::default())?;

        // context.configure({format: getPreferredCanvasFormat(), alphaMode:
        // 'premultiplied'}): bgra8unorm, never an sRGB-encoding format.
        let caps = surface.get_capabilities(&device.adapter);
        let format = [
            wgpu::TextureFormat::Bgra8Unorm,
            wgpu::TextureFormat::Rgba8Unorm,
        ]
        .into_iter()
        .find(|f| caps.formats.contains(f))
        .or_else(|| caps.formats.iter().copied().find(|f| !f.is_srgb()))
        .ok_or("the window surface has no linear 8-bit format")?;
        let alpha_mode = if caps
            .alpha_modes
            .contains(&wgpu::CompositeAlphaMode::PreMultiplied)
        {
            wgpu::CompositeAlphaMode::PreMultiplied
        } else {
            wgpu::CompositeAlphaMode::Auto
        };
        let size = window.inner_size();
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            color_space: wgpu::SurfaceColorSpace::Auto,
            width: size.width.max(1),
            height: size.height.max(1),
            desired_maximum_frame_latency: 2,
            // requestAnimationFrame: one frame per display refresh.
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode,
            view_formats: vec![],
        };
        surface.configure(&device.device, &config);

        let renderer = CanvasRenderer::new(
            &device,
            CanvasRendererOptions {
                width: args.size,
                height: args.size,
                ..Default::default()
            },
        );
        let default_media = match &args.media {
            Some(path) => Some(Rc::new(read_png_rgba8(path)?)),
            None => None,
        };
        let host = DemoHost::new(
            renderer,
            DemoHostOptions {
                default_media,
                ..Default::default()
            },
        );
        let presenter = Presenter::new(&device);
        let mut running = Running {
            window,
            surface,
            config,
            device,
            host,
            presenter,
            source: String::new(),
            failed: false,
            modified: None,
            last_check: Instant::now(),
            started: Instant::now(),
            visible: true,
            presented: 0,
        };
        running.reload(&args.program, true);
        Ok(running)
    }

    /// Load the DSL file when it changed (or `force`): rebuild the pipeline as
    /// the demo page does when its program changes.
    fn reload(&mut self, path: &std::path::Path, force: bool) {
        self.last_check = Instant::now();
        let modified = std::fs::metadata(path).and_then(|m| m.modified()).ok();
        if !force && modified == self.modified {
            return;
        }
        self.modified = modified;
        let source = match std::fs::read_to_string(path) {
            Ok(source) => source,
            Err(e) => {
                eprintln!("viewer: {}: {e}", path.display());
                self.failed = true;
                return;
            }
        };
        if !force && source == self.source {
            return;
        }
        self.source = source;
        let started = Instant::now();
        let result = self
            .host
            .rebuild_pipeline_from_dsl(&self.source, true)
            .and_then(|()| self.host.settle());
        match result {
            Ok(()) => {
                self.failed = false;
                eprintln!(
                    "viewer: loaded {} ({:.0} ms)",
                    path.display(),
                    started.elapsed().as_secs_f64() * 1000.0
                );
            }
            Err(e) => {
                self.failed = true;
                report(&self.source, &e);
            }
        }
    }

    fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device.device, &self.config);
        }
    }

    /// One iteration of the page's render loop, presented to the window. A
    /// window that cannot show a frame (occluded, minimized) renders nothing.
    fn frame(&mut self) {
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame)
            | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => frame,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(&self.device.device, &self.config);
                return;
            }
            wgpu::CurrentSurfaceTexture::Occluded => {
                self.visible = false;
                return;
            }
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Validation => {
                return;
            }
        };
        self.visible = true;
        self.render();
        let view = frame.texture.create_view(&Default::default());
        if !self.present_to(&view) {
            // No program yet: the canvas stays transparent black.
            clear(&self.device, &view);
        }
        self.window.pre_present_notify();
        self.device.queue.present(frame);
        self.presented += 1;
    }

    /// `_renderLoop` at this moment: a frame at the normalized loop time.
    fn render(&mut self) {
        if let Err(e) = self.host.renderer_mut().tick(Instant::now()) {
            eprintln!("viewer: render failed: {e}");
        }
    }

    /// `present(textureId)` of the last frame onto `view` (a texture of the
    /// surface's format and size); `false` when there is no frame.
    fn present_to(&mut self, view: &wgpu::TextureView) -> bool {
        let presented = self
            .host
            .renderer_mut()
            .pipeline()
            .and_then(|p| p.last_presented.clone());
        match (presented, self.host.renderer().backend()) {
            (Some(id), Some(backend)) => self.presenter.present(
                backend,
                &id,
                view,
                self.config.format,
                self.config.width,
                self.config.height,
            ),
            _ => false,
        }
    }

    /// What the window shows now: the frame of this moment, read back
    /// through the present blit at the window's size.
    fn screenshot(&mut self, path: &std::path::Path) -> Result<(), String> {
        self.render();
        let id = self
            .host
            .renderer_mut()
            .pipeline()
            .and_then(|p| p.last_presented.clone())
            .ok_or("no frame to capture")?;
        let backend = self.host.renderer().backend().ok_or("no pipeline")?;
        let pixels = self
            .presenter
            .capture(backend, &id, self.config.width, self.config.height)
            .map_err(|e| e.to_string())?
            .ok_or("no frame to capture")?;
        write_png_rgba8(path, pixels.width, pixels.height, &pixels.data)
    }
}

fn clear(device: &GpuDevice, view: &wgpu::TextureView) {
    let mut encoder = device
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("clear"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    device.queue.submit([encoder.finish()]);
}

struct Viewer {
    args: Args,
    running: Option<Running>,
    error: Option<String>,
}

impl Viewer {
    fn finish(&mut self, event_loop: &ActiveEventLoop) {
        if let Some(running) = &mut self.running {
            let seconds = running.started.elapsed().as_secs_f64();
            println!(
                "viewer: {} frames presented in {seconds:.1}s ({:.1} fps)",
                running.presented,
                running.presented as f64 / seconds
            );
            if running.presented == 0 {
                println!("viewer: the window was never visible (occluded or minimized)");
            }
            if let Some(path) = &self.args.screenshot {
                match running.screenshot(path) {
                    Ok(()) => println!("viewer: wrote {}", path.display()),
                    Err(e) => self.error = Some(format!("screenshot: {e}")),
                }
            }
            if running.failed && self.error.is_none() {
                self.error = Some(format!("{} did not load", self.args.program.display()));
            }
            let _ = running.host.renderer_mut().dispose();
        }
        event_loop.exit();
    }
}

impl ApplicationHandler for Viewer {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.running.is_some() {
            return;
        }
        match Running::start(event_loop, &self.args) {
            Ok(running) => {
                running.window.request_redraw();
                self.running = Some(running);
            }
            Err(e) => {
                self.error = Some(e);
                event_loop.exit();
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(running) = &mut self.running else {
            return;
        };
        match event {
            WindowEvent::CloseRequested
            | WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key: Key::Named(NamedKey::Escape),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => self.finish(event_loop),
            WindowEvent::Resized(size) => running.resize(size.width, size.height),
            WindowEvent::Occluded(occluded) => running.visible = !occluded,
            WindowEvent::RedrawRequested => running.frame(),
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        let Some(running) = &mut self.running else {
            return;
        };
        if running.last_check.elapsed() >= WATCH_INTERVAL {
            running.reload(&self.args.program, false);
        }
        if self
            .args
            .exit_after
            .is_some_and(|limit| running.started.elapsed() >= limit)
        {
            self.finish(event_loop);
            return;
        }
        // A visible window draws every display refresh (the surface's vsync
        // paces the loop); a hidden one checks back every WATCH_INTERVAL.
        running.window.request_redraw();
        if running.visible {
            event_loop.set_control_flow(ControlFlow::Poll);
        } else {
            event_loop.set_control_flow(ControlFlow::WaitUntil(Instant::now() + WATCH_INTERVAL));
        }
    }
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(args) => args,
        Err(e) => {
            if !e.is_empty() {
                eprintln!("viewer: {e}");
            }
            eprintln!("{USAGE}");
            return ExitCode::from(2);
        }
    };
    let event_loop = match EventLoop::new() {
        Ok(event_loop) => event_loop,
        Err(e) => {
            eprintln!("viewer: {e}");
            return ExitCode::from(1);
        }
    };
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut viewer = Viewer {
        args,
        running: None,
        error: None,
    };
    if let Err(e) = event_loop.run_app(&mut viewer) {
        eprintln!("viewer: {e}");
        return ExitCode::from(1);
    }
    match viewer.error {
        None => ExitCode::SUCCESS,
        Some(e) => {
            eprintln!("viewer: {e}");
            ExitCode::from(1)
        }
    }
}

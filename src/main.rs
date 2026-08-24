// Mesh2Splat macOS - Main Entry Point
// WebGPU-powered mesh to 3D Gaussian Splatting converter

mod app;
mod converter;
mod mesh_loader;
mod renderer;
mod export;

use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "mesh2splat")]
#[command(about = "Convert 3D meshes to Gaussian Splats using WebGPU", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Run GUI application
    Gui,
    
    /// Convert mesh to Gaussian splat (CLI mode)
    Convert {
        /// Input GLB/GLTF file
        #[arg(short, long)]
        input: PathBuf,
        
        /// Output file (.ply or .splat)
        #[arg(short, long)]
        output: PathBuf,
        
        /// Sampling density (0.1 - 2.0)
        #[arg(short, long, default_value = "1.0")]
        density: f32,
        
        /// Gaussian scale factor (0.1 - 2.0)
        #[arg(short, long, default_value = "0.65")]
        scale: f32,
        
        /// Verbose output
        #[arg(short, long)]
        verbose: bool,
    },
}

fn main() {
    env_logger::init();
    
    let cli = Cli::parse();
    
    match cli.command {
        Some(Commands::Convert { input, output, density, scale, verbose }) => {
            // CLI mode - batch conversion
            run_cli_conversion(input, output, density, scale, verbose);
        }
        Some(Commands::Gui) | None => {
            // GUI mode - default
            run_gui_app();
        }
    }
}

fn run_cli_conversion(
    input: PathBuf,
    output: PathBuf,
    density: f32,
    scale: f32,
    verbose: bool,
) {
    use std::time::Instant;
    
    println!("Mesh2Splat CLI - WebGPU Converter");
    println!("==================================");
    println!("Input:  {:?}", input);
    println!("Output: {:?}", output);
    println!("Density: {}, Scale: {}", density, scale);
    println!();
    
    // Load mesh
    print!("Loading mesh... ");
    let start = Instant::now();
    let mesh = match mesh_loader::load_gltf(&input) {
        Ok(m) => {
            println!("✓ ({:.2}ms)", start.elapsed().as_secs_f32() * 1000.0);
            m
        }
        Err(e) => {
            eprintln!("✗ Error: {}", e);
            std::process::exit(1);
        }
    };
    
    if verbose {
        println!("  Vertices: {}", mesh.vertices.len());
        println!("  Triangles: {}", mesh.indices.len() / 3);
    }
    
    // Initialize WebGPU
    print!("Initializing WebGPU... ");
    let start = Instant::now();
    let (device, queue) = match pollster::block_on(init_webgpu()) {
        Ok((d, q)) => {
            println!("✓ ({:.2}ms)", start.elapsed().as_secs_f32() * 1000.0);
            (d, q)
        }
        Err(e) => {
            eprintln!("✗ Error: {}", e);
            std::process::exit(1);
        }
    };
    
    // Convert to Gaussians
    print!("Converting to Gaussians... ");
    let start = Instant::now();
    let gaussians = match converter::convert_mesh_to_gaussians(
        &device,
        &queue,
        &mesh,
        density,
        scale,
    ) {
        Ok(g) => {
            let elapsed = start.elapsed().as_secs_f32() * 1000.0;
            println!("✓ ({:.2}ms)", elapsed);
            g
        }
        Err(e) => {
            eprintln!("✗ Error: {}", e);
            std::process::exit(1);
        }
    };
    
    if verbose {
        println!("  Gaussians generated: {}", gaussians.len());
    }
    
    // Export
    print!("Exporting... ");
    let start = Instant::now();
    let extension = output.extension().and_then(|e| e.to_str()).unwrap_or("");
    let result = match extension {
        "ply" => export::export_ply(&gaussians, &output),
        "splat" => export::export_splat(&gaussians, &output),
        _ => {
            eprintln!("✗ Unsupported format. Use .ply or .splat");
            std::process::exit(1);
        }
    };
    
    match result {
        Ok(_) => {
            println!("✓ ({:.2}ms)", start.elapsed().as_secs_f32() * 1000.0);
            println!("\n✓ Conversion complete!");
        }
        Err(e) => {
            eprintln!("✗ Error: {}", e);
            std::process::exit(1);
        }
    }
}

fn run_gui_app() {
    println!("Starting Mesh2Splat GUI...");
    app::run_gui()?;

}

async fn init_webgpu() -> Result<(wgpu::Device, wgpu::Queue), Box<dyn std::error::Error>> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::METAL, // macOS uses Metal backend
        ..Default::default()
    });
    
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        })
        .await
        .ok_or("Failed to find GPU adapter")?;
    
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Mesh2Splat Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        )
        .await?;
    
    Ok((device, queue))
}

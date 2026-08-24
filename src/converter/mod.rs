// Converter Module - Mesh to Gaussian conversion using WebGPU compute shaders

pub mod compute;
pub mod gaussian;

use crate::mesh_loader::Mesh;
pub use gaussian::Gaussian;

pub fn convert_mesh_to_gaussians(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    mesh: &Mesh,
    density: f32,
    scale: f32,
) -> Result<Vec<Gaussian>, Box<dyn std::error::Error>> {
    compute::run_conversion(device, queue, mesh, density, scale)
}

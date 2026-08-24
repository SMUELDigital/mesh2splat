// WebGPU Compute Shader - Mesh2Splat Conversion Algorithm
// This implements the core algorithm from the EA SEED paper

use crate::mesh_loader::{Mesh, Vertex};
use crate::converter::Gaussian;
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

// WGSL Compute Shader for Mesh2Splat
const MESH2SPLAT_SHADER: &str = r#"
struct VertexInput {
    position: vec3f,
    _pad1: f32,
    normal: vec3f,
    _pad2: f32,
    uv: vec2f,
    _pad3: vec2f,
}

struct Gaussian {
    position: vec3f,
    _pad1: f32,
    normal: vec3f,
    _pad2: f32,
    scale: vec3f,
    _pad3: f32,
    rotation: vec4f,
    color: vec4f,
    opacity: f32,
    _pad4: vec3f,
}

struct Params {
    density: f32,
    gaussian_scale: f32,
    triangle_count: u32,
    _pad: f32,
}

@group(0) @binding(0) var<storage, read> vertices: array<VertexInput>;
@group(0) @binding(1) var<storage, read> indices: array<u32>;
@group(0) @binding(2) var<storage, read_write> gaussians: array<Gaussian>;
@group(0) @binding(3) var<uniform> params: Params;

// Compute rotation quaternion from normal vector
fn rotation_from_normal(normal: vec3f) -> vec4f {
    let up = vec3f(0.0, 1.0, 0.0);
    let axis = cross(up, normal);
    let axis_length = length(axis);
    
    // Handle parallel case
    if (axis_length < 0.001) {
        if (dot(up, normal) > 0.0) {
            return vec4f(0.0, 0.0, 0.0, 1.0); // No rotation
        } else {
            return vec4f(1.0, 0.0, 0.0, 0.0); // 180° rotation
        }
    }
    
    let angle = acos(clamp(dot(up, normal), -1.0, 1.0));
    let half_angle = angle * 0.5;
    let s = sin(half_angle);
    let c = cos(half_angle);
    let norm_axis = axis / axis_length;
    
    return vec4f(norm_axis.x * s, norm_axis.y * s, norm_axis.z * s, c);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3u) {
    let tri_idx = global_id.x;
    
    if (tri_idx >= params.triangle_count) {
        return;
    }
    
    // Get triangle vertices
    let base_idx = tri_idx * 3u;
    let idx0 = indices[base_idx];
    let idx1 = indices[base_idx + 1u];
    let idx2 = indices[base_idx + 2u];
    
    let v0 = vertices[idx0];
    let v1 = vertices[idx1];
    let v2 = vertices[idx2];
    
    // Compute triangle properties
    let p0 = v0.position;
    let p1 = v1.position;
    let p2 = v2.position;
    
    let edge1 = p1 - p0;
    let edge2 = p2 - p0;
    let face_normal = normalize(cross(edge1, edge2));
    
    // Compute triangle area and average edge length
    let area = length(cross(edge1, edge2)) * 0.5;
    let edge3 = p2 - p1;
    let avg_edge_length = (length(edge1) + length(edge2) + length(edge3)) / 3.0;
    
    // Scale based on triangle size, density, and user scale
    let base_scale = avg_edge_length * params.gaussian_scale * params.density;
    
    // Generate 3 Gaussians per triangle (one per vertex)
    // This exploits GPU interpolation for smooth coverage
    for (var i = 0u; i < 3u; i++) {
        let gauss_idx = tri_idx * 3u + i;
        var gauss: Gaussian;
        
        // Position at vertex
        if (i == 0u) {
            gauss.position = p0;
        } else if (i == 1u) {
            gauss.position = p1;
        } else {
            gauss.position = p2;
        }
        
        // Normal aligned with face
        gauss.normal = face_normal;
        
        // Anisotropic scale - wide along surface, thin perpendicular
        // This creates the "splat" effect
        gauss.scale = vec3f(base_scale, base_scale, base_scale * 0.01);
        
        // Rotation aligned with surface normal
        gauss.rotation = rotation_from_normal(face_normal);
        
        // Default white color (texture sampling would go here)
        gauss.color = vec4f(0.8, 0.8, 0.8, 1.0);
        gauss.opacity = 1.0;
        
        gaussians[gauss_idx] = gauss;
    }
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuVertex {
    position: [f32; 3],
    _pad1: f32,
    normal: [f32; 3],
    _pad2: f32,
    uv: [f32; 2],
    _pad3: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    density: f32,
    gaussian_scale: f32,
    triangle_count: u32,
    _pad: f32,
}

pub fn run_conversion(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    mesh: &Mesh,
    density: f32,
    scale: f32,
) -> Result<Vec<Gaussian>, Box<dyn std::error::Error>> {
    let triangle_count = mesh.indices.len() / 3;
    let gaussian_count = triangle_count * 3; // 3 Gaussians per triangle
    
    // Convert vertices to GPU format
    let gpu_vertices: Vec<GpuVertex> = mesh.vertices.iter().map(|v| GpuVertex {
        position: v.position.to_array(),
        _pad1: 0.0,
        normal: v.normal.to_array(),
        _pad2: 0.0,
        uv: v.uv.to_array(),
        _pad3: [0.0; 2],
    }).collect();
    
    // Create shader module
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Mesh2Splat Compute Shader"),
        source: wgpu::ShaderSource::Wgsl(MESH2SPLAT_SHADER.into()),
    });
    
    // Create buffers
    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Vertex Buffer"),
        contents: bytemuck::cast_slice(&gpu_vertices),
        usage: wgpu::BufferUsages::STORAGE,
    });
    
    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Index Buffer"),
        contents: bytemuck::cast_slice(&mesh.indices),
        usage: wgpu::BufferUsages::STORAGE,
    });
    
    let gaussian_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Gaussian Buffer"),
        size: (gaussian_count * std::mem::size_of::<Gaussian>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    
    let params = Params {
        density,
        gaussian_scale: scale,
        triangle_count: triangle_count as u32,
        _pad: 0.0,
    };
    
    let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Params Buffer"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    
    // Create compute pipeline
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Mesh2Splat Bind Group Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Mesh2Splat Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: vertex_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: index_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: gaussian_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });
    
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Mesh2Splat Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });
    
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Mesh2Splat Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "main",
        compilation_options: Default::default(),
        cache: None,
    });
    
    // Execute compute shader
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Mesh2Splat Encoder"),
    });
    
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Mesh2Splat Pass"),
            timestamp_writes: None,
        });
        
        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        
        // Dispatch workgroups (64 threads per workgroup)
        let workgroup_count = (triangle_count as u32 + 63) / 64;
        compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
    }
    
    // Read back results
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Buffer"),
        size: (gaussian_count * std::mem::size_of::<Gaussian>()) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    
    encoder.copy_buffer_to_buffer(
        &gaussian_buffer,
        0,
        &staging_buffer,
        0,
        (gaussian_count * std::mem::size_of::<Gaussian>()) as u64,
    );
    
    queue.submit(Some(encoder.finish()));
    
    // Map buffer and read data
    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = futures::channel::oneshot::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).unwrap();
    });
    
    device.poll(wgpu::Maintain::Wait);
    pollster::block_on(receiver)?;
    
    let data = buffer_slice.get_mapped_range();
    let gaussians: Vec<Gaussian> = bytemuck::cast_slice(&data).to_vec();
    
    drop(data);
    staging_buffer.unmap();
    
    Ok(gaussians)
}

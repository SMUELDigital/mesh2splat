// Gaussian data structure

use glam::{Vec3, Vec4};
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Gaussian {
    pub position: [f32; 3],
    pub _padding1: f32,
    pub normal: [f32; 3],
    pub _padding2: f32,
    pub scale: [f32; 3],
    pub _padding3: f32,
    pub rotation: [f32; 4], // quaternion
    pub color: [f32; 4],
    pub opacity: f32,
    pub _padding4: [f32; 3],
}

impl Gaussian {
    pub fn new(position: Vec3, normal: Vec3, scale: Vec3, rotation: Vec4, color: Vec4, opacity: f32) -> Self {
        Self {
            position: position.to_array(),
            _padding1: 0.0,
            normal: normal.to_array(),
            _padding2: 0.0,
            scale: scale.to_array(),
            _padding3: 0.0,
            rotation: rotation.to_array(),
            color: color.to_array(),
            opacity,
            _padding4: [0.0; 3],
        }
    }
    
    pub fn default() -> Self {
        Self {
            position: [0.0; 3],
            _padding1: 0.0,
            normal: [0.0, 1.0, 0.0],
            _padding2: 0.0,
            scale: [1.0; 3],
            _padding3: 0.0,
            rotation: [0.0, 0.0, 0.0, 1.0],
            color: [1.0; 4],
            opacity: 1.0,
            _padding4: [0.0; 3],
        }
    }
}

impl Default for Gaussian {
    fn default() -> Self {
        Self::default()
    }
}


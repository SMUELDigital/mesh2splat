// SPLAT Export - Compact binary format for web viewers

use crate::converter::Gaussian;
use std::fs::File;
use std::io::{self, Write};
use std::path::Path;

/// Export Gaussians to .splat format (compact binary)
/// Format optimized for web-based Gaussian splatting viewers
pub fn write_splat(gaussians: &[Gaussian], path: &Path) -> io::Result<()> {
    let mut file = File::create(path)?;
    
    // SPLAT format: packed binary with no header
    // Each Gaussian: position(3xf32) + scale(3xf32) + color(4xu8) + rotation(4xf32)
    // Total: 48 bytes per Gaussian
    
    for gauss in gaussians {
        // Position (12 bytes)
        file.write_all(&gauss.position[0].to_le_bytes())?;
        file.write_all(&gauss.position[1].to_le_bytes())?;
        file.write_all(&gauss.position[2].to_le_bytes())?;
        
        // Scale (12 bytes)
        file.write_all(&gauss.scale[0].to_le_bytes())?;
        file.write_all(&gauss.scale[1].to_le_bytes())?;
        file.write_all(&gauss.scale[2].to_le_bytes())?;
        
        // Color as RGBA bytes (4 bytes) - more compact than floats
        let r = (gauss.color[0] * 255.0).clamp(0.0, 255.0) as u8;
        let g = (gauss.color[1] * 255.0).clamp(0.0, 255.0) as u8;
        let b = (gauss.color[2] * 255.0).clamp(0.0, 255.0) as u8;
        let a = (gauss.opacity * 255.0).clamp(0.0, 255.0) as u8;
        
        file.write_all(&[r, g, b, a])?;
        
        // Rotation quaternion (16 bytes)
        file.write_all(&gauss.rotation[0].to_le_bytes())?;
        file.write_all(&gauss.rotation[1].to_le_bytes())?;
        file.write_all(&gauss.rotation[2].to_le_bytes())?;
        file.write_all(&gauss.rotation[3].to_le_bytes())?;
    }
    
    Ok(())
}

/// Calculate file size for .splat export
pub fn splat_file_size(gaussian_count: usize) -> usize {
    gaussian_count * 48 // 48 bytes per Gaussian
}

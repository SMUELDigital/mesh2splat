// PLY Export - Standard 3D Gaussian Splatting format

use crate::converter::Gaussian;
use std::fs::File;
use std::io::{self, Write};
use std::path::Path;

pub fn write_ply(gaussians: &[Gaussian], path: &Path) -> io::Result<()> {
    let mut file = File::create(path)?;
    
    // Write PLY header
    writeln!(file, "ply")?;
    writeln!(file, "format binary_little_endian 1.0")?;
    writeln!(file, "element vertex {}", gaussians.len())?;
    writeln!(file, "property float x")?;
    writeln!(file, "property float y")?;
    writeln!(file, "property float z")?;
    writeln!(file, "property float nx")?;
    writeln!(file, "property float ny")?;
    writeln!(file, "property float nz")?;
    writeln!(file, "property float f_dc_0")?;
    writeln!(file, "property float f_dc_1")?;
    writeln!(file, "property float f_dc_2")?;
    writeln!(file, "property float opacity")?;
    writeln!(file, "property float scale_0")?;
    writeln!(file, "property float scale_1")?;
    writeln!(file, "property float scale_2")?;
    writeln!(file, "property float rot_0")?;
    writeln!(file, "property float rot_1")?;
    writeln!(file, "property float rot_2")?;
    writeln!(file, "property float rot_3")?;
    writeln!(file, "end_header")?;
    
    // Write binary data
    for gauss in gaussians {
        // Position
        file.write_all(&gauss.position[0].to_le_bytes())?;
        file.write_all(&gauss.position[1].to_le_bytes())?;
        file.write_all(&gauss.position[2].to_le_bytes())?;
        
        // Normal
        file.write_all(&gauss.normal[0].to_le_bytes())?;
        file.write_all(&gauss.normal[1].to_le_bytes())?;
        file.write_all(&gauss.normal[2].to_le_bytes())?;
        
        // Color (RGB as spherical harmonics DC component)
        file.write_all(&gauss.color[0].to_le_bytes())?;
        file.write_all(&gauss.color[1].to_le_bytes())?;
        file.write_all(&gauss.color[2].to_le_bytes())?;
        
        // Opacity
        file.write_all(&gauss.opacity.to_le_bytes())?;
        
        // Scale
        file.write_all(&gauss.scale[0].to_le_bytes())?;
        file.write_all(&gauss.scale[1].to_le_bytes())?;
        file.write_all(&gauss.scale[2].to_le_bytes())?;
        
        // Rotation (quaternion)
        file.write_all(&gauss.rotation[0].to_le_bytes())?;
        file.write_all(&gauss.rotation[1].to_le_bytes())?;
        file.write_all(&gauss.rotation[2].to_le_bytes())?;
        file.write_all(&gauss.rotation[3].to_le_bytes())?;
    }
    
    Ok(())
}

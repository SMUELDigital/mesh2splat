// Export Module - PLY and SPLAT format writers

pub mod ply;
pub mod splat;

use std::path::Path;
use crate::converter::Gaussian;

pub fn export_ply(gaussians: &[Gaussian], path: &Path) -> std::io::Result<()> {
    ply::write_ply(gaussians, path)
}

pub fn export_splat(gaussians: &[Gaussian], path: &Path) -> std::io::Result<()> {
    splat::write_splat(gaussians, path)
}

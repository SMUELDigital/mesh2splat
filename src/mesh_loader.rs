use std::path::Path;

use anyhow::{Context, Result};
use glam::{Vec2, Vec3};

#[derive(Clone, Copy, Debug, Default)]
pub struct Vertex {
    pub position: Vec3,
    pub normal: Vec3,
    pub uv: Vec2,
}

#[derive(Clone, Debug, Default)]
pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

pub fn load_mesh(path: impl AsRef<Path>) -> Result<Mesh> {
    let path = path.as_ref();

    let (doc, buffers, _images) =
        gltf::import(path).with_context(|| format!("gltf::import failed: {}", path.display()))?;

    let mut out = Mesh::default();

    // Flatten all primitives of all meshes into one Mesh.
    for mesh in doc.meshes() {
        for prim in mesh.primitives() {
            let reader = prim.reader(|buffer| Some(&buffers[buffer.index()]));

            let positions: Vec<[f32; 3]> = reader
                .read_positions()
                .context("Primitive has no POSITION attribute")?
                .collect();

            let normals: Option<Vec<[f32; 3]>> = reader.read_normals().map(|it| it.collect());

            let uvs: Option<Vec<[f32; 2]>> = reader
                .read_tex_coords(0)
                .map(|it| it.into_f32().collect());

            let indices: Vec<u32> = reader
                .read_indices()
                .context("Primitive has no indices")?
                .into_u32()
                .collect();

            // Current base vertex offset into the merged vertex list
            let base = out.vertices.len() as u32;

            // Build vertices
            out.vertices.reserve(positions.len());
            for i in 0..positions.len() {
                let p = positions[i];
                let n = normals
                    .as_ref()
                    .and_then(|ns| ns.get(i).copied())
                    .unwrap_or([0.0, 1.0, 0.0]);
                let uv = uvs
                    .as_ref()
                    .and_then(|uvs| uvs.get(i).copied())
                    .unwrap_or([0.0, 0.0]);

                out.vertices.push(Vertex {
                    position: Vec3::new(p[0], p[1], p[2]),
                    normal: Vec3::new(n[0], n[1], n[2]),
                    uv: Vec2::new(uv[0], uv[1]),
                });
            }

            // Append indices with base offset
            out.indices.reserve(indices.len());
            out.indices.extend(indices.into_iter().map(|idx| idx + base));
        }
    }

    anyhow::ensure!(
        !out.vertices.is_empty(),
        "No vertices found in {}",
        path.display()
    );
    anyhow::ensure!(
        !out.indices.is_empty(),
        "No indices found in {}",
        path.display()
    );

    Ok(out)
}

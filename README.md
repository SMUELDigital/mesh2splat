# Mesh2Splat

**WebGPU-powered mesh to 3D Gaussian Splatting converter for Windows and macOS.**

Converts 3D meshes (`.glb` / `.gltf`) into 3D Gaussian Splatting (3DGS) models in milliseconds, using WebGPU compute shaders (Metal on macOS, DirectX 12/Vulkan on Windows) via [`wgpu`](https://wgpu.rs/).

[![CI](https://github.com/SMUELDigital/mesh2splat/actions/workflows/ci.yml/badge.svg)](https://github.com/SMUELDigital/mesh2splat/actions/workflows/ci.yml)
[![Release](https://github.com/SMUELDigital/mesh2splat/actions/workflows/release.yml/badge.svg)](https://github.com/SMUELDigital/mesh2splat/actions/workflows/release.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Features

- **Cross-platform** — a single Rust codebase builds native apps for both macOS (Apple Silicon & Intel) and Windows (x86_64)
- **Blazing fast** — converts meshes to Gaussians in well under a millisecond for typical assets
- **WebGPU native** — runs on Metal (macOS) or DirectX 12 / Vulkan (Windows) via `wgpu`
- **Dual mode** — GUI application (`egui`) and CLI for pipeline/batch integration
- **Format support** — export to `.ply` and `.splat`
- **Houdini/Blender ready** — CLI designed for scripting from DCC tools

## Download

Pre-built binaries for Windows and macOS are published on the [Releases page](https://github.com/SMUELDigital/mesh2splat/releases) whenever a new version tag is pushed. Each release contains:

- `mesh2splat-x86_64-pc-windows-msvc.zip` — Windows 10/11 (x86_64)
- `mesh2splat-aarch64-apple-darwin.tar.gz` — macOS, Apple Silicon
- `mesh2splat-x86_64-apple-darwin.tar.gz` — macOS, Intel

## Building from source

### Prerequisites (both platforms)

- Rust 1.75+ via [rustup](https://rustup.rs)

### macOS

```bash
xcode-select --install   # if not already installed
git clone https://github.com/SMUELDigital/mesh2splat.git
cd mesh2splat
cargo build --release
# Binary: target/release/mesh2splat
```

### Windows

Install the [MSVC Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) (or Visual Studio with the "Desktop development with C++" workload), then:

```powershell
git clone https://github.com/SMUELDigital/mesh2splat.git
cd mesh2splat
cargo build --release
# Binary: target\release\mesh2splat.exe
```

## Usage

### GUI mode (default)

```bash
./mesh2splat
```

- Drag & drop GLB/GLTF files
- Adjustable sampling density and Gaussian scale sliders
- Live stats (triangle count, Gaussian count, conversion time)
- Export to `.ply` or `.splat`

### CLI mode (for pipeline integration)

```bash
mesh2splat convert -i model.glb -o output.ply

# with custom parameters
mesh2splat convert -i helmet.glb -o helmet_splat.ply --density 1.5 --scale 0.8 --verbose

# batch processing (bash)
for file in models/*.glb; do
  mesh2splat convert -i "$file" -o "splats/$(basename "$file" .glb).ply"
done
```

**CLI arguments:**

| Flag | Description | Default |
|------|-------------|---------|
| `-i, --input <FILE>` | Input GLB/GLTF file (required) | — |
| `-o, --output <FILE>` | Output `.ply` or `.splat` file (required) | — |
| `-d, --density <FLOAT>` | Sampling density (0.1 – 2.0) | 1.0 |
| `-s, --scale <FLOAT>` | Gaussian scale factor (0.1 – 2.0) | 0.65 |
| `-v, --verbose` | Show detailed conversion statistics | off |

### Houdini integration

```python
import subprocess

subprocess.run([
    "mesh2splat", "convert",
    "-i", "input.glb",
    "-o", "output.ply",
    "--density", str(node.parm("density").eval()),
    "--scale", str(node.parm("scale").eval()),
])
```

## How it works

1. **Mesh analysis** — parses GLB/GLTF and extracts vertices, normals, UVs
2. **WebGPU compute** — runs a WGSL compute shader on the GPU:
   - Processes each triangle in parallel (64 threads/workgroup)
   - Computes triangle area and average edge length
   - Generates 3 Gaussians per triangle (one per vertex)
   - Calculates anisotropic scale (wide on the surface, thin perpendicular to it)
   - Aligns the rotation quaternion with the surface normal
3. **GPU readback** — retrieves Gaussian data from GPU memory
4. **Export** — writes binary `.ply` or `.splat`

## File formats

**Input:** binary GLB (recommended) or text GLTF + `.bin`, with PBR materials and embedded/external textures.

**Output:**
- `.ply` — standard binary PLY with position, normal, spherical-harmonic color, opacity, anisotropic scale, and rotation quaternion per Gaussian
- `.splat` — compact packed binary format optimized for WebGL/WebGPU viewers

## Project structure

```
mesh2splat/
├── src/
│   ├── main.rs           # CLI + GUI entry point
│   ├── app.rs             # GUI application (egui)
│   ├── mesh_loader.rs      # GLTF parser
│   ├── converter/
│   │   ├── mod.rs         # Converter interface
│   │   ├── compute.rs     # WebGPU compute pipeline
│   │   └── gaussian.rs    # Gaussian data structure
│   ├── renderer/           # 3DGS preview renderer
│   └── export/             # PLY / SPLAT writers
├── docs/web-demo.html      # Standalone browser (WebGPU) demo
├── .github/workflows/       # CI + cross-platform release automation
├── Cargo.toml
└── README.md
```

## Continuous integration & releases

- `.github/workflows/ci.yml` builds and tests on every push/PR for both macOS and Windows runners.
- `.github/workflows/release.yml` builds Windows (x86_64) and macOS (Apple Silicon + Intel) binaries and publishes them to a GitHub Release whenever a tag matching `v*.*.*` is pushed:

```bash
git tag v1.0.0
git push origin v1.0.0
```

## Known limitations

- Volumetric data (foliage, hair, clouds) is not optimized (triangles only)
- Texture sampling is not yet implemented (uniform colors for now)

## Roadmap

- [ ] Texture map support (diffuse, normal, roughness)
- [ ] Multiple material handling
- [ ] LOD (level of detail) generation
- [ ] Real-time preview renderer with PBR shading
- [ ] Batch conversion queue
- [ ] Linux builds

## Credits

**Original algorithm:** Scolari, Stefano (2024). *Mesh2Splat: Gaussian Splatting from 3D Geometry and Materials*. Master's Thesis, KTH Royal Institute of Technology, in collaboration with Electronic Arts SEED. See the [original EA Mesh2Splat repository](https://github.com/electronicarts/mesh2splat).

**This cross-platform Rust/WebGPU port** targets macOS and Windows using [`wgpu`](https://wgpu.rs/), [`egui`](https://github.com/emilk/egui), and the wider Rust ecosystem.

## License

[MIT](LICENSE)

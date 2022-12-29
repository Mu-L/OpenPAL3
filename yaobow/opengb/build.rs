use std::process::Command;

fn main() {
    build_shader("lightmap_texture.vert");
    build_shader("lightmap_texture.frag");
}

fn build_shader(shader_name: &str) {
    let out_dir = std::env::var("OUT_DIR").unwrap();
    println!("{}", out_dir);
    let path = format!("src/shaders/{}", shader_name);
    println!("cargo:rerun-if-changed={}", path);
    let shader_out_dir = format!("{}/{}.spv", out_dir, shader_name);
    let output = Command::new("glslc")
        .args(&[path, "-o".to_string(), shader_out_dir])
        .output()
        .expect(&format!("Failed to compile shader {}", shader_name));

    println!("{}", std::str::from_utf8(&output.stdout).unwrap());
}

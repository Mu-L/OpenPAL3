mod engine;
mod factory;
mod material;
mod platform;
mod render_object;
mod rendering_component;
mod shader;
mod sprite;
mod texture;
mod vertex_buffer;
mod video_player;

pub use engine::RenderingEngine;
pub use factory::ComponentFactory;
pub use material::{Material, MaterialDef, SimpleMaterialDef};
pub use platform::Window;
pub use render_object::RenderObject;
pub use rendering_component::RenderingComponent;
pub use shader::{Shader, ShaderDef, SIMPLE_SHADER_DEF};
pub use sprite::Sprite;
pub use texture::{Texture, TextureDef, TextureStore};
pub use vertex_buffer::{VertexBuffer, VertexComponents};
pub use video_player::VideoPlayer;

#[cfg(
    target_os = "windows",
    target_os = "linux",
    target_os = "macos",
    target_os = "android"
)]
mod vulkan;

#[cfg(
    target_os = "windows",
    target_os = "linux",
    target_os = "macos",
    target_os = "android"
)]
pub use vulkan::VulkanRenderingEngine;

#[cfg(target_os = "psp")]
mod gu;

#[cfg(target_os = "psp")]
pub use gu::GuRenderingEngine;

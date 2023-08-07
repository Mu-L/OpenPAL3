mod core_engine;
mod debugging;
mod task_manager;
mod ui_manager;

pub use core_engine::CoreRadianceEngine;
pub use debugging::DebugLayer;
pub use task_manager::{TaskHandle, TaskManager};
pub use ui_manager::UiManager;

use crosscom::ComRc;

use crate::{
    application::Platform, audio::OpenAlAudioEngine, input::GenericInputEngine,
    rendering::VulkanRenderingEngine, scene::DefaultSceneManager,
};
use std::{cell::RefCell, error::Error, rc::Rc};

pub fn create_radiance_engine(
    platform: &mut Platform,
) -> Result<CoreRadianceEngine, Box<dyn Error>> {
    let ui_manager = Rc::new(UiManager::new(platform));

    #[cfg(target_os = "windows")]
    let window = &crate::rendering::Window {
        hwnd: platform.hwnd(),
    };

    #[cfg(not(target_os = "windows"))]
    let window = platform.get_window();

    let rendering_engine = Rc::new(RefCell::new(VulkanRenderingEngine::new(
        window,
        &ui_manager.imgui_context(),
    )?));

    #[cfg(target_os = "android")]
    {
        use winit::event::Event;
        let rendering_engine_clone = rendering_engine.clone();
        let w = window.clone();
        platform.add_message_callback(Box::new(move |event| {
            let mut rendering_engine = rendering_engine_clone.borrow_mut();
            match event {
                Event::Suspended => {
                    rendering_engine.drop_surface();
                }
                Event::Resumed => {
                    rendering_engine.recreate_surface(&w).unwrap();
                }
                _ => (),
            }
        }));
    }
    let audio_engine = Rc::new(OpenAlAudioEngine::new());
    let input_engine = GenericInputEngine::new(platform);
    let scene_manager = ComRc::from_object(DefaultSceneManager::new());

    Ok(CoreRadianceEngine::new(
        rendering_engine,
        audio_engine,
        input_engine,
        ui_manager,
        scene_manager,
    ))
}

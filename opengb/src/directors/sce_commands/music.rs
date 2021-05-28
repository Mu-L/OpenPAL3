use crate::directors::sce_vm::{SceCommand, SceState};

use imgui::Ui;
use radiance::scene::SceneManager;

#[derive(Clone)]
pub struct SceCommandMusic {
    name: String,
}

impl SceCommand for SceCommandMusic {
    fn update(
        &mut self,
        scene_manager: &mut dyn SceneManager,
        ui: &mut Ui,
        state: &mut SceState,
        delta_sec: f32,
    ) -> bool {
        state.global_state_mut().play_bgm(&self.name);
        true
    }
}

impl SceCommandMusic {
    pub fn new(name: String, unknown: i32) -> Self {
        Self { name }
    }
}

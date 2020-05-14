use super::RolePropertyNames;
use super::{RoleProperties, SceneMv3Extensions};
use crate::asset_manager::AssetManager;
use crate::director::sce_director::SceCommand;
use crate::director::sce_state::SceState;
use crate::scene::ScnScene;
use imgui::Ui;
use radiance::math::Vec3;
use radiance::scene::{CoreScene, Entity};
use std::rc::Rc;

#[derive(Clone)]
pub struct SceCommandRoleFaceRole {
    role_id: String,
    role_id2: String,
}

impl SceCommand for SceCommandRoleFaceRole {
    fn update(
        &mut self,
        scene: &mut CoreScene<ScnScene>,
        ui: &mut Ui,
        state: &mut SceState,
        delta_sec: f32,
    ) -> bool {
        let position = RoleProperties::position(state, &self.role_id2);
        RoleProperties::set_face_to(state, &self.role_id, &position);

        let entity = scene.get_mv3_entity(&RolePropertyNames::name(&self.role_id));
        entity.transform_mut().look_at(&position);
        return true;
    }
}

impl SceCommandRoleFaceRole {
    pub fn new(role_id: i32, role_id2: i32) -> Self {
        Self {
            role_id: format!("{}", role_id),
            role_id2: format!("{}", role_id2),
        }
    }
}

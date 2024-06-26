use crosscom::ComRc;

use crate::{
    comdef::{IEntity, IStaticMeshComponent},
    components::mesh::{Geometry, StaticMeshComponent, TexCoord},
    math::Vec3,
    rendering::{ComponentFactory, SimpleMaterialDef},
    scene::CoreEntity,
};

pub fn create_box_entity(component_factory: std::rc::Rc<dyn ComponentFactory>) -> ComRc<IEntity> {
    const HALF_SIZE: f32 = 3.;
    let vertices = vec![
        Vec3::new(-HALF_SIZE, -HALF_SIZE, -HALF_SIZE),
        Vec3::new(-HALF_SIZE, -HALF_SIZE, HALF_SIZE),
        Vec3::new(HALF_SIZE, -HALF_SIZE, HALF_SIZE),
        Vec3::new(HALF_SIZE, -HALF_SIZE, -HALF_SIZE),
        Vec3::new(-HALF_SIZE, HALF_SIZE, -HALF_SIZE),
        Vec3::new(-HALF_SIZE, HALF_SIZE, HALF_SIZE),
        Vec3::new(HALF_SIZE, HALF_SIZE, HALF_SIZE),
        Vec3::new(HALF_SIZE, HALF_SIZE, -HALF_SIZE),
    ];

    let texcoords = vec![vec![
        TexCoord::new(0., 0.),
        TexCoord::new(0., 1.),
        TexCoord::new(1., 0.),
        TexCoord::new(1., 1.),
        TexCoord::new(0., 0.),
        TexCoord::new(0., 1.),
        TexCoord::new(1., 0.),
        TexCoord::new(1., 1.),
    ]];

    let indices = vec![
        0, 2, 1, 0, 3, 2, 0, 4, 7, 0, 7, 3, 0, 5, 4, 0, 1, 5, 6, 1, 2, 6, 5, 1, 6, 2, 3, 6, 3, 7,
        6, 7, 4, 6, 4, 5,
    ];

    let material = SimpleMaterialDef::create2("box", None, false);
    let geometry = Geometry::new(&vertices, None, &texcoords, indices, material);

    let entity = ComRc::<IEntity>::from_object(CoreEntity::new("box".to_string(), true));
    let mesh_component =
        StaticMeshComponent::new(entity.clone(), vec![geometry], component_factory);
    entity.add_component(
        IStaticMeshComponent::uuid(),
        ComRc::from_object(mesh_component),
    );

    entity
}

pub fn create_box_entity2(
    component_factory: std::rc::Rc<dyn ComponentFactory>,
    vertices: Vec<Vec3>,
) -> ComRc<IEntity> {
    let texcoords = vec![vec![
        TexCoord::new(0., 0.),
        TexCoord::new(0., 1.),
        TexCoord::new(1., 0.),
        TexCoord::new(1., 1.),
        TexCoord::new(0., 0.),
        TexCoord::new(0., 1.),
        TexCoord::new(1., 0.),
        TexCoord::new(1., 1.),
    ]];

    let indices = vec![
        0, 2, 1, 0, 3, 2, 0, 4, 7, 0, 7, 3, 0, 5, 4, 0, 1, 5, 6, 1, 2, 6, 5, 1, 6, 2, 3, 6, 3, 7,
        6, 7, 4, 6, 4, 5,
    ];

    let material = SimpleMaterialDef::create2("box", None, false);
    let geometry = Geometry::new(&vertices, None, &texcoords, indices, material);

    let entity = ComRc::<IEntity>::from_object(CoreEntity::new("box".to_string(), true));
    let mesh_component =
        StaticMeshComponent::new(entity.clone(), vec![geometry], component_factory);
    entity.add_component(
        IStaticMeshComponent::uuid(),
        ComRc::from_object(mesh_component),
    );

    entity
}

pub fn create_triangle_entity(
    component_factory: std::rc::Rc<dyn ComponentFactory>,
    v1: Vec3,
    v2: Vec3,
    v3: Vec3,
) -> ComRc<IEntity> {
    let vertices = vec![v1, v2, v3];

    let texcoords = vec![vec![
        TexCoord::new(0., 0.),
        TexCoord::new(0., 1.),
        TexCoord::new(1., 0.),
    ]];

    let indices = vec![0, 1, 2];

    let material = SimpleMaterialDef::create2("triangle", None, false);
    let geometry = Geometry::new(&vertices, None, &texcoords, indices, material);

    let entity = ComRc::<IEntity>::from_object(CoreEntity::new("triangle".to_string(), true));
    let mesh_component =
        StaticMeshComponent::new(entity.clone(), vec![geometry], component_factory);
    entity.add_component(
        IStaticMeshComponent::uuid(),
        ComRc::from_object(mesh_component),
    );

    entity
}

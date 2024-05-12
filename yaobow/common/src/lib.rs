pub mod read_ext;
pub mod store_ext;

use std::io::{Read, Seek};

pub trait SeekRead: Read + Seek {}
impl<T> SeekRead for T where T: Read + Seek {}

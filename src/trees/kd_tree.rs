use std::f64::{NEG_INFINITY, INFINITY};

#[derive(Debug)]
pub struct KDTree<'a, T: 'a> {
	// point
	point: Option<T>,

	// children
	left: Option<&'a mut KDTree<'a, T>>,
	right: Option<&'a mut KDTree<'a, T>>,

	// data
	range_min: f64,
	range_max: f64,
	depth: usize,
	capacity: usize,
	size: usize,
	left_size: usize,
	right_size: usize,
	point_of_split: Option<f64>,
	axis_of_split: Option<usize>,
}


impl<'a, T> KDTree<'a, T> {
	pub fn new(dimensions: usize) -> KDTree<'a, T> {
		KDTree { point: None,
				 left: None,
				 right: None,
				 range_min: NEG_INFINITY,
				 range_max: INFINITY,
				 depth: dimensions,
				 capacity: 64,
				 size: 0,
				 left_size: 0,
				 right_size: 0,
				 point_of_split: None,
				 axis_of_split: None,
		}
	}

	pub fn root(&self) -> T {
		let mut out: T;
		match self.point {
			Some(val) => out = val,
			None => panic!("Nothing in the root!"),
		}

		out
	}
}
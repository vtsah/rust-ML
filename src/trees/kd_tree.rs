use std::f64::{NEG_INFINITY, INFINITY};
use linsys::vector::Vector;

#[derive(Debug)]
pub struct KDTree<'a, T: 'a> {
	// point
	class: Option<T>,
	point: Option<Vector<f64>>,

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


impl<'a, T: Clone> KDTree<'a, T> {
	pub fn new(dimensions: usize) -> KDTree<'a, T> {
		KDTree { class: None,
				 point: None,
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

	pub fn size(&self) -> usize {
		self.size
	}

	pub fn capacity(&self) -> usize {
		self.capacity
	}

	pub fn depth(&self) -> usize {
		self.depth
	}

	pub fn left_size(&self) -> usize {
		self.left_size
	}

	pub fn right_size(&self) -> usize {
		self.right_size
	}

	// pub fn insert(&mut self, class: T, point: Vector<f64>) {
	// 	let matcher: Option<T> = self.class.clone();
	// 	match matcher {
	// 		Some(val) => {  if *point.get(self.axis_of_split.unwrap()) > self.point_of_split.unwrap() {
	// 							self.right.unwrap().insert(class, point);
	// 						} else {
	// 							self.left.unwrap().insert(class, point);
	// 						}
	// 					 },
	// 		None => { self.class = Some(class);
	// 				  self.point = Some(point);
	// 				  self.axis_of_split = Some(0);
	// 				  self.point_of_split = Some(*self.point.unwrap().get(0));
	// 				},
	// 	}
	// }

	pub fn insert(&mut self, class: T, point: Vector<f64>) {
		let empty: bool;
		let left_right: bool;
		match self.class {
			Some(_) => empty = false,
			None => empty = true,
		}

		if empty {
			let value: f64 = *point.get(0);
			self.class = Some(class.clone());
			self.point = Some(point.clone());
			self.size = 1;
			self.point_of_split = Some(value);
			self.axis_of_split = Some(0);
		} else {
			let value: f64 = *point.get(0);
			if value < self.point_of_split.unwrap() {
				match self.left {
					Some(tree) => tree.insert(class, point),
					None => { self.left = Some(KDTree::new(self.depth)); self.left.insert_helper(class, point, 1); },
				}
			} else {
				match self.right {
					Some(tree) => tree.insert(class, point),
					None => { self.right = Some(KDTree::new(self.depth)); self.right.insert_helper(class, point, 1); },
				}
			}
		}
	}

	fn insert_helper(&mut self, class: T, point: Vector<f64>, dim: usize) {
		let empty: bool;
		let bottom: bool;
		match self.class {
			Some(_) => empty = false,
			None => empty = true,
		}
		if dim == self.depth {
			bottom = true;
		}

		if empty && !bottom {
			let value: f64 = *point.get(dim);
			self.class = Some(class.clone());
			self.point = Some(point.clone());
			self.size = 1;
			self.point_of_split = Some(value);
			self.axis_of_split = Some(dim);
		} else if !empty && !bottom {
			let value: f64 = *point.get(dim);
			if value < self.point_of_split.unwrap() {
				match self.left {
					Some(tree) => tree.insert(class, point),
					None => { self.left = Some(KDTree::new(self.depth)); self.left.insert_helper(class, point, 1); },
				}
			} else {
				match self.right {
					Some(tree) => tree.insert(class, point),
					None => { self.right = Some(KDTree::new(self.depth)); self.right.insert_helper(class, point, 1); },
				}
			}
		} else {
			holder.unwrap().push((class, point));
		}
	}
}
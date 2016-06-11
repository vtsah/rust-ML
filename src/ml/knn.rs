// use trees::kd_tree::KDTree;
use linsys::matrix::Matrix;
use linsys::vector::Vector;
use std::vec::Vec;

pub struct KNN {
	k: usize,
	data: Option<Matrix<f64>>,
	classes: Option<Vec<String>>,
	// tree: KDTree<'a, String>,
	dimensions: usize,
}

fn distance_mat(vec: Vector<f64>, col: usize, mat: Matrix<f64>) -> f64 {
	let mut sum = 0f64;
	for i in 0..vec.dim() {
		sum = sum + (vec.get(i) - mat.get(i, col)) * (vec.get(i) - mat.get(i, col));
	}

	sum
}

fn distance(vec1: Vector<f64>, vec2: Vector<f64>) -> f64 {
	let mut sum = 0f64;
	for i in 0..vec1.dim() {
		sum = sum + (vec1.get(i) - vec2.get(i)) * (vec1.get(i) - vec2.get(i));
	}

	sum
}

impl KNN {
	pub fn new(k: usize, dim: usize) -> KNN {
		// KNN { k: k, tree: KDTree::new(dim), dimensions: dim }
		KNN { k: k, data: None, classes: None, dimensions: dim }
	}

	pub fn k(&self) -> usize {
		self.k
	}

	// pub fn data(&self) -> Option<Matrix<f64>> {
	// 	match self.data {
	// 		None => None,
	// 		Some(matrix) => Some(matrix),
	// 	}
	// }

	// pub fn classes(&self) -> Option<Vec<String>> {
	// 	self.classes
	// }

	// pub fn class_list(&self) -> Option<Vec<String>> {
	// 	match self.classes {
	// 		Some(list) => list.sort().dedup(),
	// 		None => None,
	// 	}
	// }

	pub fn dim(&self) -> usize {
		self.dimensions
	}

	pub fn train(&mut self, classes: Vec<String>, points: Matrix<f64>) {
		self.classes = Some(classes);
		self.data = Some(points);
	}

	pub fn test(&self, points: Matrix<f64>) -> Vec<String> {
		let mut out: Vec<String> = Vec::new();

		for column in 0..points.cols() {
			out.push(self.nearest_classes(&points.col_to_vector(column)));
		}

		out
	}

	pub fn nearest_classes(&self, point: &Vector<f64>) -> String {
		let mut distances: Vec<(f64, String)> = Vec::new();
		let mut curr_dist: f64;
		// let mut curr_class: String = String::new();

		if self.classes == None {
			panic!("No KNN training classifications assigned!");
		}

		if self.data == None {
			panic!("No KNN training data assigned!");
		}

		for i in 0..self.data.clone().unwrap().cols() {
			curr_dist = distance(point.clone(), self.data.clone().unwrap().col_to_vector(i));
			for j in 0..distances.len() {
				let (val, _): (f64, String) = distances.get(0).unwrap().clone();
				if curr_dist > val {
					distances.insert(j, (curr_dist, self.classes.clone().unwrap().get(i).unwrap().clone()));
					break;
				}
			}
		}

		let (_, ret): (f64, String) = distances.get(0).unwrap().clone();
		ret
	}
}
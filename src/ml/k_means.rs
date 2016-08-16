use linsys::matrix::Matrix;
use linsys::vector::Vector;
use rand::{thread_rng, sample};
use std::vec::Vec;
use std::string::String;

#[derive(Debug,PartialEq)]
pub struct KMeans {
	k: usize,
	iterations: usize,
	means: Matrix<f64>,
	classes: Vec<String>,
}

fn distance(vec1: Vector<f64>, vec2: Vector<f64>) -> f64 {
	let mut sum = 0f64;
	for i in 0..vec1.dim() {
		sum = sum + (vec1.get(i) - vec2.get(i)) * (vec1.get(i) - vec2.get(i));
	}

	sum
}

fn centroid(columns: Vec<usize>, training: &Matrix<f64>) -> Vector<f64> {
	let mut center: Vector<f64> = Vector::zeroes(training.rows());

	for i in 0..columns.len() {
		center.add_in_place(&training.col_to_vector(columns[i]));
	}

	center
}

impl KMeans {
	pub fn new(k: usize, iter: usize) -> KMeans {
		if k < 2{
			panic!("k must be at least two!");
		}

		KMeans { k: k, iterations: iter, means: Matrix::empty(), classes: Vec::new() }
	}

	pub fn k(&self) -> usize {
		self.k
	}

	pub fn iterations(&self) -> usize {
		self.iterations
	}

	pub fn means(&self) -> &Matrix<f64> {
		&self.means
	}

	pub fn set_classes(&mut self, class_list: Vec<String>) {
		self.classes = class_list;
	}

	pub fn train(&mut self, training: &Matrix<f64>) {
		self.initialize(training);
		for _ in 0..self.iterations {
			self.update_means(training);
		}
	}

	fn initialize(&mut self, training: &Matrix<f64>) {
		let mut rng = thread_rng();
		let cols = sample(&mut rng, 0..training.cols(), self.k);
		let mut means = Matrix::from_vector(training.col_to_vector(cols[0]));

		for i in 1..self.k {
			means.append_column(cols[i], training);
		}

		self.means = means;
	}

	fn update_means(&mut self, training: &Matrix<f64>) {
		let mut holder: Vec<Vec<usize>> = Vec::new();
		for _ in 0..self.k {
			holder.push(Vec::<usize>::new());
		}

		for i in 0..training.cols() {
			holder[self.get_closest_mean(i, training)].push(i);
		}

		let mut means_matrix: Matrix<f64> = Matrix::from_vector(centroid(holder[0].clone(), training));
		for i in 1..holder.len() {
			means_matrix.append_vector(centroid(holder[i].clone(), training));
		}

		self.means = means_matrix;
	}

	fn get_closest_mean(&self, col: usize, training: &Matrix<f64>) -> usize {
		let column = training.col_to_vector(col);
		let mut min_distance = distance(self.means.col_to_vector(0), column.clone());
		let mut closest_mean = 0;
		for current_mean in 1..self.means.cols() {
			let new_distance: f64 = distance(self.means.col_to_vector(current_mean), column.clone());
			if new_distance < min_distance {
				min_distance = new_distance;
				closest_mean = current_mean;
			}
		}

		closest_mean
	}

	pub fn test(&self, testing: &Matrix<f64>) -> Vec<usize> {
		let mut result: Vec<usize> = Vec::new();
		for i in 0..testing.cols() {
			result.push(self.nearest_mean(testing.col_to_vector(i)));
		}

		result
	}

	pub fn test_string_classes(&self, testing: &Matrix<f64>) ->Vec<String> {
		let mut result: Vec<String> = Vec::new();
		for i in 0..testing.cols() {
			let mean = self.nearest_mean(testing.col_to_vector(i));
			result.push(self.get_class(mean).clone());
		}

		result
	}

	fn nearest_mean(&self, point: Vector<f64>) -> usize {
		let mut min_distance = distance(self.means.col_to_vector(0), point.clone());
		let mut closest_mean = 0;
		for current_mean in 1..self.means.cols() {
			let new_distance: f64 = distance(self.means.col_to_vector(current_mean), point.clone());
			if new_distance < min_distance {
				min_distance = new_distance;
				closest_mean = current_mean;
			}
		}

		closest_mean
	}

	pub fn get_class(&self, mean: usize) -> String {
		self.classes[mean].clone()
	}
}

mod test {
	use ml::k_means;
	use linsys::matrix;
	use linsys::vector;

	#[test]
	fn km_check_new() {
		let km = k_means::KMeans::new(4, 50);
		let check = k_means::KMeans { k: 4, iterations: 50, means: matrix::Matrix::empty(), classes: Vec::new() };
		assert_eq!(km, check);
	}

	#[test]
	#[should_panic]
	fn km_panic_new() {
		k_means::KMeans::new(1, 20);
	}

	#[test]
	fn km_check_initialize() {
		let mut km = k_means::KMeans::new(4, 50);
		km.initialize(&matrix::Matrix::zeroes(5, 5));

		print!("{:?}", km);
	}

	#[test]
	fn km_check_update_means() {
		let mut km = k_means::KMeans::new(4, 50);
		let training = &matrix::Matrix::zeroes(5, 5);
		km.initialize(training);
		km.update_means(training);

		print!("{:?}", km);
	}

	#[test]
	fn km_check_train() {
		let mut km = k_means::KMeans::new(4, 50);
		let training = &matrix::Matrix::zeroes(5, 5);
		km.initialize(training);
		km.train(training);

		print!("{:?}", km);
	}

	#[test]
	fn km_check_test() {
		let mut km = k_means::KMeans::new(4, 50);
		let training = &matrix::Matrix::zeroes(5, 5);
		km.initialize(training);
		km.train(training);

		print!("{:?}", km);
	}
}
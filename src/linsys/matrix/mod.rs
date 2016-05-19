use std::ops::{Add, Mul, Sub, Div};
use std::num::Zero;


#[derive(Debug,PartialEq)]
pub struct Matrix<T> {
	rows: usize,
	cols: usize,
	vals: Vec<T>,
}

impl<T> Matrix<T> {
	pub fn new(data: Vec<T>, r: usize, c: usize) -> Matrix<T> {
		if r * c != data.len() {
			panic!("Your matrix row and column sizes do not match the data size!");
		}

		Matrix { vals: data, rows: r, cols: c }
	}

	pub fn get(&self, r: usize, c: usize) -> &T {
		if r > self.rows - 1 {
			panic!("You've exceeded the number of rows in the matrix!");
		}
		if c > self.cols - 1 {
			panic!("You've exceeded the number of columns in the matrix!");
		}

		let index = r * (self.cols) + c;
		&self.vals[index]
	}

	pub fn set(&mut self, value: T, r: usize, c: usize) {
		if r > self.rows - 1 {
			panic!("You've exceeded the number of rows in the matrix!");
		}
		if c > self.cols - 1 {
			panic!("You've exceeded the number of columns in the matrix!");
		}

		let index = r * (self.cols) + c;
		&self.vals.remove(index);
		&self.vals.insert(index, value);
	}

	fn set_index(&mut self, value: T, index: usize) {
		if index > (self.rows * self.cols - 1) {
			panic!("You've exceeded the number of entries in the matrix!");
		}

		&self.vals.remove(index);
		&self.vals.insert(index, value);
	}

	pub fn rows(&self) -> usize {
		self.rows
	}

	pub fn cols(&self) -> usize {
		self.cols
	}

	pub fn data(&self) -> &Vec<T> {
		&self.vals
	}

	pub fn mut_data(&mut self) -> &mut Vec<T> {
		&mut self.vals
	}
}

impl<T: Zero> Matrix<T> {
	pub fn zeroes(r: usize, c: usize) -> Matrix<T> {
		if r < 1 {
			panic!("The number of rows must be a positive integer!");
		}
		if c < 1 {
			panic!("The number of columns must be a positive integer!");
		}

		let size = r * c;
		let mut zero_mat: Vec<T> = Vec::with_capacity(size);
		for _ in 0..size {
			zero_mat.push(T::zero());
		}

		Matrix { rows: r, cols: c, vals: zero_mat }
	}
}

impl <T: Clone + Add<T, Output = T>> Matrix<T> {
	pub fn add(&self, v: &Matrix<T>) -> Matrix<T> {
		if self.rows() != v.rows() {
			panic!("Matrixs need to have the same number of rows to add!");
		}
		if self.cols() != v.cols() {
			panic!("Matrixs need to have the same number of cols to add!");
		}

		let mut out: Matrix<T> = v.clone();
		let count = self.rows * self.cols;
		for i in 0..count {
			out.set_index(self.vals[i].clone() + v.vals[i].clone(), i);
		}

		out
	}
}

impl <T: Clone + Sub<T, Output = T>> Matrix<T> {
	pub fn sub(&self, v: &Matrix<T>) -> Matrix<T> {
		if self.rows() != v.rows() {
			panic!("Matrixs need to have the same number of rows to add!");
		}
		if self.cols() != v.cols() {
			panic!("Matrixs need to have the same number of cols to add!");
		}

		let mut out: Matrix<T> = v.clone();
		let count = self.rows * self.cols;
		for i in 0..count {
			out.set_index(self.vals[i].clone() - v.vals[i].clone(), i);
		}

		out
	}
}

// impl <T: Clone + Mul<T, Output = T>> Matrix<T> {
// 	pub fn negative(&self) -> Matrix<T> {
// 		let mut out: Matrix<T> = self.clone();
// 		for i in 0..self.rows {
// 			for j in 0..self.cols {
// 				let val = (-1) * self.get(i, j);
// 				out.set(val, i, j);
// 			}
// 		}

// 		out
// 	}
// }

impl <T: Clone + Zero + Add<T, Output = T> + Mul<T, Output = T>> Matrix<T> {
	pub fn mat_mul(&self, v: &Matrix<T>) -> Matrix<T> {
		if self.cols() != v.rows() {
			panic!("Matrix1 row size and Matrix2 col size need to be the same to multiply!");
		}

		let mut out: Matrix<T> = Matrix::zeroes(self.rows, v.cols);
		for i in 0..self.rows {
			for j in 0..v.cols {
				let mut sum: T = T::zero();
				for k in 0..self.cols {
					let v1: T = self.vals[i * self.cols + k].clone();
					let v2: T = v.vals[k * v.cols + j].clone();
					sum = sum + v1 * v2;
				}
				out.set(sum, i, j);
			}
		}

		out
	}
}

impl <T: Clone + Mul<T, Output = T>> Matrix<T> {
	pub fn scalar_mul(&self, v: T) -> Matrix<T> {
		if self.rows() * self.cols() < 1 {
			panic!("Matrix needs to be of dimension at least one to scalar multiply!");
		}

		let mut out: Matrix<T> = self.clone();
		let index = self.rows * self.cols;
		for i in 0..index {
			out.set_index(v.clone() * self.vals[i].clone(), i);
		}

		out
	}
}

impl<T: Clone> Clone for Matrix<T> {
	fn clone(&self) -> Matrix<T> {
		Matrix { rows: self.rows, cols: self.cols, vals: self.vals.clone() }
	}
}

mod test {
	use super::Matrix;

	#[test]
	#[should_panic]
	#[allow(unused_variables)]
	fn panic_new() {
		let mat = Matrix::new(vec![8, 3, 9, 1], 2, 1);
	}

	#[test]
	fn check_size() {
		let mat = Matrix::new(vec![8, 3, 4, 9, 2, 1], 2, 3);
		let rows: usize = 2;
		let cols: usize = 3;
		assert_eq!(mat.rows(), rows);
		assert_eq!(mat.cols(), cols);
	}

	#[test]
	fn check_zero() {
		let mat: Matrix<i32> = Matrix::zeroes(2, 2);
		let zero = Matrix::new(vec![0, 0, 0, 0], 2, 2);

		assert_eq!(mat, zero);
	}

	// #[test]
	// fn check_dot() {
	// 	let mat1 = Matrix::new(vec![8, 3, 9]);
	// 	let mat2 = Matrix::new(vec![8, 3, 9]);

	// 	assert_eq!(mat1.dot(&mat2), 154);
	// }

	// #[test]
	// #[should_panic]
	// fn panic_dot() {
	// 	let mat1 = Matrix::new(vec![8, 3]);
	// 	let mat2 = Matrix::new(vec![8, 3, 9]);

	// 	assert_eq!(mat1.dot(&mat2), 154);
	// }

	#[test]
	fn check_add() {
		let mat1 = Matrix::new(vec![8, 3, 9, 1], 2, 2);
		let mat2 = Matrix::new(vec![1, 6, 4, 2], 2, 2);

		assert_eq!(mat1.add(&mat2), Matrix::new(vec![9, 9, 13, 3], 2, 2));
	}

	#[test]
	#[should_panic]
	fn panic_add() {
		let mat1 = Matrix::new(vec![8, 3], 2, 1);
		let mat2 = Matrix::new(vec![1, 6, 4, 2], 2, 2);

		assert_eq!(mat1.add(&mat2), Matrix::new(vec![20, 54, 13, 56], 2, 2));
	}

	#[test]
	fn check_sub() {
		let mat1 = Matrix::new(vec![8,  3, 9,  1], 2, 2);
		let mat2 = Matrix::new(vec![1,  6, 4,  2], 2, 2);
		let mat3 = Matrix::new(vec![7, -3, 5, -1], 2, 2);

		assert_eq!(mat1.sub(&mat1), Matrix::zeroes(2, 2));
		assert_eq!(mat1.sub(&mat2), mat3);
	}

	#[test]
	#[should_panic]
	fn panic_sub() {
		let mat1 = Matrix::new(vec![8, 3], 2, 1);
		let mat2 = Matrix::new(vec![1,  6, 4,  2], 2, 2);
		let mat3 = Matrix::new(vec![7, -3, 5, -1], 2, 2);

		assert_eq!(mat1.sub(&mat1), Matrix::zeroes(2, 2));
		assert_eq!(mat1.sub(&mat2), mat3);
	}

	#[test]
	fn check_mul() {
		let mat1 = Matrix::new(vec![ 8,  3,  9,  1], 2, 2);
		let mat2 = Matrix::new(vec![ 1,  6,  4,  2], 2, 2);
		let mat3 = Matrix::new(vec![20, 54, 13, 56], 2, 2);

		assert_eq!(mat1.mat_mul(&mat2), mat3);
	}

	#[test]
	#[should_panic]
	fn panic_mul() {
		let mat1 = Matrix::new(vec![ 8,  3], 2, 1);
		let mat2 = Matrix::new(vec![ 1,  6,  4,  2], 2, 2);
		let mat3 = Matrix::new(vec![20, 54, 13, 56], 2, 2);

		assert_eq!(mat1.mat_mul(&mat2), mat3);
	}

	#[test]
	fn check_scalar_mul() {
		let mat1 = Matrix::new(vec![ 8, 3,  9, 1], 2, 2);
		let mat2 = Matrix::new(vec![16, 6, 18, 2], 2, 2);
		let mat3 = Matrix::zeroes(2, 2);

		assert_eq!(mat1.scalar_mul(2), mat2);
		assert_eq!(mat1.scalar_mul(0), mat3);
	}
}
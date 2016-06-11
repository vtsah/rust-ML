use std::ops::{Add, Mul, Sub, Div};
use std::num::{Zero, One};
use linsys::vector::Vector;
use std::vec::Vec;


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

	pub fn empty() -> Matrix<T> {
		Matrix { vals: Vec::new(), rows: 0, cols: 0 }
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

	pub fn get_index(&self, index: usize) -> &T {
		if index > (self.rows * self.cols - 1) {
			panic!("You've exceeded the number of entries in the matrix!");
		}

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

impl<T: Zero + One> Matrix<T> {
	pub fn identity(n: usize) -> Matrix<T> {
		if n < 1 {
			panic!("The dimension must be a positive integer!");
		}
		let mut id: Vec<T> = Vec::with_capacity(n * n);
		for i in 0..n {
			for j in 0..n {
				if i == j {
					id.push(T::one());
				} else {
					id.push(T::zero());
				}
			}
		}

		Matrix { rows: n, cols: n, vals: id }
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

impl <T: Clone + Zero + One + Add<T, Output = T> + Sub<T, Output = T> + Mul<T, Output = T> + Div<T, Output = T> + PartialEq + PartialOrd> Matrix<T> {
	// for more efficient solution, we may be able to reduce to upper triangular and get trace
	pub fn det(&self) -> T {
		if self.rows != self.cols {
			panic!("Cannot find the determinant of a non-square matrix!");
		}

		if self.rows == 1 {
			return self.vals[0].clone();
		}

		let mut sum: T = T::zero();
		for i in 0..self.cols {
			if i % 2 == 0 {
				sum = sum + self.vals[i].clone() * self.matrix_cut(0, i).det();
			} else {
				sum = sum - self.vals[i].clone() * self.matrix_cut(0, i).det();
			}
		}

		sum
	}

	// pub fn row_elimination(&self) -> Matrix<T> {
	// 	let mut out: Matrix<T> = self.clone();
	// 	let mut min_dim = self.rows;
	// 	if self.rows > self.cols {
	// 		min_dim = self.cols;
	// 	}

	// 	for i in 0..min_dim {
	// 		let index = Matrix::argmax(&self.vals[i..min_dim]);
	// 		if *self.get(index, i) == T::zero() {
	// 			panic!("Singular matrix! Cannot eliminate!");
	// 		}

	// 		out.swap_rows(index, i);

	// 		for j in (i+1)..out.rows {
	// 			let m = *out.get(j, i) / *out.get(i, i);
	// 			for k in (i+1)..out.rows {
	// 				out.set(*out.get(j, k) - *out.get(i, k) * m, j, k);
	// 			}

	// 			out.set(T::zero(), j, i);
	// 		}
	// 	}

	// 	out
	// }

	// #[allow(dead_code)]
	// fn argmax(slice: &[T]) -> usize {
	// 	let mut max: T;
	// 	let mut max_index: usize = 0;
	// 	match slice.first() {
	// 		None => panic!("Slice is empty! Cannot find argmax!"),
	// 		Some(k) => max = k.clone(),
	// 	}

	// 	let mut index: usize = 0;

	// 	for val in slice {
	// 		if max < *val {
	// 			max = val.clone();
	// 			max_index = index;
	// 		}
	// 		index = index + 1;
	// 	}

	// 	max_index
	// }

	// fn swap_rows(&mut self, first: usize, second: usize) {

	// }
}

impl<T: Clone + Zero + One + Add<T, Output = T> + Sub<T, Output = T> + Mul<T, Output = T> + Div<T, Output = T> + PartialEq> Matrix<T> {
	/// Will use the block multiplication method to invert a matrix:
	/// | A B |^-1         |    (A - BD^-1C)^-1     -A^-1B(D - CA^-1B)^-1 |
	/// | C D |    becomes | -D^-1C(A - BD^-1C)^-1     (D - CA^-1B)^-1    |
	///
	/// | A B |^-1         |    (A - BD^-1C)^-1     -(A - BD^-1C)^-1BD^-1 |
	/// | C D |    becomes | -(D - CA^-1B)^-1CA^-1     (D - CA^-1B)^-1    |
	///
	/// So, we need to calculate five different inverses:
	/// A, C, D, A - BD^-1C, and D - CA^-1B
	/// And then it's just regular multiplication, addition, and subtraction
	pub fn inv(&self) -> Matrix<T> {
		if self.rows != self.cols {
			panic!("It isn't possible to invert a non-square matrix");
		}

		if self.rows == 1 {
			if *self.get(0, 0) == T::zero() {
				panic!("Singular matrix! Cannot compute the inverse!");
			}
			let mut vec: Vec<T> = Vec::with_capacity(1);
			vec.push(T::one() / self.get(0, 0).clone());

			return Matrix::new(vec, 1, 1);
		}

		if self.rows == 2 {
			if self.get(0, 0).clone() * self.get(1, 1).clone() == self.get(0, 1).clone() * self.get(1, 0).clone() {
				panic!("Singular matrix! Cannot compute the inverse!");
			}

			let mut vec: Vec<T> = Vec::with_capacity(4);
			let a = self.get_index(0).clone();
			let b = self.get_index(0).clone();
			let c = self.get_index(0).clone();
			let d = self.get_index(0).clone();

			vec.push(d.clone() / (a.clone() * d.clone() - b.clone() * c.clone()));
			vec.push(T::zero() - b.clone() / (a.clone() * d.clone() - b.clone() * c.clone()));
			vec.push(T::zero() - c.clone() / (a.clone() * d.clone() - b.clone() * c.clone()));
			vec.push(a.clone() / (a.clone() * d.clone() - b.clone() * c.clone()));

			return Matrix::new(vec, 2, 2);
		}

		let less_half: usize = self.rows / 2;

		let mat_a: Matrix<T> = self.section(0, 0, less_half, less_half);
		let mat_b: Matrix<T> = self.section(0, less_half + 1, less_half, self.cols);
		let mat_c: Matrix<T> = self.section(less_half + 1, 0, self.rows, less_half);
		let mat_d: Matrix<T> = self.section(less_half + 1, less_half + 1, self.rows, self.cols);

		let inv_a: Matrix<T> = mat_a.inv();
		let inv_d: Matrix<T> = mat_d.inv();
		let inv_long_a: Matrix<T> = mat_a.sub(&mat_b.mat_mul(&inv_d).mat_mul(&mat_c)).inv();
		let inv_long_d: Matrix<T> = mat_d.sub(&mat_c.mat_mul(&inv_a).mat_mul(&mat_b)).inv();

		let top_right: Matrix<T> = Matrix::zeroes(less_half, self.cols - less_half).sub(&inv_a.mat_mul(&mat_b).mat_mul(&inv_long_d));
		let bottom_left: Matrix<T> = Matrix::zeroes(self.cols - less_half, less_half).sub(&inv_d.mat_mul(&mat_c).mat_mul(&inv_long_a));

		let top: Matrix<T> = inv_long_a.augment(&top_right);
		let bottom: Matrix<T> = bottom_left.augment(&inv_long_d);

		top.augment_below(&bottom)
	}
}

impl<T: Clone + Sub<T, Output = T>> Matrix<T> {
	pub fn sub(&self, v: &Matrix<T>) -> Matrix<T> {
		if self.rows() != v.rows() {
			panic!("Matrices need to have the same number of rows to add!");
		}
		if self.cols() != v.cols() {
			panic!("Matrices need to have the same number of cols to add!");
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

	pub fn apply_vector(&self, x: Vector<T>) -> Vector<T> {
		let mut out: Vec<T> = Vec::new();
		for i in 0..self.rows {
			let mut sum: T = T::zero();
			for j in 0..self.cols {
				sum = sum + x.get(j).clone() * self.get(i, j).clone();
			}
			out.push(sum);
		}

		Vector::new(out)
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

impl <T: Clone + Zero> Matrix<T> {
	pub fn transpose(&self) -> Matrix<T> {
		let mut vec: Vec<T> = Vec::with_capacity(self.rows * self.cols);
		// let mut out: Matrix<T> = Matrix::zeroes(self.cols, self.rows);

		for i in 0..self.cols {
			for j in 0..self.rows {
				vec.push(self.get(j, i).clone());
			}
		}

		Matrix::new(vec, self.cols, self.rows)
	}

	pub fn section(&self, row_start: usize, col_start: usize, row_end: usize, col_end: usize) -> Matrix<T> {
		let mut out: Matrix<T> = Matrix::zeroes(row_end - row_start, col_end - col_start);

		for i in 0..out.rows() {
			for j in 0..out.cols() {
				out.set(self.get(row_start + i, col_start + j).clone(), i, j);
			}
		}

		out
	}

	pub fn augment(&self, left_mat: &Matrix<T>) -> Matrix<T> {
		let mut out: Matrix<T> = self.clone();

		for i in 0..left_mat.rows() {
			for j in 0..left_mat.cols() {
				out.vals.insert((i + 1) * self.cols() + j, left_mat.get(i, j).clone());
			}
		}

		out
	}

	fn augment_below(&self, beneath_mat: &Matrix<T>) -> Matrix<T> {
		let mut out: Matrix<T> = self.clone();

		let num: usize = beneath_mat.rows() * beneath_mat.cols();
		for i in 0..num {
			out.vals.push(beneath_mat.get_index(i).clone());
		}

		out
	}

	fn matrix_cut(&self, r: usize, c: usize) -> Matrix<T> {
		if r > self.rows - 1 {
			panic!("The matrix does not have that many rows!");
		}
		if c > self.cols - 1 {
			panic!("The matrix does not have that many columns!");
		}

		let mut new_data = Vec::new();
		for i in 0..self.rows {
			if i == r {
				continue;
			}

			for j in 0..self.cols {
				if j == c {
					continue;
				}

				let index = i * self.cols + j;
				new_data.push(self.vals[index].clone());
			}
		}

		Matrix::new(new_data, self.rows - 1, self.cols - 1)
	}
}

impl<T: Clone> Matrix<T> {
	pub fn from_vector(data: Vector<T>) -> Matrix<T> {
		let mut vec = Vec::new();
		for i in 0..data.dim() {
			vec.push(data.get(i).clone());
		}
		
		Matrix::new(vec, data.dim(), 1)
	}

	pub fn append_vector(&mut self, vector: Vector<T>) {
		for i in 0..vector.dim() {
			self.vals.insert((i + 1) * self.cols + i, vector.get(i).clone());
		}
	}

	pub fn append_column(&mut self, column: usize, mat: &Matrix<T>) {
		for i in 0..mat.rows() {
			self.vals.insert((i + 1) * self.cols + i, mat.get(i, column).clone());
		}
	}

	pub fn col_to_vector(&self, column: usize) -> Vector<T> {
		let mut out = Vec::new();
		for i in 0..self.rows {
			out.push(self.get_index(i * self.cols + column).clone());
		}

		let fin: Vec<T> = out;
		Vector::new(fin)
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

	#[test]
	fn check_det() {
		let mat1 = Matrix::new(vec![1, 2, -4, 3, -2, 6, -1, 2, 3], 3, 3);
		let mat2 = Matrix::new(vec![8, 3, 9, 1], 2, 2);
		let mat3 = Matrix::new(vec![3], 1, 1);
		let mat4 = Matrix::new(vec![ 6,  2,  5, -3, -2,  0,
									 5,  4,  0,  6,  3,  9,
									 2,  8,-10,  2,  1, -1,
									-3,  4,  7, -7,  5,  0,
									-2, -2,  4,  3,  0,  1,
									 1, -8,  3,  2,  1,  0], 6, 6);

		assert_eq!(mat1.det(),    -64);
		assert_eq!(mat2.det(),    -19);
		assert_eq!(mat3.det(),      3);
		assert_eq!(mat4.det(), 157732);
	}

	#[test]
	#[should_panic]
	fn panic_det() {
		let mat1 = Matrix::new(vec![1, 2, -4, 3, -2, 6], 2, 3);

		assert_eq!(mat1.det(), -64);
	}

	#[test]
	fn check_cut() {
		let mat1 = Matrix::new(vec![ 6,  2,  5, -3, -2,  0,
									 5,  4,  0,  6,  3,  9,
									 2,  8,-10,  2,  1, -1,
									-3,  4,  7, -7,  5,  0,
									-2, -2,  4,  3,  0,  1,
									 1, -8,  3,  2,  1,  0], 6, 6);
		let mat2 = Matrix::new(vec![ 6,  2,  5, -3, -2,
									 2,  8,-10,  2,  1,
									-3,  4,  7, -7,  5,
									-2, -2,  4,  3,  0,
									 1, -8,  3,  2,  1,], 5, 5);
		let mat3 = Matrix::new(vec![ 8,-10,  2,  1,
									 4,  7, -7,  5,
									-2,  4,  3,  0,
									-8,  3,  2,  1,], 4, 4);
		let mat4 = Matrix::new(vec![ 8,-10,  2,
									 4,  7, -7,
									-2,  4,  3], 3, 3);

		assert_eq!(mat1.matrix_cut(1, 5), mat2);
		assert_eq!(mat2.matrix_cut(0, 0), mat3);
		assert_eq!(mat3.matrix_cut(3, 3), mat4);
	}

	#[test]
	#[should_panic]
	fn panic_cut() {
		let mat1 = Matrix::new(vec![ 6,  2,  5, -3, -2,  0,
									 5,  4,  0,  6,  3,  9,
									 2,  8,-10,  2,  1, -1,
									-3,  4,  7, -7,  5,  0,
									-2, -2,  4,  3,  0,  1,
									 1, -8,  3,  2,  1,  0], 6, 6);
		let mat2 = Matrix::new(vec![ 6,  2,  5, -3, -2,
									 2,  8,-10,  2,  1,
									-3,  4,  7, -7,  5,
									-2, -2,  4,  3,  0,
									 1, -8,  3,  2,  1,], 5, 5);
		let mat3 = Matrix::new(vec![ 8,-10,  2,  1,
									 4,  7, -7,  5,
									-2,  4,  3,  0,
									-8,  3,  2,  1,], 4, 4);
		let mat4 = Matrix::new(vec![ 8,-10,  2,
									 4,  7, -7,
									-2,  4,  3], 3, 3);

		assert_eq!(mat1.matrix_cut(8, 5), mat2); // each should panic
		assert_eq!(mat2.matrix_cut(0, 9), mat3);
		assert_eq!(mat3.matrix_cut(3, 3), mat4);
	}
}
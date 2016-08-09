use std::ops::{Add, Mul, Sub, Div};
use std::num::{Zero, One};
use linsys::vector::Vector;
use std::vec::Vec;


#[derive(Debug,PartialEq)]
pub struct Dim<T> {
	dims: usize,
	size: usize,
	lengths: Vec<usize>,
	vals: Vec<T>,
}

impl<T> Dim<T> {
	pub fn new(data: Vec<T>, lengths: Vec<usize>) -> Dim<T> {
		if Dim::size_given_lengths(lengths) != data.len() {
			panic!("Your data does not match the size of the dim!");
		}

		let num_dimensions: usize = lengths.len();
		let size = Dim::size_given_lengths(lengths);

		Dim { vals: data, lengths: lengths.clone(), dims: num_dimensions, size: size }
	}

	pub fn empty() -> Dim<T> {
		Dim { vals: Vec::new(), lengths: Vec::with_capacity(0), dims: 0 as usize, size: 0 as usize }
	}

	pub fn get(&self, index: Vec<usize>) -> &T {
		for i in 0..self.dims {
			if index[i] > self.lengths[i] {
				panic!("Index out of bounds!");
			}
		}

		let mut index = 0;
		for axis in (0..self.dims).rev() {
			if axis != self.dims {
				index = index * self.lengths[axis + 1];
			}
			index = index + self.lengths[axis];
		}

		&self.vals[index]
	}

	pub fn get_index(&self, index: usize) -> &T {
		if index > (self.size - 1) {
			panic!("You've exceeded the number of entries in the dim!");
		}

		&self.vals[index]
	}

	pub fn set(&mut self, value: T, index: Vec<usize>) {
		for i in 0..self.dims {
			if index[i] > self.lengths[i] {
				panic!("Index out of bounds!");
			}
		}

		let mut index = 0;
		for axis in (0..self.dims).rev() {
			if axis != self.dims {
				index = index * self.lengths[axis + 1];
			}
			index = index + self.lengths[axis];
		}

		&self.vals.remove(index);
		&self.vals.insert(index, value);
	}

	fn set_index(&mut self, value: T, index: usize) {
		if index > (self.size - 1) {
			panic!("You've exceeded the number of entries in the dim!");
		}

		&self.vals.remove(index);
		&self.vals.insert(index, value);
	}

	pub fn axis_length(&self, axis: usize) -> usize {
		self.lengths[axis]
	}

	pub fn data(&self) -> &Vec<T> {
		&self.vals
	}

	pub fn mut_data(&mut self) -> &mut Vec<T> {
		&mut self.vals
	}

	pub fn size(&self) -> usize {
		self.size
	}

	pub fn dimensions(&self) -> usize {
		self.dims
	}

	pub fn dim_lengths(&self) -> Vec<usize> {
		self.lengths.clone()
	}

	pub fn size_given_lengths(lengths: Vec<usize>) -> usize {
		let mut size: usize = 0;

		for i in 0..lengths.len() {
			size = size * lengths[i];
		}

		size
	}

	pub fn position_given_index(&self, index: usize) -> Vec<usize> {
		let mut indices: Vec<usize> = Vec::with_capacity(self.dims);
		let mut size = self.size;

		for i in 0..self.dims {
			indices.push(size % self.lengths[i]);
			size = size / self.lengths[i];
		}

		indices[index]
	}

	pub fn differential_size_given_lengths(starts: Vec<usize>, ends: Vec<usize>) -> usize {
		let mut size: usize = 0;

		for i in 0..lengths.len() {
			size = size * (ends[i] - starts[i]);
		}

		size
	}
}

impl<T: PartialOrd + PartialEq> Dim<T> {
	pub fn max(&self, min_index: Vec<usize>, max_index: Vec<usize>) -> T {
		let mut min: T = self.get(min_index);

		let count: usize = differential_size_given_lengths(min_index, max_index);

		for i in 0..count {
			
		}
	}
}

impl<T: Zero> Dim<T> {
	pub fn zeroes(lengths: Vec<usize>) -> Dim<T> {
		for i in 0..lengths.len() {
			if lengths[i] < 1 {
				panic!("The {}-th dimension must be a positive integer!", i);
			}
		}

		let size: usize = Dim::size_given_lengths(lengths);
		let num_dimensions: usize = lengths.len();
		let mut zero_mat: Vec<T> = Vec::with_capacity(size);
		for _ in 0..size {
			zero_mat.push(T::zero());
		}

		Dim { vals: zero_mat, lengths: lengths, dims: num_dimensions, size: size }
	}
}

impl <T: Clone + Add<T, Output = T>> Dim<T> {
	pub fn add(&self, v: &Dim<T>) -> Dim<T> {
		for i in 0..self.dims {
			if self.lengths[i] != v.axis_length(i) {
				panic!("Mismatch in {}-th dimension! Cannot add!", i);
			}
		}

		let mut out: Dim<T> = v.clone();
		let count = self.size;
		for i in 0..count {
			out.set_index(self.vals[i].clone() + v.vals[i].clone(), i);
		}

		out
	}
}

impl<T: Clone + Sub<T, Output = T>> Dim<T> {
	pub fn sub(&self, v: &Dim<T>) -> Dim<T> {
		for i in 0..self.dims {
			if self.lengths[i] != v.axis_length(i) {
				panic!("Mismatch in {}-th dimension! Cannot add!", i);
			}
		}

		let mut out: Dim<T> = v.clone();
		let count = self.size;
		for i in 0..count {
			out.set_index(self.vals[i].clone() - v.vals[i].clone(), i);
		}

		out
	}
}

impl <T: Clone + Mul<T, Output = T>> Dim<T> {
	pub fn scalar_mul(&self, v: T) -> Dim<T> {
		if self.size < 1 {
			panic!("Dim needs to be non-empty to scalar multiply!");
		}

		let mut out: Dim<T> = self.clone();
		for i in 0..self.size {
			out.set_index(v.clone() * self.vals[i].clone(), i);
		}

		out
	}
}

// impl <T: Clone + Zero> Dim<T> {
// 	pub fn section(&self, starts: Vec<usize>, ends: Vec<usize>) -> Dim<T> {
// 		let mut new_lengths: Vec<usize> = Vec::with_capacity(starts.len());
// 		for i in 0..starts.len() {
// 			new_lengths.push(ends[i] - starts[i]);
// 		}
// 		let mut out: Dim<T> = Dim::zeroes(new_lengths);

// 		for i in 0..out.size() {
// 			let index: Vec<usize> = self.get_position(i, starts, ends);
// 			out.set_index(self.get_index(index).clone());
// 		}

// 		out
// 	}

// 	fn get_position(&self, index: usize, starts: Vec<usize>, ends: Vec<usize>) -> Vec<usize> {
// 		let mut holder: usize = 0;
// 		let mut out: Vec<usize> = Vec::with_capacity(starts.len());

// 		for axis in 0..self.dims {
			
// 		}
// 	}

// 	pub fn augment(&self, left_mat: &Dim<T>) -> Dim<T> {
// 		let mut out: Dim<T> = self.clone();

// 		for i in 0..left_mat.rows() {
// 			for j in 0..left_mat.cols() {
// 				out.vals.insert((i + 1) * self.cols() + j, left_mat.get(i, j).clone());
// 			}
// 		}

// 		out
// 	}

// 	fn augment_below(&self, beneath_mat: &Dim<T>) -> Dim<T> {
// 		let mut out: Dim<T> = self.clone();

// 		let num: usize = beneath_mat.rows() * beneath_mat.cols();
// 		for i in 0..num {
// 			out.vals.push(beneath_mat.get_index(i).clone());
// 		}

// 		out
// 	}

// 	fn matrix_cut(&self, r: usize, c: usize) -> Dim<T> {
// 		if r > self.rows - 1 {
// 			panic!("The matrix does not have that many rows!");
// 		}
// 		if c > self.cols - 1 {
// 			panic!("The matrix does not have that many columns!");
// 		}

// 		let mut new_data = Vec::new();
// 		for i in 0..self.rows {
// 			if i == r {
// 				continue;
// 			}

// 			for j in 0..self.cols {
// 				if j == c {
// 					continue;
// 				}

// 				let index = i * self.cols + j;
// 				new_data.push(self.vals[index].clone());
// 			}
// 		}

// 		Matrix::new(new_data, self.rows - 1, self.cols - 1)
// 	}
// }

// impl<T: Clone> Dim<T> {
// 	pub fn from_vector(data: Vector<T>) -> Dim<T> {
// 		let mut vec = Vec::new();
// 		for i in 0..data.dim() {
// 			vec.push(data.get(i).clone());
// 		}
		
// 		Dim::new(vec, data.dim(), vec![1])
// 	}

// 	pub fn append_vector(&mut self, vector: Vector<T>) {
// 		for i in 0..vector.dim() {
// 			self.vals.insert((i + 1) * self.cols + i, vector.get(i).clone());
// 		}
// 	}

// 	pub fn append_column(&mut self, column: usize, mat: &Dim<T>) {
// 		for i in 0..mat.rows() {
// 			self.vals.insert((i + 1) * self.cols + i, mat.get(i, column).clone());
// 		}
// 	}

// 	pub fn col_to_vector(&self, axis: usize, column: Vec<usize>) -> Vector<T> {
// 		let mut out = Vec::new();

// 		// calculate start point and jump length to get each successive value
// 		for i in 0..self.lengths[axis] {
// 			out.push(self.get_index(i * self.cols + column).clone());
// 		}

// 		let fin: Vec<T> = out;
// 		Vector::new(fin)
// 	}
// }

impl<T: Clone> Clone for Dim<T> {
	fn clone(&self) -> Dim<T> {
		Dim { lengths: self.lengths.clone(), dims: self.dims, size: self.size, vals: self.vals.clone() }
	}
}
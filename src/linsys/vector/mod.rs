use std::ops::{Add, Mul, Sub, Div};
use std::num::Zero;

#[derive(Debug,PartialEq)]
pub struct Vector<T> {
	vals: Vec<T>,
	dim: usize,
}

impl<T> Vector<T> {
	pub fn new(data: Vec<T>) -> Vector<T> {
		let dimension = data.len();
		Vector { vals: data, dim: dimension }
	}

	pub fn get(&self, index: usize) -> &T {
		if index >= self.dim {
			panic!("The index given is too high! The vector is not that larger!");
		}

		&self.vals[index]
	}

	pub fn set(&mut self, value: T, index: usize) {
		&self.vals.remove(index);
		&self.vals.insert(index, value);
	}

	pub fn dim(&self) -> usize {
		self.dim
	}

	pub fn data(&self) -> &Vec<T> {
		&self.vals
	}

	pub fn mut_data(&mut self) -> &Vec<T> {
		&mut self.vals
	}
}

impl<T: Zero> Vector<T> {
	pub fn zero(size: usize) -> Vector<T> {
		if size < 1 {
			panic!("The size of the vector must be a positive integer!");
		}

		let mut zero_vec: Vec<T> = Vec::with_capacity(size);
		for _ in 0..size {
			zero_vec.push(T::zero());
		}

		Vector { dim: size, vals: zero_vec }
	}
}

impl<T: Copy + Zero + Mul<T, Output = T> + Add<T, Output = T>> Vector<T> {
	pub fn dot(&self, v: &Vector<T>) -> T {
		if self.dim() != v.dim() {
			panic!("Vectors need to be of the same dimension to dot together!");
		}

		let mut sum: T = T::zero();
		for i in 0..self.dim {
			sum = sum + self.vals[i] * v.vals[i];
		}

		sum
	}
}

impl <T: Clone + Add<T, Output = T>> Vector<T> {
	pub fn add(&self, v: &Vector<T>) -> Vector<T> {
		if self.dim() != v.dim() {
			panic!("Vectors need to be of the same dimension to add!");
		}

		let mut out: Vector<T> = v.clone();
		for i in 0..self.dim {
			out.set(self.vals[i].clone() + v.vals[i].clone(), i);
		}

		out
	}
}

impl <T: Clone + Sub<T, Output = T>> Vector<T> {
	pub fn sub(&self, v: &Vector<T>) -> Vector<T> {
		if self.dim() != v.dim() {
			panic!("Vectors need to be of the same dimension to subtract!");
		}

		let mut out: Vector<T> = v.clone();
		for i in 0..self.dim {
			out.set(self.vals[i].clone() - v.vals[i].clone(), i);
		}

		out
	}
}

impl <T: Clone + Mul<T, Output = T>> Vector<T> {
	pub fn mul(&self, v: &Vector<T>) -> Vector<T> {
		if self.dim() != v.dim() {
			panic!("Vectors need to be of the same dimension to multiply!");
		}

		let mut out: Vector<T> = v.clone();
		for i in 0..self.dim {
			out.set(self.vals[i].clone() * v.vals[i].clone(), i);
		}

		out
	}
}

impl <T: Clone + Mul<T, Output = T>> Vector<T> {
	pub fn scalar_mul(&self, v: T) -> Vector<T> {
		if self.dim() < 1 {
			panic!("Vector needs to be of dimension at least one to scalar multiply!");
		}

		let mut out: Vector<T> = self.clone();
		for i in 0..self.dim {
			out.set(v.clone() * self.vals[i].clone(), i);
		}

		out
	}
}

impl<T: Clone> Clone for Vector<T> {
	fn clone(&self) -> Vector<T> {
		Vector { dim: self.dim, vals: self.vals.clone() }
	}
}

mod test {
	use super::Vector;

	#[test]
	fn check_size() {
		let vec = Vector::new(vec![8, 3, 9]);
		let dimension: usize = 3;
		assert_eq!(vec.dim(), dimension);
	}

	#[test]
	fn check_zero() {
		let vec: Vector<i32> = Vector::zero(3);
		let zero = Vector::new(vec![0, 0, 0]);

		assert_eq!(vec, zero);
	}

	#[test]
	fn check_dot() {
		let vec1 = Vector::new(vec![8, 3, 9]);
		let vec2 = Vector::new(vec![8, 3, 9]);

		assert_eq!(vec1.dot(&vec2), 154);
	}

	#[test]
	#[should_panic]
	fn panic_dot() {
		let vec1 = Vector::new(vec![8, 3]);
		let vec2 = Vector::new(vec![8, 3, 9]);

		assert_eq!(vec1.dot(&vec2), 154);
	}

	#[test]
	fn check_add() {
		let vec1 = Vector::new(vec![8, 3, 9]);
		let vec2 = Vector::new(vec![8, 3, 9]);

		assert_eq!(vec1.add(&vec2), Vector::new(vec![16, 6, 18]));
	}

	#[test]
	#[should_panic]
	fn panic_add() {
		let vec1 = Vector::new(vec![8, 3]);
		let vec2 = Vector::new(vec![8, 3, 9]);

		assert_eq!(vec1.add(&vec2), Vector::new(vec![16, 6, 18]));
	}

	#[test]
	fn check_sub() {
		let vec1 = Vector::new(vec![8,  3, 9]);
		let vec2 = Vector::new(vec![8,  3, 9]);
		let vec3 = Vector::new(vec![1,  4, 3]);
		let vec4 = Vector::new(vec![7, -1, 6]);

		assert_eq!(vec1.sub(&vec2), Vector::zero(3));
		assert_eq!(vec1.sub(&vec3), vec4);
	}

	#[test]
	#[should_panic]
	fn panic_sub() {
		let vec1 = Vector::new(vec![8, 3]);
		let vec2 = Vector::new(vec![1, 4, 3]);
		let vec3 = Vector::new(vec![7, -1, 6]);

		assert_eq!(vec1.sub(&vec2), vec3);
	}

	#[test]
	fn check_mul() {
		let vec1 = Vector::new(vec![ 8, 3, 9]);
		let vec2 = Vector::new(vec![ 3, 2, 1]);
		let vec3 = Vector::new(vec![24, 6, 9]);
		let vec4 = Vector::zero(3);

		assert_eq!(vec1.mul(&vec2), vec3);
		assert_eq!(vec1.mul(&vec4), Vector::zero(3));
	}

	#[test]
	#[should_panic]
	fn panic_mul() {
		let vec1 = Vector::new(vec![ 8, 3]);
		let vec2 = Vector::new(vec![ 3, 2, 1]);
		let vec3 = Vector::new(vec![24, 6, 9]);

		assert_eq!(vec1.mul(&vec2), vec3);
	}

	#[test]
	fn check_scalar_mul() {
		let vec1 = Vector::new(vec![ 8, 3,  9]);
		let vec2 = Vector::new(vec![16, 6, 18]);
		let vec3 = Vector::zero(3);

		assert_eq!(vec1.scalar_mul(2), vec2);
		assert_eq!(vec1.scalar_mul(0), vec3);
	}
}
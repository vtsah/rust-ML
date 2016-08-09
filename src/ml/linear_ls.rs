use linsys::matrix::Matrix;
use linsys::vector::Vector;
use std::vec::Vec;

pub struct LLS {
	coeff: Vector<f64>,
	dim: usize,
}

impl LLS {
	pub fn coefficients(&self) -> Vector<f64> {
		self.coeff.clone()
	}

	pub fn dim(&self) -> usize {
		self.dim
	}

	pub fn train(input: Matrix<f64>, vec: Vector<f64>) -> LLS {
		let mat: Matrix<f64> = input.transpose();
		let product: Matrix<f64> = mat.clone().transpose().mat_mul(&mat.clone()).inv().mat_mul(&mat.clone().transpose());
		let b: Vector<f64> = product.apply_vector(vec);
		let dimension: usize = mat.rows();

		LLS { coeff: b, dim: dimension }
	}

	pub fn test(&self, mat: Matrix<f64>) -> Vec<f64> {
		let mut out: Vec<f64> = Vec::with_capacity(mat.cols());

		for i in 0..mat.cols() {
			out.push(self.coeff.dot(&mat.col_to_vector(i)));
		}

		out
	}
}
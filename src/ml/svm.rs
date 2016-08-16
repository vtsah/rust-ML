use linsys::matrix::Matrix;
use linsys::vector::Vector;
use std::vec::Vec;

pub struct SVM {
	dim: usize,
	reg_parameter: usize,
	tolerance: f64,
	pass_limit: usize,
	data: Matrix<f64>,
	multipliers: Vec<f64>,
	threshold: f64,
}

impl SVM {
	pub fn train(dimensions: usize, reg_parameter: usize, tolerance: f64, max_passes: usize, data: Matrix<f64>, classes: Vec<String>) -> SVM {
		let mut multipliers: Vec<f64> = Vec::with_capacity(dimensions);
		for _ in 0..dimensions {
			multipliers.push(0f64);
		}
		let mut threshold: f64 = 0;
		let mut passes: usize = 0;
		let mut num_changed: usize = 0;

		while passes < max_passes {
			num_changed = 0;

			for i in 0..m {
				let y: f64 = value(data.col_to_vector(i));
				let E = classifier(data.col_to_vector(i) - y);

				if ((y * E < -tol) && (multipliers.get(i) < C)) || ((y * E > tol) && (multipliers.get(i) > 0)) {
					let j: usize = 6; // random j != i

					let y2: f64 = value(data.col_to_vector(j));
					let E2 = classifier(data.col_to_vector(j)) - y;

					let a_i: f64 = multipliers.get(i);
					let a_j: f64 = multipliers.get(j);

					let mut L: f64 = L(multipliers);
					let mut H: f64 = H(reg_parameter, multipliers);
					if y == y2 {
						L = L2(multipliers);
						H = H2(multipliers);
					}

					if L == H {
						continue;
					}

					let nu = nu(data.col_to_vector(i), data.col_to_vector(j));

					if nu >= 0f64 {
						continue;
					}

					multipliers.set(clip(recalc_j(a_j, y2, E, E2, nu)));

					if multipliers.get(j) - a_j < 0.00001 {
						continue;
					}

					multipliers.set(recalc_i(a_i. y, y2, multipliers.get(j), a_j));

					let b_1: f64 = b_1(b, E, y, y2, multipliers, a_i, a_j, mat.col_to_vector(i), mat.col_to_vector(j));
					let b_2: f64 = b_2(b, E2, y, y2, multipliers, a_i, a_j, mat.col_to_vector(i), mat.col_to_vector(j));
					threshold =  balance(b_1, b_2);

					num_changed = num_changed + 1;
				}
			}

			if num_changed == 0 {
				passes = passes + 1;
			} else {
				passes = 0;
			}
		}

		SVM { dim: dimensions, reg_parameter: reg_parameter, tolerance: tolerance, pass_limit: max_passes, data: Matrix<f64>, multipliers: multipliers, threshold: b }
	}

	pub fn test(input: Vector<f64>) -> f64 {
		let sum: f64 = 0f64;
		for i in 0..self.dim {
			sum = sum + self.multipliers.get(i) * value(self.data.col_to_vector(i)) * input.dot(self.data.col_to_vector(i)) + self.threshold;
		}

		sum
	}

	fn value(input: Vector<f64>) -> f64 {
		input.get(0)
	}

	fn classifier(&self, input: Vector<f64>) -> f64 {
		input.dot(self.w) + self.threshold
	}

	fn L(multipliers: Vec<f64>, i: usize, j: usize) -> f64 {
		max(0, multipliers.get(i) - multipliers.get(j))
	}

	fn H(threshold: f64, multipliers: Vec<f64>, i: usize, j: usize) -> f64 {
		min(threshold, threshold + multipliers.get(j) - multipliers.get(i))
	}

	fn L2(threshold: f64, multipliers: Vec<f64>, i: usize, j: usize) -> f64 {
		max(0, multipliers.get(i) + multipliers.get(j) - threshold)
	}

	fn H2(threshold: f64, multipliers: Vec<f64>, i: usize, j: usize) -> f64 {
		min(threshold, multipliers.get(j) + multipliers.get(i))
	}

	fn nu(mat: Matrix<f64>, i: usize, j: usize) -> f64 {
		2 * mat.col_to_vector(i).dot(mat.col_to_vector(j)) - mat.col_to_vector(i).dot(mat.col_to_vector(i)) - mat.col_to_vector(j).dot(mat.col_to_vector(j))
	}

	fn recalc_i(a_i: f64, a_j: f64, old_j: f64, y: f64, y2: f64) -> f64 {
		a_i + y * y2 * (old_j + a_j)
	}

	fn recalc_j(a_j: f64, nu: f64, E: f64, E2: f64, y: f64) -> f64 {
		a_j - (y * (E - E2)) / nu
	}

	fn clip(a_j: f64, H: f64, L: f64) -> f64 {
		if a_j > H {
			return H;
		} else if a_j < L {
			return L;
		} else {
			return a_j;
		}
		a_j
	}

	fn b_1(threshold: f64, E: f64, y: f64, a_i: f64, old_i0: f64, i: usize, y2: f64, a_j: f64, old_j: f64, j: usize, mat: Matrix<f64>) -> f64 {
		threshold - E - y * (a_i - old_i) * mat.col_to_vector(i).dot(mat.col_to_vector(i)) - y2 *(a_j - old_j) * mat.col_to_vector(i).dot(mat.col_to_vector(j))
	}

	fn b_2(threshold: f64, E2: f64, y: f64, a_i: f64, old_i0: f64, i: usize, y2: f64, a_j: f64, old_j: f64, j: usize, mat: Matrix<f64>) -> f64 {
		threshold - E2 - y * (a_i - old_i) * mat.col_to_vector(i).dot(mat.col_to_vector(j)) - y2 *(a_j - old_j) * mat.col_to_vector(j).dot(mat.col_to_vector(j))
	}

	fn balancer(b_1: f64, b_2: f64, a_i: f64, a_j: f64, threshold: f64) -> f64 {
		if 0 < a_i && a_i < threshold {
			return b_1;
		} else if 0 < a_j && a_j < threshold {
			return b_2;
		} else {
			return (b_1 + b_2) / 2;
		}

		(b_1 + b_2) / 2
	}
}

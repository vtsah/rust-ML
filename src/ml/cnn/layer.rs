enum LayerType {
	Convolution,
	Pooling,
	ReLU,
	Full,
}

struct Layer {
	type: LayerType,
	WeightsIn: Matrix<f64>,
	PrevSize: Vec<usize>,
	WeightsOut: Matrix<f64>,
	NextSize: Vec<usize>,
}

impl Layer {
	pub fn new(type: LayerType) -> Layer {
		Layer { type: type, WeightsIn: Matrix::empty(), WeightsOut: Matrix::empty() }
	}

	pub fn new_input(input_dim: Dim<f64>) -> Layer {
		Layer { type: LayerType::Input, WeightsIn: Matrix::empty(), PrevSize: 0 as usize, WeightsOut: Matrix::empty(), NextSize: 0 as usize }
	}

	pub fn new_output(input_dim: Dim<f64>, previous_layer: Layer, WeightsIn: Matrix<f64>) -> Layer {
		Layer { type: LayerType::Output, WeightsIn: WeightsIn, PrevSize: previous_layer.size(), WeightsOut: Matrix::empty(), NextSize: 0 as usize }
	}

	pub fn new_convolution(previous_layer: Layer, WeightsIn: Matrix<f64>) -> Layer {
		let new_layer: Layer = Layer { type: LayerType::Convolution, WeightsIn: WeightsIn, PrevSize: previous_layer.size(), WeightsOut: Matrix::empty(), NextSize: 0 as usize }
		previous_layer.set_next(new_layer);
		new_layer
	}

	pub fn attach_previous(&mut self, previous_layer: Matrix<f64>) {
		self.WeightsIn = previous_layer;
	}

	pub fn attach_previous(&mut self, next_layer: Matrix<f64>) {
		self.WeightsOut = next_layer;
	}

	pub fn set_next(&mut self, next_layer: Layer) {
		self.NextSize = next_layer.size();
		self.WeightsOut = next_layer.get_weights_in();
	}

	pub fn get_weights_in(&self) -> Matrix<f64> {
		self.WeightsIn.clone()
	}

	pub fn process(&self, input_dim: Dim<f64>) -> Dim<f64> {
		match self.type {
			LayerType::Convolution => self.process_convolution(input_dim),
			LayerType::Pooling => self.process_max_pool(input_dim),
			LayerType::ReLU => self.process_relu(input_dim),
			LayerType::Full => self.process_full(input_dim),
		}
	}

	pub fn process_max_pool(&self, input_dim: Dim<f64>, length: usize, step: usize) -> Dim<f64> {
		let mut dimensions: Vec<usize> = Vec::with_capacity(input_dim.dimensions());
		let in_lengths: Vec<usize> = input_dim.dim_lengths();
		for i in 0..in_lengths.len() {
			dimensions.push((in_lengths[i] - 1) / step + length - 1);
		}
		let mut out: Dim<f64> = Dim::zeroes(dimensions);

		for i in 0..input_dim.size() {
			if input_dim.position_given_index(i) % step == 0 {
				out.set_index(out.max(min_index, max_index), i);
			}
		}

		out
	}
}

impl <T: Clone + Zero> Layer {
	pub fn section(&self, starts: Vec<usize>, ends: Vec<usize>) -> Dim<T> {
		let mut new_lengths: Vec<usize> = Vec::with_capacity(starts.len());
		for i in 0..starts.len() {
			new_lengths.push(ends[i] - starts[i]);
		}
		let mut out: Dim<T> = Dim::zeroes(new_lengths);

		for i in 0..out.size() {
			let index: Vec<usize> = self.get_position(i, starts, ends);
			out.set_index(self.get_index(index).clone());
		}

		out
	}

	fn get_position(&self, index: usize, starts: Vec<usize>, ends: Vec<usize>) -> Vec<usize> {
		let mut holder: usize = 0;
		let mut out: Vec<usize> = Vec::with_capacity(starts.len());

		for axis in 0..self.dims {
			
		}
	}

	pub fn augment(&self, left_mat: &Dim<T>) -> Dim<T> {
		let mut out: Dim<T> = self.clone();

		for i in 0..left_mat.rows() {
			for j in 0..left_mat.cols() {
				out.vals.insert((i + 1) * self.cols() + j, left_mat.get(i, j).clone());
			}
		}

		out
	}

	fn augment_below(&self, beneath_mat: &Dim<T>) -> Dim<T> {
		let mut out: Dim<T> = self.clone();

		let num: usize = beneath_mat.rows() * beneath_mat.cols();
		for i in 0..num {
			out.vals.push(beneath_mat.get_index(i).clone());
		}

		out
	}

	fn cut(&self, index: vec<usize>) -> Dim<T> {

		let mut new_data = Vec::with_capacity(self.size());
		for count in 0..self.data() {
			for i in 0..self.dimensions().len() {
				if 
			}
		}

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

		Dim::new(new_data, self.dimensions())
	}
}
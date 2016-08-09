struct CNN {
	Layers: Option<Vec<Layer>>,
	NumInputDims: usize,
	InputBounds: Matrix<usize>,
	NumOutputDims: usize,
	OutputBounds: Matrix<usize>,
}

impl CNN {
	pub fn new()
}